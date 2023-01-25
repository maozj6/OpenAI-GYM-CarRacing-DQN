# for the purpose of creating visualizations

import argparse
import time
import os
import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
import numpy as np
import safe_judger

import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

import json
import sys

from dream_env import make_env
import time

from vae.vae import ConvVAE
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size

render_mode = True

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3  # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH


def make_model(load_model=True):
    # can be extended in the future.
    model = Model(load_model=load_model)
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def clip(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)


def passthru(x):
    return x


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sample(p):
    return np.argmax(np.random.multinomial(1, p))


class Model:
    ''' simple one layer model for car racing '''

    def __init__(self, load_model=True):
        self.env_name = "carracing"
        self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)

        self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

        if load_model:
            self.vae.load_json('vae/vae.json')
            self.rnn.load_json('rnn/rnn.json')

        self.state = rnn_init_state(self.rnn)
        self.rnn_mode = True

        self.input_size = rnn_output_size(EXP_MODE)
        self.z_size = 32

        if EXP_MODE == MODE_Z_HIDDEN:  # one hidden layer
            self.hidden_size = 40
            self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
            self.bias_hidden = np.random.randn(self.hidden_size)
            self.weight_output = np.random.randn(self.hidden_size, 3)
            self.bias_output = np.random.randn(3)
            self.param_count = ((self.input_size + 1) * self.hidden_size) + (self.hidden_size * 3 + 3)
        else:
            self.weight = np.random.randn(self.input_size, 3)
            self.bias = np.random.randn(3)
            self.param_count = (self.input_size) * 3 + 3

        self.render_mode = False

    def make_env(self, seed=-1, render_mode=False):
        self.render_mode = render_mode
        self.env = make_env(self.env_name, agent=self, seed=seed)

    def reset(self):
        self.state = rnn_init_state(self.rnn)

    def encode_obs(self, obs):
        # convert raw obs to z, mu, logvar
        result = np.copy(obs).astype(np.float) / 255.0
        result = result.reshape(1, 64, 64, 3)
        mu, logvar = self.vae.encode_mu_logvar(result)
        mu = mu[0]
        logvar = logvar[0]
        s = logvar.shape
        z = mu + np.exp(logvar / 2.0) * np.random.randn(*s)
        return z, mu, logvar

    def get_action(self, z):
        h = rnn_output(self.state, z, EXP_MODE)

        '''
        action = np.dot(h, self.weight) + self.bias
        action[0] = np.tanh(action[0])
        action[1] = sigmoid(action[1])
        action[2] = clip(np.tanh(action[2]))
        '''
        if EXP_MODE == MODE_Z_HIDDEN:  # one hidden layer
            h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
            action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
        else:
            action = np.tanh(np.dot(h, self.weight) + self.bias)

        action[1] = (action[1] + 1.0) / 2.0
        action[2] = clip(action[2])

        self.state = rnn_next_state(self.rnn, z, action, self.state)

        return action

    def set_model_params(self, model_params):
        if EXP_MODE == MODE_Z_HIDDEN:  # one hidden layer
            params = np.array(model_params)
            cut_off = (self.input_size + 1) * self.hidden_size
            params_1 = params[:cut_off]
            params_2 = params[cut_off:]
            self.bias_hidden = params_1[:self.hidden_size]
            self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
            self.bias_output = params_2[:3]
            self.weight_output = params_2[3:].reshape(self.hidden_size, 3)
        else:
            self.bias = np.array(model_params[:3])
            self.weight = np.array(model_params[3:]).reshape(self.input_size, 3)

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def get_random_model_params(self, stdev=0.1):
        # return np.random.randn(self.param_count)*stdev
        return np.random.standard_cauchy(self.param_count) * stdev  # spice things up

    def init_random_model_params(self, stdev=0.1):
        params = self.get_random_model_params(stdev=stdev)
        self.set_model_params(params)


def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):
    reward_list = []
    t_list = []

    max_episode_length = 1000
    recording_mode = False
    penalize_turning = False

    if train_mode and max_len > 0:
        max_episode_length = max_len

    if (seed >= 0):
        random.seed(seed)
        np.random.seed(seed)
        model.env.seed(seed)

    for episode in range(num_episode):

        model.reset()
        # tmp=model.reset()
        z = model.env.reset()
        # z=model
        for t in range(max_episode_length):

            action = model.get_action(z)

            if render_mode:
                model.env.render("human")
            else:
                model.env.render('rgb_array')

            z, reward, done, info = model.env.step(action)

            if (render_mode):
                print("action", action, "sum.square.z", np.sum(np.square(z)))

            if done:
                break

        t_list.append(t)

    return reward_list, t_list
def pretest():
    use_model = True
    filename = "log/carracing.cma.16.64.best.json"

    model = make_model()
    print('model size', model.param_count)

    model.make_env(render_mode=render_mode)


    model.reset()
    # tmp=model.reset()
    z = model.env.reset()

    path="record/out/20230118154453.npz"

    data=np.load(path)
    obs=data["obs"]

    action=data["action"]

    img=cv2.resize(obs[100],(64,64))
    zobs=model.encode_obs(img)
    fobs=model.env.decode_obs(zobs[0])

    plt.subplot(1, 3, 1)

    plt.imshow(img)
    plt.subplot(132)

    plt.imshow(fobs)

    plt.subplot(133)

    plt.imshow(obs[100])

    plt.show()

    print("end")
def predict():

    ##episodes
    play_episodes = 1
    ##which frame to stop, since the initial frames is not suitable to predict
    end_guard=100
    ##number of the frame for the prediction
    pred_num=20

    ##DQN_env
    train_model = 'save/trial_400.h5'
    # print(train_model)
    # outdir=args.output
    # if not os.path.exists(outdir+'/'):  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     os.makedirs(outdir+'/')
    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent(epsilon=0)  # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(train_model)
    dqnimg=[]
    dqnsafes=[]
    for e in range(play_episodes):
        recording_obs = []
        recording_action = []
        init_state = env.reset()
        init_state = process_state_image(init_state)
        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        ##guard
        DQNguard=0
        while True:
            DQNguard=DQNguard+1
            env.render()
            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            recording_action.append(action)
            next_state, reward, done, info = env.step(action)
            img_temp=cv2.resize(next_state, (64, 64))
            recording_obs.append(img_temp)
            total_reward += reward
            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e + 1, play_episodes,
                                                                                             time_frame_counter,
                                                                                             float(total_reward)))
                break
            # time_frame_counter += 1
            # if DQNguard==end_guard:
            #     posx, posy = info[0]
            #
            #     if safe_judger.isInTrack(info[0], info[1]) == False:
            #         e=e-1
            #         break
            #     else:
            #         initial_img=recording_obs[len(recording_obs)-1]
            #         imgs=[]
            #         safes=[]
            #         for itr in range(pred_num):
            #             env.render()
            #             current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            #             action = agent.act(current_state_frame_stack)
            #             recording_action.append(action)
            #             next_state, reward, done, info = env.step(action)
            #             imgs.append(next_state)
            #             safes.append(safe_judger.isInTrack(info[0], info[1]))
            #             recording_obs.append(next_state)
            #             total_reward += reward
            #             next_state = process_state_image(next_state)
            #             state_frame_stack_queue.append(next_state)
            #             time_frame_counter += 1
            #         dqnimg.append(imgs)
            #         dqnsafes.append(safes)
            #         break
        use_model = True
        filename = "log/carracing.cma.16.64.best.json"

        model = make_model()
        print('model size', model.param_count)

        model.make_env(render_mode=render_mode)


        model.reset()
        z = model.env.reset()
        # for i in range(end_guard):
        #     action = model.get_action(z)
        #     z, reward, done, info = model.env.step(action)

        # tmp=model.reset()
        # initial_img2 = cv2.resize(initial_img, (64, 64))
        zs=[]
        pred_imgs=[]
        recording_action=np.array(recording_action, dtype=np.float16)
        for j in range(100):
            zs.append(model.encode_obs(recording_obs[j]))
            action = model.get_action(zs[j][0])
            z, reward, done, info = model.env.step(recording_action[j])
            pred_imgs.append(model.env.decode_obs(z))


        for j in range(900):
            action = model.get_action(z)
            z, reward, done, info = model.env.step(action)
            pred_imgs.append(model.env.decode_obs(z))

    plt.subplot(2, 3,  1)
    plt.imshow(recording_obs[100])

    plt.subplot(2, 3, 2)
    plt.imshow(recording_obs[101])

    plt.subplot(2, 3, 3)
    plt.imshow(recording_obs[102])

    plt.subplot(2, 3,  4)
    plt.imshow(pred_imgs[100])

    plt.subplot(2, 3, 5)
    plt.imshow(pred_imgs[101])

    plt.subplot(2, 3, 6)
    plt.imshow(pred_imgs[102])
    plt.show()

    print("end")

    dirName="record/"+'/'+time.strftime("%Y%m%d%H%M%S", time.localtime())+'/'
    if not os.path.exists(dirName):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(dirName)
    #
    for i in range(25):

        cv2.imwrite(dirName+"original"+str(i)+".jpg",recording_obs[90+i])

        cv2.imwrite(dirName+"pred"+str(i)+".jpg",pred_imgs[90+i])


    # for i in range(pred_num/2):
    #     plt.subplot(2, pred_num/2, i+1)
    #     plt.imshow(imgs[i+pred_num/2])
    #
    #     plt.subplot(2,pred_num/2,pred_num/2+1+i)
    #
    #     plt.imshow(pred_imgs[i+pred_num])


    plt.show()

    print("end")






if __name__ == "__main__":
    predict()
