import argparse
import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue
import numpy as np
if __name__ == '__main__':
    recording_obs = []
    recording_action = []

    train_model = "./save/trial_400.h5"
    play_episodes = 1

    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent(epsilon=0)  # Set epsilon to 0 to ensure all actions are instructed by the agent
    agent.load(train_model)

    for e in range(play_episodes):
        init_state = env.reset()

        init_state = process_state_image(init_state)

        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1

        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            recording_action.append(action)

            next_state, reward, done, info = env.step(action)

            recording_obs.append(next_state)

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e + 1, play_episodes,
                                                                                             time_frame_counter,
                                                                                             float(total_reward)))
                break
            time_frame_counter += 1

        print(recording_obs)
        print(recording_action)

        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.float16)
        np.savez_compressed("record/data2.npz", obs=recording_obs, action=recording_action)
        print("end")
