import math


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getx(self):
        return self.x

    def gety(self):
        return self.y


def GetCross(p1, p2, p):
    return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y)

class Getlen:
  def __init__(self,p1,p2):
    self.x=p1.getx()-p2.getx()
    self.y=p1.gety()-p2.gety()
    #use math.sqrt() to get the square root 用math.sqrt（）求平方根
    self.len= math.sqrt((self.x**2)+(self.y**2))
  #define the function of getting the length of line 定义得到直线长度的函数
  def getlen(self):
    return self.len

def IsPointInMatrix(p1, p2, p3, p4, p):
    isPointIn = GetCross(p1, p2, p) * GetCross(p3, p4, p) >= 0 and GetCross(p2, p3, p) * GetCross(p4, p1, p) >= 0
    return isPointIn

def getDis(p1, p2, p3, p4, p):
    # define the object 定义对象
    l1 = Getlen(p1, p2)
    l2 = Getlen(p1, p3)
    l3 = Getlen(p2, p3)
    # get the length of two points/获取两点之间直线的长度
    d1 = l1.getlen()
    d2 = l2.getlen()
    d3 = l3.getlen()

def isInTrack(position,trackList):

    x,y=position
    pp = Point(x, y)
    for i in range(len(trackList)):
        p1 = Point(trackList[i][0][0][0],trackList[i][0][0][1])
        p2 = Point(trackList[i][0][1][0],trackList[i][0][1][1])
        p3 = Point(trackList[i][0][2][0],trackList[i][0][2][1])
        p4 = Point(trackList[i][0][3][0],trackList[i][0][3][1])
        if IsPointInMatrix(p1, p2, p3, p4, pp):

            return True


    return False