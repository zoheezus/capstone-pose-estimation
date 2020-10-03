from queue import Queue

import math
import numpy as np

class PoseGeom:
  # using coco dataset

  LEFT_SHOULDER = 2
  LEFT_ELBOW = 3
  LEFT_HAND = 4
  RIGHT_SHOULDER = 5
  RIGHT_ELBOW = 6
  RIGHT_HAND = 7

  LIST_OF_JOINTS = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_HAND, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HAND]

  @classmethod
  def angle_btw_2_points(cls, joint_A, joint_B):
    # becuase y coor increases from top to bottom, we swtich to y
    # x1 y1 are the points along horizontal axis
    # https://stackoverflow.com/questions/2676719/calculating-the-angle-between-the-line-defined-by-two-points
    angle_rads = math.atan2(joint_B.y - joint_A.y, joint_B.x - joint_A.x)
    angle_rads = -angle_rads
    return math.degrees(angle_rads)
