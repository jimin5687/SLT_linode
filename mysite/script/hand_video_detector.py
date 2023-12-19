import cv2
import mediapipe as mp
from mysite.coordinate import Coor
import cv2
from mysite.pose_media import mediapipe_pose
import numpy as np
from keras.models import load_model
from mysite.train_dataset import pose_landmark_dataset
import os
from pathlib import Path

model_path = Path(r"D:\SLT_Website_Django\mysite\mysite\models2\model.h5")

coor = Coor()
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils
media = mediapipe_pose()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# model_path = os.path.join('mysite', 'mysite', 'models2', 'model.h5')

seq = []
action_seq = []
seq_length = 30
action = "?"

actions = np.array(['stand', 'hello', 'happy', 'iloveyou'])
model = load_model(model_path, compile=False)

# this method is used to deduce the hand_video method
# plz refer to the hand_video method accordingly


def hand_video(flags, frame, holistic):
    # For static images:
    # parameters for the detector
    if not holistic:
        return frame, 1
    else:
        image, data = pose_landmark_dataset(frame, actions, holistic=holistic)
        return image, data

    # flip it back and return
    