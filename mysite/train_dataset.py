import cv2
import mediapipe as mp
import time
from mysite.pose_media import mediapipe_pose
import csv
import numpy as np
from mysite.coordinate import Coor
import os
import pandas as pd

# mediapipe 불러오기
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils
media = mediapipe_pose()
coor = Coor()

csv_path = "datasets\coords_dataset_test.csv"



start_time = time.time()

frame_counter = 0
sequences = 30
sequence_length = 30

use_coord = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# pose 랜드마크를 찍어주는 함수
def pose_landmark_dataset(img, label, holistic):
    global frame_counter

    img_height, img_width, _ = img.shape
    img = cv2.resize(img, (int(img_width * (800 / img_height)), 800))
    # 홀리스틱으로 landmark 찾아주는 함수
    image, results = media.drawing_holistic(img, holistic)


    data = coor.record_coordinates(results, csv_path, label, use_coord)
    
    # landmark draw 해주는 함수
    media.draw_styled_landmarks(image, results)

    # cv2.putText(image,"FPS:" +str(int(fps)),(10,100), cv2.FONT_HERSHEY_PLAIN, 2,(255,0,190),2,cv2.LINE_AA)
    return image, data

DATA_PATH = os.path.join('datasets3')

# data_from_excel = pd.read_excel('E:/code/need_data.xlsx')

actions = np.array(['나', '짜증', '너', '사감', '우유', '무섭다', '청소당번', '가다', '배고프다', 
                    '나이', '기분', '때문', '좋다', '배부르다', '밥', 
                    '덥다', '동대문', '춥다', '형편없다', '차다', '온도', '불복종', '오늘', '어디'])

for action in actions:
    for sequence in range(30, 60):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
# for action in actions:
#     print(data[data["name"]==action])
# Holistic 오픈
# [20:22]
# print(data[data["name"]=='나'])
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for action in [actions[0]]:
            print(action)
            cv2.waitKey(1000)
            for sequence in range(30, 31):
                for frame_num in range(sequence_length):

                    ret, img = cap.read()
                    image, data = pose_landmark_dataset(img, action, holistic)
                    
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Video', image)
                        cv2.waitKey(1000)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Video', image)
                        
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, data)    

                        
                    if not ret:
                        break
                    if cv2.waitKey(10) == ord('q'):
                        break
    cap.release()
# data = np.array(data)
# print(data.shape)
# print(data)
# print(coor)

