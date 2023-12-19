import cv2
import mediapipe as mp
import time
from pose_media import mediapipe_pose
import csv
import numpy as np
from coordinate import Coor
import os
from keras.models import load_model
from keras import Sequential
from train_dataset import pose_landmark_dataset
import random
from PIL import ImageFont, ImageDraw, Image
from translate import return_sentence
from gtts import gTTS
import pygame

# mediapipe 불러오기
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils
media = mediapipe_pose()
coor = Coor()

cap = cv2.VideoCapture(0)

model_path = os.path.join('models3', 'model.h5')


seq = []
action_seq = []
sen = []
seq_length = 30
action = "?"
display_sen = ''
translate_sen = ''

actions = np.array(['싫어하다', '나', '짜증', '너', '사감', '우유', '무섭다', '청소당번', '가다', '지름길', '배고프다', '대박', 
                    '답답', '나이', '기분', '못생기다', '때문', '좋다', '배부르다', '친구', '똑똑하다', '밥', '빠르다', 
                    '덥다', '동대문', '춥다', '형편없다', '차다', '온도', '불복종', '오늘'])

# kor = np.array(['서다', '안녕하세요', '기쁘다', '사랑해'])
# eng_to_kor = {actions[i]:kor[i] for i in range(len(actions))}

def myPutText(src, text, pos, font_size, font_color) :
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('fonts/gulim.ttc', font_size)
    draw.text(pos, text, font=font, fill= font_color)
    return np.array(img_pil)

model = load_model(model_path, compile=False)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        img_height, img_width, _ = frame.shape

        stop_start_state = "Stop"

        img = cv2.resize(frame, (int(img_width * (400 / img_height)), 400))

        image, data = pose_landmark_dataset(frame, actions, holistic=holistic)
        
        try:
            if data == None:
                continue
        except:
            pass
        
        seq.append(data)
        
        if len(seq) < seq_length:
            continue
        
        if (data[61] < data[53] or data[65] < data[57]) and (data[55] > 0.5 and data[59] > 0.5) and (data[63] > 0.5 and data[67] > 0.5):
            
            pre_list = []
            for i in range(30):
                ret, frame = cap.read()
                image, data = pose_landmark_dataset(frame, actions, holistic=holistic)
                pre_list.append(data)

                stop_start_state = "Start"
                # cv2.putText(image, stop_start_state, org=(30, 65), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
                image = myPutText(image, stop_start_state, (30, 65), 30, (255, 0, 0))

                cv2.imshow('OpenCV', image)
                cv2.waitKey(10)

            input_data = np.expand_dims(np.array(pre_list), axis=0)
            y_pred = model.predict(input_data)
            y_pred = y_pred.squeeze(0)  
            i_pred = int(np.argmax(y_pred))
            action = actions[i_pred]
            cv2.waitKey(1000)
            sen.append(action)
            display_sen += action+' '
            start = time.time()
            
        if ((time.time() - start) > 5.5) and sen:
            translate_sen = return_sentence(sen)
            print(translate_sen)
            tts = gTTS(text=translate_sen, lang='ko')
            tts.save("result.mp3")
            pygame.mixer.init()
            pygame.mixer.music.load("result.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy()==True:
                continue
            pygame.quit()
            display_sen = ''
            print(sen)
            sen = []
            start = time.time()
            print("translation")
            
        
        if sen:
            # cv2.putText(image, str(5-int(time.time()-start)), org=(300, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            image = myPutText(image, str(5-int(time.time()-start)), (300, 30), 30, (255, 0, 0)) 
        # cv2.putText(image, f'{action.upper()}', org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)  
        image = myPutText(image, f'{action.upper()}', (30, 30), 30, (255, 0, 0))
        
        # if conf < 0.8:
        #     continue
            
        stop_start_state = "Stop"
        # cv2.putText(image, stop_start_state, org=(30, 65), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        image = myPutText(image, stop_start_state, (30, 65), 30, (255, 0, 0))
        image = myPutText(image, display_sen, (200, 20), 30, (255, 0, 0))
        image = myPutText(image, translate_sen, (100, 200), 30, (255, 0, 0))
        
        cv2.imshow('OpenCV', image)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()