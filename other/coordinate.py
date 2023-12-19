import cv2
import mediapipe as mp
import time
from pose_media import mediapipe_pose
import csv
import numpy as np

# mediapipe 불러오기
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils
media = mediapipe_pose()

class Coor():
    def __init__(self):
        pass
    # csv에 X{}, Y{}, Z{}을 31까지 저장
    # def save_csv(self, create_csv, use_coord):
    #     try:
    #         num_coords = len(use_coord) # num_coords: 33

    #         landmarks = ['class']
    #         for val in use_coord:
    #             landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]

    #         with open(create_csv, mode='w', newline='') as f:
    #             csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #             csv_writer.writerow(landmarks)
    #     except:
    #         print("으익")


    # 좌표를 따서 저장해주는 함수
    def record_coordinates(self, results, csv_file, class_name, use_coord):
        try:
            temp = []
            pose = results.pose_landmarks.landmark
            # pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            joint = np.zeros((33, 4))
            for j, lm in enumerate(pose):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                
            v1 = joint[[12,11,23,24,12,14,16,18,20,11,13,15,17,19], :3] # Parent joint
            v2 = joint[[11,23,24,12,14,16,18,20,16,13,15,17,19,15], :3] # Child joint
            v = v2 - v1 # [14, 3]
            
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            
            angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10],:], 
                    v[[1,2,3,5,6,7,9,10,11],:])) # [9,]
            
            angle = np.degrees(angle)
            
            angle_label = np.array([angle], dtype=np.float32)
            angle_label = np.append(angle_label, 1)
                
            land = media.extract_keypoints(results)
            d = np.concatenate([joint.flatten(), angle_label[:-1], land])

            return d
        except:
            pass

        # for i in pose_row:
        #     repo.append(round(i, 3))

        # repo.insert(0, class_name)
            

        # with open(csv_file, mode='a', newline='') as f:
            
        #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     csv_writer.writerow(repo)

        # except:
        #     print("으익")