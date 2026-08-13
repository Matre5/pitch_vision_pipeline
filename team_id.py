import cv2
import csv
import numpy as np
from sklearn.cluster import KMeans

def get_torso(frame, x1, y1, x2, y2):
    height = y2 -y1
    top_torso = y1 + int(height * 0.20)
    bottom_torso = y1 + int(height * 0.55)
    
    return frame[top_torso:bottom_torso, x1:x2]

def get_dominant_hue(torso_crop):
    if torso_crop.size == 0:
        return None
    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 0])

cap = cv2.VideoCapture("data/worldcup_clip_trimmed.mp4")

hue_data = []

current_frame_num = -1
current_frame_img = None

with open("data/track_info.csv", "r") as file:
    reader = csv.DictReader(file)
    # rows = list(reader)
    for row in reader:
        frame_num = int(row['frame'])
        
        if frame_num != current_frame_num:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame, sorry....")
                break
            
            current_frame_img = frame
            current_frame_num = frame_num
            
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        
        torso = get_torso(current_frame_img, x1, y1, x2, y2)
        hue = get_dominant_hue(torso)
        
        if hue is not None:
            hue_data.append((int(row['track_id']), hue))

            
    
    
    
    

