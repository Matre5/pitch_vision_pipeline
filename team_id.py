import cv2
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def get_torso(frame, x1, y1, x2, y2):
    height = y2 - y1
    top_torso = y1 + int(height * 0.20)
    bottom_torso = y1 + int(height * 0.55)
    
    return frame[top_torso:bottom_torso, x1:x2]

def get_dominant_hue(torso_crop):
    if torso_crop.size == 0:
        return None
    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    hue = np.mean(hsv[:, : , 0])
    sat = np.mean(hsv[:, : , 1])
    return hue, sat

cap = cv2.VideoCapture("data/worldcup_clip_final.mp4")

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
        
        result = get_dominant_hue(torso)
        if result is not None:
            hue, sat = result
            hue_data.append((int(row['track_id']), hue, sat))
 
    
# print(f"Total hue readings: {len(hue_data)}")
# print(hue_data[:15])

    
    
# Clustering
track_ids = [item[0] for item in hue_data]
features = [[item[1], item[2]] for item in hue_data] # getting the hue and sat

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(scaled_features)

for i in range(15):
    print(f"track_id: {track_ids[i]}, hue: {features[i][0]:.1f}, sat: {features[i][1]:.1f}, cluster: {labels[i]}")