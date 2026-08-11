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

with open("data/track_info.csv", "r") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    
    
    
    
    

