import cv2
import numpy as np
from sklearn.cluster import KMeans
from src import track

# def get_torso(frame, box):
#     pass

# print(type(box))

first_fr = track.results[0]
boxess = first_fr.boxes.xyxy

print(type(boxess))
print(boxess)