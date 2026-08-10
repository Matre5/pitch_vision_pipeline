from ultralytics import YOLO
import csv

model = YOLO("yolo26n.pt")
results = model.track(
    source="data/worldcup_clip_trimmed.mp4",
    tracker="botsort.yaml",
    imgsz=960,
    conf=0.4,
    persist=True,
    save=True,
    project="outputs",
    name="tracking_test_BSrt_Conf",
    verbose=False,
    device=0,
    stream=True
)

header = 

with open("player_info.csv", "w", newline=" ", encoding="utf-8") as file:
    writer = csv.writer(file)