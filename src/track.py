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
    # device=0,
    stream=True
)

header = ['frame', 'track_id', 'x1', 'y1', 'x2', 'y2']

with open("track_info.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    
    writer.writerow(header)
    
    for frame_num, result in enumerate(results):
        if result.boxes.id is not None:
            for track_id, box in zip(result.boxes.id, result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                writer.writerow([frame_num, int(track_id), x1, y1, x2, y2])
