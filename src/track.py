from ultralytics import YOLO

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
    verbose=False
    
)