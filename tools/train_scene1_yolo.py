"""Train YOLO on scene1_yolo dataset, then test on the original scene."""
import sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path(r"C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp")
sys.path.insert(0, str(REPO / ".yolo_runtime"))
from ultralytics import YOLO

DATA = REPO / "training_data" / "scene1_yolo" / "data.yaml"
SCENE = REPO / "monster" / "测试集" / "场景1.png"
RUN_NAME = "scene1_v1"
RUN_DIR = REPO / "training_runs" / RUN_NAME
WEIGHTS = REPO / "yolo26n.pt"

model = YOLO(str(WEIGHTS))
model.train(
    data=str(DATA),
    epochs=30,
    imgsz=640,
    batch=8,
    device="0",
    workers=0,
    project=str(REPO / "training_runs"),
    name=RUN_NAME,
    exist_ok=True,
    pretrained=True,
    patience=8,
    seed=17,
    plots=True,
    verbose=False,
)
print("train done, weights:", RUN_DIR / "weights" / "best.pt")

# Test on the original scene
best_model = YOLO(str(RUN_DIR / "weights" / "best.pt"))
img = cv2.imdecode(np.fromfile(str(SCENE), dtype=np.uint8), cv2.IMREAD_COLOR)
results = best_model.predict(source=img, conf=0.25, iou=0.45, verbose=False)
r = results[0]
names = r.names
boxes = r.boxes
print(f"detections: {len(boxes)}")
vis = img.copy()
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{names[cls_id]} {conf:.2f}"
    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    cv2.putText(vis, label, (int(x1), max(15, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
out_path = REPO / "probe_output" / "scene1_yolo_pred.png"
ok, buf = cv2.imencode(".png", vis)
out_path.write_bytes(buf.tobytes())
print("saved prediction to", out_path)