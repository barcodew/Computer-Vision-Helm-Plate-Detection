from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # atau "yolov8n.pt" kalau fine-tuning
model.train(data="E:\\ComputerVision\\plat\\data.yaml", epochs=50, imgsz=640, project="runs", name="plat_model")
