import cv2
import pytesseract
from ultralytics import YOLO
import csv
import os
from datetime import timedelta

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load model YOLO
helmet_model = YOLO(r"ProjectCV\helmModel.pt")
plate_model = YOLO(r"ProjectCV\plateModel.pt")

# Buka video
# cap = cv2.VideoCapture(r"E:\ComputerVision\20250613_165727.mp4")
cap = cv2.VideoCapture(r"E:\ComputerVision\20250613_165220.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("hasil_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Siapkan CSV
csv_file = open("hasil_ocr.csv", mode='w', newline='', encoding='utf-8')
writer = csv.writer(csv_file)
writer.writerow(["waktu", "frame", "jenis", "confidence", "x1", "y1", "x2", "y2", "text"])

frame_num = 0
ocr_interval = 1  # Jalankan OCR di setiap frame (bisa diubah ke 5 untuk percepatan)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    waktu = str(timedelta(seconds=frame_num / fps))

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    pelanggaran_helm = True
    plat_terdeteksi = False

    helm_results = helmet_model(frame, imgsz=640)[0]
    for box in helm_results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = helmet_model.names[cls].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0) if 'helmet' in label else (0, 0, 255)

        if 'helmet' in label:
            pelanggaran_helm = False

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.writerow([waktu, frame_num, label, conf, x1, y1, x2, y2, ""])

    ##### 🚘 DETEKSI PLAT + OCR #####
    if frame_num % ocr_interval == 0:
        plate_results = plate_model(frame, imgsz=960)[0]
        for box in plate_results.boxes:
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 50:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            roi_up = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(roi_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(thresh, config=config).strip()

            if text:
                plat_terdeteksi = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            writer.writerow([waktu, frame_num, "plat_nomor", conf, x1, y1, x2, y2, text])
            cv2.imshow("ROI OCR", thresh)

    out.write(frame)
    cv2.imshow("Deteksi Helm & Plat", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
cv2.waitKey(1)

print("✅ Selesai. Hasil disimpan ke: hasil_ocr.csv dan hasil_output.mp4")
