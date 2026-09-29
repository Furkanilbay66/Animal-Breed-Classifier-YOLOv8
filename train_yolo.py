# train_yolo.py
from ultralytics import YOLO
import os

# 1. YOLOv8 modelini yükle (önceden eğitilmiş)
model = YOLO('yolov8n.pt')  # 'n' = nano (hızlı), 'm' = medium, 'l' = large

# 2. Veri seti yapılandırması (data.yaml oluştur)
data_yaml = """
path: ./animal-dataset
train: train/images
val: val/images

names:
  0: butterfly
  1: cat
  2: chicken
  3: cow
  4: dog
  5: horse
  6: sheep
"""

with open('data.yaml', 'w') as f:
    f.write(data_yaml)

# 3. Modeli eğit (GPU kullanarak)
model.train(
    data='data.yaml',
    epochs=50,              # Tur sayısı
    imgsz=640,              # Resim boyutu
    batch=16,               # Batch size (GPU'na göre ayarla)
    device=0,               # 0 = GPU, 'cpu' = CPU
    workers=4,
    patience=10,            # Erken durdurma
    save=True,
    project='animal_classifier'
)

# 4. Modeli kaydet
model.export(format='onnx')  # Web için optimize