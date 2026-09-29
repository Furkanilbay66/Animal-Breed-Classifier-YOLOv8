"""
YOLOv8 Classification — Hayvan Sınıflandırıcı Eğitimi
GPU varsa otomatik kullanır, VRAM'e göre batch size ayarlar.
Eğitim sonunda best.pt kök dizine kopyalanır.
"""
from ultralytics import YOLO
import torch
import shutil
import os


def main():
    print("=" * 55)
    print("  YOLOv8 Classification — Hayvan Sınıflandırıcı")
    print("=" * 55)

    # ── GPU / CPU seçimi ──────────────────────────────────────
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 12:
            batch = 64
        elif vram_gb >= 6:
            batch = 32
        elif vram_gb >= 3:
            batch = 16
        else:
            batch = 8
        print(f"  GPU  : {gpu_name}")
        print(f"  VRAM : {vram_gb:.1f} GB  →  batch={batch}")
    else:
        device = 'cpu'
        batch  = 16
        print("  UYARI: GPU bulunamadı, CPU kullanılıyor (yavaş olabilir)")

    # ── Dataset kontrolü ─────────────────────────────────────
    if not os.path.isdir('animal-dataset/train'):
        print("\n  HATA: Dataset bulunamadı!")
        print("  Önce şunu çalıştırın: python prepare_dataset.py")
        return

    # ── Model yükleme ─────────────────────────────────────────
    # yolov8m-cls → medium sınıflandırıcı (doğruluk/hız dengesi)
    print("\n  Model: YOLOv8m-cls (pretrained ImageNet)")
    model = YOLO('yolov8m-cls.pt')

    # ── Eğitim ────────────────────────────────────────────────
    print("  Eğitim başlıyor...\n")
    model.train(
        data        = 'animal-dataset',   # sınıf klasörleri (train/val)
        epochs      = 50,
        imgsz       = 224,
        batch       = batch,
        device      = device,
        workers     = 4 if device != 'cpu' else 2,
        patience    = 15,                 # erken durdurma
        save        = True,
        project     = 'runs/classify',
        name        = 'animal_classifier',
        pretrained  = True,
        optimizer   = 'AdamW',
        lr0         = 0.001,
        warmup_epochs = 3,
        dropout     = 0.2,
        verbose     = True,
    )

    # ── Değerlendirme ─────────────────────────────────────────
    print("\n  Model değerlendiriliyor...")
    metrics = model.val()
    print(f"\n  Top-1 Accuracy : %{metrics.top1 * 100:.2f}")
    print(f"  Top-5 Accuracy : %{metrics.top5 * 100:.2f}")

    # ── En iyi modeli kök dizine kopyala ─────────────────────
    best_src = 'runs/classify/animal_classifier/weights/best.pt'
    if os.path.exists(best_src):
        shutil.copy(best_src, 'best.pt')
        print("\n  'best.pt' kök dizine kopyalandı.")

    print("\n  Eğitim tamamlandı! Şimdi: python app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
