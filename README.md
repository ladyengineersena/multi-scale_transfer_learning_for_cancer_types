# Multi-Scale Transfer Learning for Histopathological Images

Bu proje, farklı kanser tipleri arasında multi-scale transfer öğrenme gerçekleştirmek için tasarlanmıştır.

## Özellikler

- **Cross-Cancer Transfer Learning**: Meme → Akciğer, Pankreas gibi farklı kanser tipleri arasında öğrenme aktarımı
- **Multi-Scale Architecture**: Farklı çözünürlüklerde (512x512, 256x256, 128x128) özellik çıkarımı
- **Adaptive Domain Adaptation**: Farklı domenler arasında uyum sağlama
- **Ensemble Learning**: Multi-scale özellikleri birleştirerek güçlü kararlar alma

## Proje Yapısı

```
multi-scale-transfer-learning/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multi_scale_transfer.py      # Ana multi-scale model mimarisi
│   │   └── attention_modules.py         # Attention mekanizmaları
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py             # Veri ön işleme
│   │   └── augmentation.py              # Veri augmentasyonu
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                   # Training loop
│   │   └── losses.py                    # Loss fonksiyonları
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py             # Görselleştirme araçları
│       └── metrics.py                   # Metrikler
├── configs/
│   ├── config_breast_to_lung.yaml      # Meme→Akciğer konfigürasyonu
│   └── config_pancreas_to_lung.yaml    # Pankreas→Akciğer konfigürasyonu
├── scripts/
│   ├── train.py                         # Training script
│   └── evaluate.py                      # Evaluation script
├── data/                                # Veri klasörü (siz ekleyeceksiniz)
│   ├── breast_cancer/
│   ├── lung_cancer/
│   └── pancreas/
├── requirements.txt
└── README.md
```

## Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/ladyengineersena/multi-scale_transfer_learning_for_cancer_types.git
cd multi-scale_transfer_learning_for_cancer_types

# Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

## Kullanım

### Veri Hazırlama

Histopatolojik görüntüleri uygun klasörlere yerleştirin:

```
data/
├── breast_cancer/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── val/
├── lung_cancer/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── val/
```

### Model Eğitimi

#### Meme → Akciğer Transferi

```bash
python scripts/train.py \
    --source-domain breast_cancer \
    --target-domain lung_cancer \
    --config configs/config_breast_to_lung.yaml \
    --epochs 100 \
    --batch-size 16
```

#### Pankreas → Akciğer Transferi

```bash
python scripts/train.py \
    --source-domain pancreas \
    --target-domain lung_cancer \
    --config configs/config_pancreas_to_lung.yaml \
    --epochs 100 \
    --batch-size 16
```

### Model Değerlendirme

```bash
python scripts/evaluate.py \
    --checkpoint path/to/checkpoint.pth \
    --test-data data/lung_cancer/test
```

## Model Mimarisi

### Multi-Scale Feature Extraction

Model, farklı çözünürlüklerde (scale'lerde) özellik çıkarımı yapar:

1. **High Resolution (512x512)**: İnce detaylar için
2. **Medium Resolution (256x256)**: Orta seviye özellikler
3. **Low Resolution (128x128)**: Global bağlam

### Transfer Learning Mekanizması

```
Source Domain (e.g., Breast Cancer)
    ↓
[Pre-trained Feature Extractor]
    ↓
[Multi-Scale Fusion]
    ↓
[Domain Adapter]
    ↓
Target Domain (e.g., Lung Cancer)
```

## Hyperparametreler

- **Learning Rate**: 0.001 (source), 0.0001 (target adaptation)
- **Batch Size**: 16
- **Epochs**: 100
- **Optimizer**: AdamW
- **Loss**: Combined CE + Domain Adversarial Loss

## Sonuçlar

Model eğitimi tamamlandığında, TensorBoard ile sonuçları görselleştirebilirsiniz:

```bash
tensorboard --logdir runs/
```

## Lisans

MIT License

## Referanslar

- X. Chen et al., "Multi-Scale Transfer Learning for Medical Image Classification"
- Y. Wang et al., "Cross-Domain Histopathological Image Analysis"

