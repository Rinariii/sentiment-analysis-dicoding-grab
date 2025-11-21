# Flower Classification using CNN
Model deep learning untuk klasifikasi 5 jenis bunga menggunakan arsitektur CNN kustom dengan TensorFlow/Keras.

---

## Struktur
```
├── saved_model/
│   ├── saved_model.pb
│   └── variables/
│
├── tflite/
│   ├── model.tflite
│   └── label.txt
│
├── tfjs_model/
│   ├── model.json
│   └── group1-shard1of1.bin
│
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## Dataset
Dataset yang digunakan adalah **Custom Flower Dataset: 5-Class Image Dataset** dari Kaggle (https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset).

Kelas:
- Rose
- Marigold
- Petunia
- Hibiscus
- Chrysanthemum

Setiap gambar di-resize menjadi **224 × 224** untuk input.

---

## Model

Model dirancang agar ringan dan tidak overfitting, mengingat jumlah data per kelas terbatas (~200 gambar/class).

**Struktur:**
- Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.15)
- Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.20)
- Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
- Block 4: Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.30)
- GlobalAveragePooling2D
- Dense(128) + BatchNorm + Dropout(0.4)
- Dense(5, Softmax)

**Callbacks:**
- ReduceLROnPlateau  
- EarlyStopping  

---

## Training Configuration

| Parameter | Nilai |
|----------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Input Shape | 224 × 224 × 3 |
| Epochs | 20|

---

## Model Exports

### SavedModel
Tersimpan di:

```
submission/saved_model/
```
---

## TensorFlow Lite (TFLite)
Tersimpan di:

```
submission/tflite/model.tflite
submission/tflite/label.txt
```
---

### TensorFlow.js
Tersimpan di:

```
submission/tfjs_model/
```

---

## Requirements

Install dependencies:

```
pip install -r requirements.txt
```

---

## Summary

Proyek ini memiliki akurasi pada training mencapai **97-98**% dengan akurasi pada test set 98%
