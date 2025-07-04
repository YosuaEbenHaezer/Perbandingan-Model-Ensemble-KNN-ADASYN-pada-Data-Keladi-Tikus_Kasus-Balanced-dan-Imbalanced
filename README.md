
# 🌱 Perbandingan Model Ensemble & KNN dengan ADASYN pada Data Keladi Tikus  
## Kasus Balanced vs Imbalanced Data

Proyek ini mengeksplorasi pengaruh teknik penyeimbangan data (khususnya **ADASYN**) terhadap performa model **KNN** dan **Model Ensemble** seperti Random Forest, Bagging, dan Boosting dalam klasifikasi data Keladi Tikus.

---

## 📁 File Proyek

- `Project Makalah_Pembelajaran Mesin_Pake ADASYN.ipynb` → Implementasi model dengan **ADASYN** (balanced)
- `tanpa ADASYN.ipynb` → Implementasi model **tanpa ADASYN** (imbalanced)

---

## 📊 Tujuan

Membandingkan performa model KNN dan model ensemble dalam dua kondisi:
1. **Data Imbalanced (tanpa ADASYN)**
2. **Data Balanced (menggunakan ADASYN)**

Dengan harapan dapat:
- Mengetahui pengaruh balancing terhadap akurasi model
- Menemukan model yang paling stabil di kedua kondisi

---

## 📦 Library yang Digunakan

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

# ADASYN
from imblearn.over_sampling import ADASYN
🚀 Cara Menjalankan
Pastikan environment Python kamu sudah memiliki semua library di atas.
Install via pip jika belum:

bash
Salin kode
pip install pandas numpy scikit-learn imbalanced-learn
Jalankan notebook tanpa ADASYN.ipynb untuk eksperimen dengan data imbalanced.

Jalankan notebook Project Makalah_Pembelajaran Mesin_Pake ADASYN.ipynb untuk eksperimen dengan data yang diseimbangkan menggunakan ADASYN.

🧪 Evaluasi Model
Metrik yang digunakan:

Akurasi

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

🔍 Hasil yang Diharapkan
Model ensemble cenderung lebih stabil terhadap data imbalanced.

KNN sangat bergantung pada distribusi data → performanya membaik setelah balancing dengan ADASYN.

ADASYN dapat meningkatkan kinerja model dengan membuat data minoritas lebih terwakili.

📌 Catatan Tambahan
Dataset Keladi Tikus merupakan data klasifikasi dengan distribusi kelas yang tidak seimbang secara alami.

ADASYN digunakan untuk mensintesis contoh baru dari kelas minoritas berdasarkan distribusi lokal.

Proyek ini bisa dijadikan dasar untuk eksperimen dengan SMOTE, Borderline-SMOTE, atau metode balancing lainnya.

✍️ Proyek ini disusun untuk mengevaluasi dampak balancing data terhadap performa algoritma machine learning klasik dalam konteks klasifikasi medis/tumbuhan lokal.
