# Capstone-Project-Module-3
Project Capstone Module 3 Purwadhika ~ David Gosal

# 🏠 California Housing Price Prediction

## 📌 Project Introduction
Proyek ini bertujuan membangun **model prediksi harga median rumah per distrik di California** berdasarkan data sensus tahun 1990 (*California Housing Dataset*).  
Dengan memanfaatkan faktor lokasi, demografi, dan karakteristik fisik hunian, model ini diharapkan membantu pengambil keputusan seperti:
- **Agen properti** → menentukan harga listing yang kompetitif.
- **Pengembang** → memprioritaskan area pembangunan.
- **Lembaga pembiayaan** → melakukan appraisal awal yang akurat.

Tanpa pendekatan berbasis data, risiko **overpricing** (harga terlalu tinggi) atau **underpricing** (harga terlalu rendah) menjadi besar, yang berdampak pada waktu penjualan, margin keuntungan, dan efisiensi operasional.

---

## 🎯 Goals
- Mengembangkan **Pricing Assistant** untuk memperkirakan harga median rumah per distrik.
- Menyediakan **driver analysis** agar alasan di balik prediksi bisa dipahami.
- Mendukung **what-if analysis** (misalnya, pengaruh peningkatan pendapatan median atau perubahan kepadatan).
- Mengukur performa model dengan metrik:
  - MAE
  - RMSE
  - R²
  - MdAPE

---

## 📂 Dataset
**California Housing Dataset** (sensus 1990) dengan 10 kolom utama:
- `longitude`, `latitude` (lokasi geografis)
- `housing_median_age` (umur bangunan)
- `total_rooms`, `total_bedrooms`
- `population`, `households`
- `median_income`
- `ocean_proximity` (kategori jarak ke pantai)
- `median_house_value` (target)

---

## 🛠️ Methods
1. **EDA & Data Cleaning**
   - Imputasi *missing values* (metode ratio-aware).
   - Capping *outlier* berdasarkan distribusi.
   - Feature engineering (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`).
2. **Modeling**
   - Benchmark model: Linear Regression, KNN, Decision Tree, Random Forest, XGBoost.
   - **Hyperparameter tuning** dengan RandomizedSearchCV (XGBoost).
   - Stacking model (Random Forest + XGBoost).
3. **Evaluation**
   - Perbandingan metrik train vs test untuk deteksi overfitting.
   - Scatter plot prediksi vs aktual.
---

## 📊 Key Results
| Model | MAE | RMSE | R² | MdAPE |
|-------|-----|------|----|-------|
| **XGB (CV best)** | ~29,191 | ~46,148 | 0.834 | ~10.78% |
| **Stacking (RF+XGB)** | ~29,205 | ~46,006 | 0.835 | ~10.84% |

📌 **Insight**:
- XGB hasil tuning memberikan performa terbaik dengan error rata-rata sekitar **$29K** (MAE).
- Stacking sedikit meningkatkan R² namun perbedaan sangat tipis.
- MdAPE ~10.8% berarti prediksi rata-rata meleset ±10.8% dari harga sebenarnya.

---

## 📦 Model Deployment
Model terbaik (`xgb_cv_best`) telah disimpan dalam format `.pkl` sehingga dapat digunakan kembali tanpa training ulang:

```python
import pickle

with open("xgb_cv_best.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
