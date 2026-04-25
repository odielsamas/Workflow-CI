# Workflow-CI: Developer Burnout Prediction

## 📌 Deskripsi Proyek

Repositori ini berisi workflow CI/CD menggunakan **GitHub Actions** untuk menjalankan eksperimen Machine Learning terkait prediksi tingkat burnout developer.  
Workflow otomatis akan:

- Menjalankan script `modelling.py` dengan MLflow.
- Menghasilkan artefak hasil preprocessing (`train.csv`, `test.csv`).
- Mengupload artefak ke tab Actions sebagai bukti keberhasilan.

---

## 📂 Struktur Folder

Workflow-CI/
├── .github/
│ └── workflows/
│ └── ci.yml # File workflow GitHub Actions
├── developer_burnout_preprocessing/
│ ├── train.csv # Data hasil preprocessing (train)
│ └── test.csv # Data hasil preprocessing (test)
├── developer_burnout_raw/
│ └── developer_burnout.csv # Dataset mentah
├── MLProject # File konfigurasi MLflow Project
├── conda.yaml # Environment MLflow
├── modelling.py # Script training & evaluasi model
└── README.md # Dokumentasi proyek

## ⚙️ Cara Menjalankan

### 🔹 Manual (Lokal)

1. Buat virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate

   ```

2. Install dependencies:
   pip install -r requirements.txt

3. Jalankan script:
   python modelling.py
