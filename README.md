# 🚀 India Startup Funding Intelligence

> An open-source platform to explore, analyze, and predict startup funding trends across India (2015–2024).

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🌐 Live Demo

| | URL |
|---|---|
| 📊 Dashboard | https://india-startup-intelligence.streamlit.app |
| ⚡ API | https://india-startup-intelligence-production.up.railway.app |
| 📖 API Docs | https://india-startup-intelligence-production.up.railway.app/docs |

---

## 🧩 Problem

India's startup funding data is scattered across Crunchbase, Tracxn, Inc42, and YourStory.
Existing tools are either paywalled or not India-specific.
There is no free, open-source tool that unifies this data, surfaces trends, and predicts fundraising likelihood.

**This project solves that.**

---

## ✨ Features

- 📊 **Interactive Dashboard** — funding trends by year, top sectors, top cities
- 🔍 **Startup Explorer** — filter 3,200+ startups by sector and city
- 🤖 **ML Prediction** — XGBoost model predicts fundraising likelihood (77% accuracy)
- ⚡ **REST API** — 6 endpoints, auto-documented via OpenAPI/Swagger
- 🗄️ **Data Pipeline** — scraper + cleaner + loader for reproducible data ingestion

---

## 🏗️ Architecture
```
Internet / Kaggle CSVs

↓

Pipeline/scraper.py      ← BeautifulSoup scraper (Inc42)

Pipeline/cleaner.py      ← Pandas cleaning + deduplication

Pipeline/loader.py       ← SQLite database loader

↓

Data/startups.db         ← 3,215 clean startup records

↓

Backend/main.py          ← FastAPI REST API (6 endpoints)

Model/train.py           ← XGBoost classifier (77% accuracy)

↓

Dashboard/app.py         ← Streamlit UI
```
---

## 🛠️ Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Data Collection | requests + BeautifulSoup | Lightweight scraping |
| Data Processing | Pandas | Industry standard |
| Database | SQLite | Zero setup, file-based |
| Backend API | FastAPI | Auto-docs, production-grade |
| ML Model | XGBoost + scikit-learn | Best for tabular data |
| Frontend | Streamlit | Python-only, deployable |
| API Deployment | Railway | Free, GitHub-connected |
| Dashboard Deployment | Streamlit Cloud | Free, one-click deploy |

---

## 📁 Project Structure
india-startup-intelligence/
```
├── Pipeline/
│   ├── scraper.py        ← Inc42 news scraper
│   ├── cleaner.py        ← data cleaning pipeline
│   └── loader.py         ← SQLite loader
├── Backend/
│   └── main.py           ← FastAPI app (6 endpoints)
├── Model/
│   ├── train.py          ← XGBoost training script
│   ├── model.pkl         ← saved model
│   └── encoders.pkl      ← saved label encoders
├── Dashboard/
│   └── app.py            ← Streamlit dashboard
├── Data/
│   └── raw/              ← place downloaded CSVs here
├── requirements.txt
└── README.md
```
---

## 🚀 Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/mohdkaif3002/india-startup-intelligence.git
cd india-startup-intelligence
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download datasets**

Download these three CSVs from Kaggle and place them in `Data/raw/`:
- [Indian Startup Funding (2015–2020)](https://www.kaggle.com/datasets/sudalairajkumar/indian-startup-funding)
- [Indian Startup Funding 2023](https://www.kaggle.com/datasets/venkateswarlukurarva/indian-startup-funding-2023)
- [Indian Startup Funding 2024](https://www.kaggle.com/datasets/aniketdash7/indian-startup-funding-data-2024)

**5. Run the pipeline**
```bash
python Pipeline/cleaner.py
python Pipeline/loader.py
```

**6. Train the model**
```bash
python Model/train.py
```

**7. Start the API**
```bash
uvicorn Backend.main:app --reload
```

**8. Launch the dashboard**
```bash
streamlit run Dashboard/app.py
```

Or visit the live demo: https://india-startup-intelligence.streamlit.app

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/startups` | List startups (filter by sector, city) |
| GET | `/trends` | Funding by year |
| GET | `/sectors` | Top sectors by deal count |
| GET | `/cities` | Top cities by deal count |
| GET | `/predict` | ML prediction for fundraising likelihood |

**Base URL:** https://india-startup-intelligence-production.up.railway.app  
**API Docs:** https://india-startup-intelligence-production.up.railway.app/docs

---

## 🤖 ML Model

- **Algorithm:** XGBoost Classifier
- **Features:** sector, city, investment type, amount raised
- **Label:** startup raised multiple rounds (1) vs single round (0)
- **Accuracy:** 77.29% on test split
- **Known limitation:** class imbalance (26% positive labels). Next version will use SMOTE oversampling.

---

## 📊 Data Sources

| Source | Records | Years |
|---|---|---|
| Kaggle (SRK dataset) | ~3,000 | 2015–2020 |
| Kaggle (2023 dataset) | ~3,000 | 2023 |
| Kaggle (2024 dataset) | 436 | 2024 |

After cleaning and deduplication: **3,215 unique records**

---

## 🗺️ Roadmap

- [x] Data pipeline (scraper, cleaner, loader)
- [x] FastAPI backend with 6 endpoints
- [x] XGBoost ML model (77% accuracy)
- [x] Streamlit dashboard
- [x] Deploy to Railway + Streamlit Cloud
- [ ] PostgreSQL migration
- [ ] SMOTE to fix class imbalance
- [ ] Real-time Inc42 scraper with pagination
- [ ] FastAPI authentication
- [ ] GitHub Actions CI/CD

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built by Mohammad Kaif as part of an open-source data engineering portfolio.  
GitHub: [@mohdkaif3002](https://github.com/mohdkaif3002)
