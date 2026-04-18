# Agri-Credit Risk & Loan Default Prediction Pipeline

## 📌 Problem Statement
Farmers and agricultural workers often lack traditional financial credit histories, making it difficult for banks and micro-lenders to assess their creditworthiness. Because of this "thin-file" problem, many farmers are excluded from formal credit systems. 

Financial institutions need a reliable way to assess agricultural loan risk using **alternative data**—such as historical crop yield, farm size, rainfall, soil quality, and fertilizer usage—to determine a fair credit score and accurately predict the likelihood of a loan default.

## 💡 Proposed Solution
We built an end-to-end machine learning and data engineering pipeline on the **Databricks** platform. This solution bridges the gap between raw agricultural metrics and financial risk assessment through a Medallion architecture:

1. **Synthetic CIBIL Scoring (Feature Engineering):** We process raw agricultural and meteorological data to engineer a custom "Agri-Score". By utilizing PySpark Window functions and SciPy's skew-normal distribution (`scipy.stats.skewnorm`), we map raw farm metrics to a realistic credit score distribution (300-900 range, negatively skewed with a mean around ~730).
2. **Loan Default Prediction (Predictive AI):** We utilize **Databricks AutoML** to automatically train, tune, and log optimal predictive models (e.g., XGBoost, Random Forest) that determine the probability of loan default based on the farmer's parameters and their new synthetic Agri-Score.
3. **GenAI Advisory (Prescriptive AI):** We extract feature importances from the trained ML model and feed them into an ultra-fast LLM via the **Groq API**. This generates personalized, step-by-step advice for the farmer on how to optimize their farm parameters to improve their score and secure a loan.

---

## 🏗️ System Architecture

Code output
README.md created successfully

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             1. DATA SOURCES                                 │
│  ┌────────────────────┐ ┌────────────────────┐ ┌─────────────────────────┐  │
│  │ Farmer Details UI  │ │ Kaggle / APIs      │ │ Historical Loan Data    │  │
│  │ (Farm size, fert.) │ │ (Rainfall, soil)   │ │ (Default status)        │  │
│  └─────────┬──────────┘ └─────────┬──────────┘ └────────────┬────────────┘  │
└────────────┼──────────────────────┼─────────────────────────┼───────────────┘
             │                      │                         │
             ▼                      ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  2. DATABRICKS DATA ENGINEERING (ETL)                       │
│                                                                             │
│  [Unity Catalog Volume]                                                     │
│   📁 /Volumes/Catalog1/Schema1/Volume1/crop_yield_data.csv                  │
│             │                                                               │
│             ▼                                                               │
│  [Bronze Layer] -> Raw PySpark DataFrames ingested directly from CSV.       │
│             │                                                               │
│             ▼                                                               │
│  [Silver Layer] -> Cleaned Delta Table. Handled missing values, renamed     │
│                    columns (removed spaces), formatted data types.          │
│             │                                                               │
│             ▼                                                               │
│  [Gold Layer]   -> Feature-Engineered Delta Table (`farmers_scored_gold`).  │
│                    • Calculated weighted scores (Yield, Soil, Rain, etc.).  │
│                    • Applied SciPy Skew-Normal mapping (mean=725).          │
│                    • Generated final 'synthetic_agri_cibil'.                │
└─────────────────────────────┬───────────────────────────┬───────────────────┘
                              │                           │
          ┌───────────────────┘                           └───────────────────┐
          ▼                                                                   ▼
┌───────────────────────────────────┐               ┌───────────────────────────────────┐
│     3. PREDICTIVE ML PIPELINE     │               │     4. GENERATIVE AI ADVISOR      │
│                                   │               │                                   │
│  • Databricks AutoML              │               │  • Feature Importance Extractor   │
│    (Trains XGBoost, Random Forest)│               │    (Pulls weights from ML model)  │
│  • MLflow Experiment Tracking     │               │  • Groq API Integration           │
│    (Logs R2, hyperparams, models) │               │    (bharatgenai / Llama 3)        │
│  • Output: Loan Default Model     │               │  • Output: Personalized Action    │
│                                   │               │    Plan for the Farmer            │
└─────────────────┬─────────────────┘               └─────────────────┬─────────────────┘
                  │                                                   │
                  ▼                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            5. SERVING & CONSUMPTION                         │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Databricks Model Serving API                      │  │
│  │  (Serverless REST endpoint hosting both the ML model & ETL logic)     │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    End-User Application / Dashboard                   │  │
│  │ 1. Displays Farmer's Synthetic CIBIL (e.g., 735)                      │  │
│  │ 2. Displays Probability of Default (e.g., 12% Risk)                   │  │
│  │ 3. Displays AI Advice ("Increase fertilizer by 200kg to boost score") │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
