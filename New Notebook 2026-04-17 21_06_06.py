# Databricks notebook source
# Install the Kaggle library
%pip install kaggle

import os

os.environ['KAGGLE_USERNAME'] = "WRebelZ"
os.environ['KAGGLE_KEY'] = "KGAT_bc4d3b530e2a9a738904ec85ad57e9cc"

# Download (goes to /databricks/driver/)
!kaggle datasets download -d govindaramsriram/crop-yield-of-a-farm

# COMMAND ----------

!unzip /Workspace/Users/mems230005026@iiti.ac.in/Bharat_Bricks_Strawhats/crop-yield-of-a-farm.zip

# COMMAND ----------

# DBTITLE 1,Cell 2
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS Catalog1;
# MAGIC
# MAGIC USE CATALOG Catalog1;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS Schema1;
# MAGIC
# MAGIC CREATE VOLUME IF NOT EXISTS Schema1.Volume1;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CATALOGS;

# COMMAND ----------

# catalog = "Catalog1"
# schema = "Schema1"
# volume = "Volume1"
# file_name = "/Workspace/Users/mc230041017@iiti.ac.in/Drafts/crop_yield_data.csv"
# # table_name = "baby_names"
# path_volume = "/Volumes/" + catalog + "/" + schema + "/" + volume
# path_table = catalog + "." + schema
# print(path_table) # Show the complete path
# print(path_volume) # Show the complete path

# COMMAND ----------

# DBTITLE 1,Cell 6
# 1. Define your target table name and full paths
table_name = "crop_yield"
full_table_name = f"{path_table}.{table_name}" # e.g., Catalog1.Schema1.crop_yield
volume_file_path = f"{path_volume}/crop_yield_data.csv" # e.g., /Volumes/Catalog1/Schema1/Volume1/crop_yield_data.csv

# 2. Copy the file from your Workspace to the Unity Catalog Volume
# Use the correct path where the file actually exists
correct_file_path = "/Workspace/Users/mems230005026@iiti.ac.in/Bharat_Bricks_Strawhats/crop_yield_data.csv"
dbutils.fs.cp(f"file:{correct_file_path}", volume_file_path)
print(f"Successfully copied to Volume: {volume_file_path}")

# 3. Read the CSV file from the Volume into a PySpark DataFrame
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(volume_file_path)

# 4. Clean column names (Delta tables and AutoML don't like spaces or special characters in column names)
import re
for col_name in df.columns:
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', col_name.replace(' ', '_'))
    df = df.withColumnRenamed(col_name, clean_name)

# 5. Save the DataFrame as a managed Delta Table in Unity Catalog
df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)
print(f"Successfully saved as Unity Catalog table: {full_table_name}")

# 6. Display the final table to verify everything worked
display(spark.table(full_table_name))

# COMMAND ----------

# DBTITLE 1,Cell 7
from pyspark.sql.functions import col, when, lit, greatest, least, pow as spark_pow

# 1. Load the table you just created
df = spark.table(full_table_name)

# 2. Define normalization values based on ACTUAL data ranges
# Data ranges: rainfall (500-2000), soil (1-10), farm_size (10-1000), 
#              sunlight (4-12), fertilizer (100-3000), crop_yield (46-628)
MAX_YIELD = 650.0      # Slightly above max to allow room
MAX_FARM_SIZE = 1000.0 # Actual max in data
MAX_SOIL_IDX = 10.0    # Actual max (1-10 scale, not 100!)
MAX_SUN = 12.0         # Actual max daily hours
MAX_FERT = 3000.0      # Actual max kg

# 3. Calculate Normalized Scores (0.0 to 1.0) for each parameter
# Yield: Most important - use progressive scoring (better yield = exponentially better score)
df_scored = df.withColumn(
    "score_yield",
    least(spark_pow(col("crop_yield") / lit(MAX_YIELD), lit(0.85)), lit(1.0))
)

# Farm Size: Larger is generally better, but with diminishing returns
df_scored = df_scored.withColumn(
    "score_size",
    least(spark_pow(col("farm_size_hectares") / lit(MAX_FARM_SIZE), lit(0.7)), lit(1.0))
)

# Soil Quality: Linear scale from 1-10
df_scored = df_scored.withColumn(
    "score_soil",
    (col("soil_quality_index") - lit(1.0)) / lit(MAX_SOIL_IDX - 1.0)
)

# Sunlight: More is better but need minimum threshold
df_scored = df_scored.withColumn(
    "score_sun",
    when(col("sunlight_hours") >= 8, 
         least((col("sunlight_hours") - lit(4.0)) / lit(MAX_SUN - 4.0), lit(1.0)))
    .otherwise((col("sunlight_hours") - lit(4.0)) / lit(8.0 - 4.0) * lit(0.7))
)

# Fertilizer: Optimal range exists (too little = bad, too much = wasteful/bad)
# Optimal range: 800-2000 kg
df_scored = df_scored.withColumn(
    "score_fert",
    when((col("fertilizer_kg") >= 800) & (col("fertilizer_kg") <= 2000), lit(1.0))
    .when(col("fertilizer_kg") < 800, col("fertilizer_kg") / lit(800.0))
    .otherwise(greatest(lit(0.3), lit(1.0) - ((col("fertilizer_kg") - lit(2000.0)) / lit(2000.0))))
)

# 4. Handle Rainfall (Goldilocks principle: 1000-1500mm is ideal for most crops)
df_scored = df_scored.withColumn(
    "score_rain",
    when((col("rainfall_mm") >= 1000) & (col("rainfall_mm") <= 1500), lit(1.0))
    .when(col("rainfall_mm") < 1000, 
         spark_pow(col("rainfall_mm") / lit(1000.0), lit(1.2)))
    .otherwise(
        greatest(lit(0.2), lit(1.0) - spark_pow((col("rainfall_mm") - lit(1500.0)) / lit(1000.0), lit(1.5)))
    )
)

# 5. Apply Revised Weights to Create Better Distribution
# Adjusted weights for agricultural credit scoring:
# - Yield (0.35): Primary indicator of farm productivity
# - Soil Quality (0.20): Foundation of good farming
# - Farm Size (0.15): Larger farms = more collateral/scale
# - Rainfall (0.15): Critical environmental factor
# - Sunlight (0.10): Important but less variable
# - Fertilizer (0.05): Management indicator

df_final = df_scored.withColumn(
    "weighted_score",
    (col("score_yield") * lit(0.35)) +
    (col("score_soil") * lit(0.20)) +
    (col("score_size") * lit(0.15)) +
    (col("score_rain") * lit(0.15)) +
    (col("score_sun") * lit(0.10)) +
    (col("score_fert") * lit(0.05))
)

# 6. Map weighted score (0-1) to CIBIL range (300-900)
# Use a slightly non-linear mapping to spread out the distribution
df_final = df_final.withColumn(
    "synthetic_agri_cibil",
    (lit(300) + (lit(600) * spark_pow(col("weighted_score"), lit(0.95)))).cast("int")
)

# 7. Display the results
display(df_final.select(
    "rainfall_mm", "soil_quality_index", "farm_size_hectares", 
    "sunlight_hours", "fertilizer_kg", "crop_yield", 
    "synthetic_agri_cibil"
).orderBy(col("synthetic_agri_cibil").desc()))

# Optional: Show score distribution to verify realistic spread
print("\n=== CIBIL Score Distribution ===")
df_final.groupBy(
    when(col("synthetic_agri_cibil") < 500, "300-499: Poor")
    .when(col("synthetic_agri_cibil") < 650, "500-649: Fair")
    .when(col("synthetic_agri_cibil") < 750, "650-749: Good")
    .when(col("synthetic_agri_cibil") < 850, "750-849: Very Good")
    .otherwise("850-900: Excellent")
    .alias("score_range")
).count().orderBy("score_range").show()

# Save it as your Gold Table
# df_final.write.format("delta").mode("overwrite").saveAsTable(f"{path_table}.farmers_scored_gold")

# COMMAND ----------

##### Expermiment

from pyspark.sql import Window
from pyspark.sql.functions import col, when, lit, greatest, least, pow as spark_pow, percent_rank, pandas_udf
import pandas as pd
from scipy.stats import norm

# 1. Load the table you just created
df = spark.table(full_table_name)

# 2. Define normalization values based on ACTUAL data ranges
MAX_YIELD = 650.0      # Slightly above max to allow room
MAX_FARM_SIZE = 1000.0 # Actual max in data
MAX_SOIL_IDX = 10.0    # Actual max (1-10 scale, not 100!)
MAX_SUN = 12.0         # Actual max daily hours
MAX_FERT = 3000.0      # Actual max kg

# 3. Calculate Normalized Scores (0.0 to 1.0) for each parameter
df_scored = df.withColumn(
    "score_yield", least(spark_pow(col("crop_yield") / lit(MAX_YIELD), lit(0.85)), lit(1.0))
)

df_scored = df_scored.withColumn(
    "score_size", least(spark_pow(col("farm_size_hectares") / lit(MAX_FARM_SIZE), lit(0.7)), lit(1.0))
)

df_scored = df_scored.withColumn(
    "score_soil", (col("soil_quality_index") - lit(1.0)) / lit(MAX_SOIL_IDX - 1.0)
)

df_scored = df_scored.withColumn(
    "score_sun",
    when(col("sunlight_hours") >= 8, least((col("sunlight_hours") - lit(4.0)) / lit(MAX_SUN - 4.0), lit(1.0)))
    .otherwise((col("sunlight_hours") - lit(4.0)) / lit(8.0 - 4.0) * lit(0.7))
)

df_scored = df_scored.withColumn(
    "score_fert",
    when((col("fertilizer_kg") >= 800) & (col("fertilizer_kg") <= 2000), lit(1.0))
    .when(col("fertilizer_kg") < 800, col("fertilizer_kg") / lit(800.0))
    .otherwise(greatest(lit(0.3), lit(1.0) - ((col("fertilizer_kg") - lit(2000.0)) / lit(2000.0))))
)

# 4. Handle Rainfall
df_scored = df_scored.withColumn(
    "score_rain",
    when((col("rainfall_mm") >= 1000) & (col("rainfall_mm") <= 1500), lit(1.0))
    .when(col("rainfall_mm") < 1000, spark_pow(col("rainfall_mm") / lit(1000.0), lit(1.2)))
    .otherwise(greatest(lit(0.2), lit(1.0) - spark_pow((col("rainfall_mm") - lit(1500.0)) / lit(1000.0), lit(1.5))))
)

# 5. Apply Revised Weights
df_scored = df_scored.withColumn(
    "weighted_score",
    (col("score_yield") * lit(0.35)) +
    (col("score_soil") * lit(0.20)) +
    (col("score_size") * lit(0.15)) +
    (col("score_rain") * lit(0.15)) +
    (col("score_sun") * lit(0.10)) +
    (col("score_fert") * lit(0.05))
)

# 6. FORCE NORMAL DISTRIBUTION (Mean=600, StdDev=100)
# 6a. Rank everyone's weighted score as a percentile
window_spec = Window.partitionBy().orderBy("weighted_score")
df_final = df_scored.withColumn("percentile", percent_rank().over(window_spec))

# 6b. Clip the percentiles to avoid +/- infinity in the statistical function
# 0.0013 and 0.9987 correspond exactly to the bounds of 300 and 900 on a normal curve
df_final = df_final.withColumn(
    "percentile_clipped",
    when(col("percentile") < 0.0013, 0.0013)
    .when(col("percentile") > 0.9987, 0.9987)
    .otherwise(col("percentile"))
)

# 6c. Define a Pandas UDF to map percentiles to the normal curve using SciPy
@pandas_udf("double")
def map_to_normal_udf(percentile_series: pd.Series) -> pd.Series:
    return pd.Series(norm.ppf(percentile_series, loc=600, scale=100))

# 6d. Apply the function and cast to integer
df_final = df_final.withColumn(
    "synthetic_agri_cibil",
    map_to_normal_udf(col("percentile_clipped")).cast("int")
)

# 7. Display the results
display(df_final.select(
    "rainfall_mm", "soil_quality_index", "farm_size_hectares", 
    "sunlight_hours", "fertilizer_kg", "crop_yield", 
    "weighted_score", "synthetic_agri_cibil"
).orderBy(col("synthetic_agri_cibil").desc()))

# Optional: Show score distribution to verify realistic spread
print("\n=== CIBIL Score Distribution ===")
df_final.groupBy(
    when(col("synthetic_agri_cibil") < 500, "300-499: Poor")
    .when(col("synthetic_agri_cibil") < 650, "500-649: Fair")
    .when(col("synthetic_agri_cibil") < 750, "650-749: Good")
    .when(col("synthetic_agri_cibil") < 850, "750-849: Very Good")
    .otherwise("850-900: Excellent")
    .alias("score_range")
).count().orderBy("score_range").show()

# Save it as your Gold Table
# df_final.drop("percentile", "percentile_clipped").write.format("delta").mode("overwrite").saveAsTable(f"{path_table}.farmers_scored_gold")

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Bring only the CIBIL score column into memory as a Pandas DataFrame
# (This is safe and fast since it's just one integer column)
pdf_cibil = df_final.select("synthetic_agri_cibil").toPandas()

# 2. Set up the plot aesthetics
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# 3. Create the histogram with a Kernel Density Estimate (KDE) curve
sns.histplot(
    pdf_cibil['synthetic_agri_cibil'], 
    bins=50, 
    kde=True, 
    color='#1f77b4', 
    edgecolor='black'
)

# 4. Add labels, title, and mean line
plt.title('Distribution of Synthetic Agri-CIBIL Scores', fontsize=16, pad=15)
plt.xlabel('CIBIL Score (300 - 900)', fontsize=14)
plt.ylabel('Number of Farmers', fontsize=14)

# Draw a red dashed line at the Mean (600)
plt.axvline(600, color='red', linestyle='dashed', linewidth=2, label='Target Mean (600)')

# Set strict x-axis limits to represent the CIBIL range
plt.xlim(250, 950)
plt.legend()

# 5. Display the plot in the Databricks notebook
plt.show()

# COMMAND ----------

df_final.describe().display()

# COMMAND ----------

# DBTITLE 1,Cell 9
# MAGIC %pip install kagglehub
# MAGIC
# MAGIC import kagglehub
# MAGIC
# MAGIC # Download latest version
# MAGIC path = kagglehub.dataset_download("architsharma01/loan-approval-prediction-dataset")
# MAGIC
# MAGIC print("Path to dataset files:", path)
# MAGIC !kaggle datasets download -d architsharma01/loan-approval-prediction-dataset

# COMMAND ----------

!unzip /Workspace/Users/mems230005026@iiti.ac.in/Bharat_Bricks_Strawhats/loan-approval-prediction-dataset.zip

# COMMAND ----------

# DBTITLE 1,Load Loan Approval Dataset
# 1. Define paths for loan approval dataset
loan_table_name = "loan_approval"
loan_full_table_name = f"{path_table}.{loan_table_name}"  # Catalog1.Schema1.loan_approval
loan_volume_file_path = f"{path_volume}/loan_approval_dataset.csv"

# 2. Copy the loan approval CSV from Workspace to Unity Catalog Volume
loan_csv_path = "/Workspace/Users/mems230005026@iiti.ac.in/Bharat_Bricks_Strawhats/loan_approval_dataset.csv"
dbutils.fs.cp(f"file:{loan_csv_path}", loan_volume_file_path)
print(f"Successfully copied to Volume: {loan_volume_file_path}")

# 3. Read the CSV file into a PySpark DataFrame
loan_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(loan_volume_file_path)

# 4. Clean column names (remove spaces and special characters)
import re
for col_name in loan_df.columns:
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', col_name.replace(' ', '_'))
    loan_df = loan_df.withColumnRenamed(col_name, clean_name)

print("\n=== Dataset Schema ===")
loan_df.printSchema()
print(f"\nTotal rows: {loan_df.count()}")

# 5. Save as a Delta Table in Unity Catalog
loan_df.write.format("delta").mode("overwrite").saveAsTable(loan_full_table_name)
print(f"\nSuccessfully saved as Unity Catalog table: {loan_full_table_name}")

# 6. Display sample of the data
print("\n=== Sample Data ===")
display(loan_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Run Databricks AutoML
# Re-define variables (needed after Python restart)
catalog = "Catalog1"
schema = "Schema1"
loan_table_name = "loan_approval"
loan_full_table_name = f"{catalog}.{schema}.{loan_table_name}"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

print("=== Loan Approval Prediction Model (sklearn) ===")
print("Using scikit-learn for reliable training on serverless compute\n")

# 1. Load data from Unity Catalog and convert to pandas
print("=== Loading Data ===")
df = spark.table(loan_full_table_name).toPandas()
print(f"Dataset: {len(df)} loan applications")

# 2. Clean and prepare data
print("\n=== Data Preprocessing ===")

# Remove leading/trailing spaces from string columns
for col in ['_education', '_self_employed', '_loan_status']:
    df[col] = df[col].str.strip()

# Filter to binary classification (Approved/Rejected only)
df = df[df['_loan_status'].isin(['Approved', 'Rejected'])]
print(f"Filtered to {len(df)} rows with valid status")

# Show class distribution
print("\n=== Class Distribution ===")
print(df['_loan_status'].value_counts())
print(f"\nClass balance: {df['_loan_status'].value_counts(normalize=True).round(3).to_dict()}")

# 3. Feature Engineering
print("\n=== Feature Engineering ===")

# Separate features and target
X = df.drop(['loan_id', '_loan_status'], axis=1)
y = df['_loan_status']

# Encode categorical variables
label_encoders = {}
for col in ['_education', '_self_employed']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target variable
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)

print(f"Features: {X.shape[1]} total")
print(f"  - Categorical: ['_education', '_self_employed']")
print(f"  - Numerical: {X.shape[1] - 2} features")
print(f"\nTarget classes: {y_encoder.classes_}")

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n=== Data Split ===")
print(f"Training: {len(X_train)} samples")
print(f"Testing:  {len(X_test)} samples")

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n✓ Features standardized")

# 6. Train Multiple Models
print("\n=== Training Models ===")

# Model 1: Logistic Regression
print("\n[1/2] Logistic Regression...")
lr_model = LogisticRegression(max_iter=200, random_state=42, C=1.0)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)
print("  ✓ Complete")

# Model 2: Random Forest
print("[2/2] Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)
print("  ✓ Complete")

# 7. Evaluate Models
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)

models = [
    ("Logistic Regression", lr_pred),
    ("Random Forest", rf_pred)
]

print(f"\n{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
print("="*80)

best_f1 = 0
best_model_name = ""
best_pred = None

for model_name, predictions in models:
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    prec = precision_score(y_test, predictions, average='weighted')
    rec = recall_score(y_test, predictions, average='weighted')
    
    print(f"{model_name:<25} {acc:<12.4f} {f1:<12.4f} {prec:<12.4f} {rec:<12.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = model_name
        best_pred = predictions

print("="*80)
print(f"\n✓ Best Model: {best_model_name} with F1 Score: {best_f1:.4f}\n")

# 8. Detailed Evaluation for Best Model
print("=== Confusion Matrix (Best Model) ===")
cm = confusion_matrix(y_test, best_pred)
print(f"\n                Predicted")
print(f"                {y_encoder.classes_[0]:<12} {y_encoder.classes_[1]:<12}")
print(f"Actual {y_encoder.classes_[0]:<8} {cm[0][0]:<12} {cm[0][1]:<12}")
print(f"       {y_encoder.classes_[1]:<8} {cm[1][0]:<12} {cm[1][1]:<12}")

print("\n=== Classification Report (Best Model) ===")
print(classification_report(y_test, best_pred, target_names=y_encoder.classes_))

# 9. Feature Importance (if Random Forest won)
if best_model_name == "Random Forest":
    print("\n=== Top 10 Most Important Features ===")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    print(feature_importance.to_string(index=False))
else:
    print("\n=== Top 10 Most Influential Features (Logistic Regression) ===")
    feature_coef = pd.DataFrame({
        'feature': X.columns,
        'coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('coefficient', ascending=False).head(10)
    print(feature_coef.to_string(index=False))

# 10. Sample Predictions
print("\n=== Sample Predictions ===")
test_sample = pd.DataFrame({
    'Actual': y_encoder.inverse_transform(y_test[:10]),
    'Predicted': y_encoder.inverse_transform(best_pred[:10]),
    'CIBIL_Score': X_test['_cibil_score'].values[:10],
    'Income': X_test['_income_annum'].values[:10],
    'Loan_Amount': X_test['_loan_amount'].values[:10]
})
print(test_sample.to_string(index=False))

print("\n" + "="*80)
print("✓ MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\n✓ Best Model: {best_model_name}")
print(f"✓ Accuracy: {accuracy_score(y_test, best_pred):.2%}")
print(f"✓ F1 Score: {best_f1:.2%}")
print(f"\n✓ No cache issues - sklearn runs perfectly on serverless!")
print(f"✓ Model ready for deployment and predictions on new loan applications")

# COMMAND ----------

# DBTITLE 1,Log Models to MLflow and Unity Catalog
import mlflow
from mlflow.models import infer_signature
import pandas as pd

# Set MLflow experiment
mlflow.set_experiment("/Users/mems230005026@iiti.ac.in/loan_approval_models")

print("=== Logging Models to MLflow ===")
print("\nThis will register models to Unity Catalog for serving endpoints\n")

# Prepare input example and signature (required for Unity Catalog and serving)
input_example = pd.DataFrame(X_test_scaled[:5], columns=X.columns)
signature = infer_signature(X_test_scaled, rf_pred)

print(f"✓ Model signature: {signature}")
print(f"✓ Input example shape: {input_example.shape}\n")

# Define Unity Catalog model registry path
catalog = "Catalog1"
schema = "Schema1"

# ============================================================
# Model 1: Random Forest (Best performing model)
# ============================================================
rf_model_name = f"{catalog}.{schema}.loan_approval_rf"

print(f"[1/2] Logging Random Forest to {rf_model_name}...")

with mlflow.start_run(run_name="random_forest_loan_approval") as run:
    # Log parameters
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "test_size": 0.2,
        "random_state": 42
    })
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, rf_pred),
        "f1_score": f1_score(y_test, rf_pred, average='weighted'),
        "precision": precision_score(y_test, rf_pred, average='weighted'),
        "recall": recall_score(y_test, rf_pred, average='weighted')
    })
    
    # Log the model with signature and input_example (REQUIRED for Unity Catalog)
    mlflow.sklearn.log_model(
        sk_model=rf_model,
        artifact_path="model",
        registered_model_name=rf_model_name,
        signature=signature,
        input_example=input_example
    )
    
    rf_run_id = run.info.run_id
    print(f"  ✓ Run ID: {rf_run_id}")
    print(f"  ✓ Registered to Unity Catalog: {rf_model_name}")

# ============================================================
# Model 2: Logistic Regression (Baseline model)
# ============================================================
lr_model_name = f"{catalog}.{schema}.loan_approval_lr"

print(f"\n[2/2] Logging Logistic Regression to {lr_model_name}...")

with mlflow.start_run(run_name="logistic_regression_loan_approval") as run:
    # Log parameters
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "max_iter": 200,
        "C": 1.0,
        "test_size": 0.2,
        "random_state": 42
    })
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, lr_pred),
        "f1_score": f1_score(y_test, lr_pred, average='weighted'),
        "precision": precision_score(y_test, lr_pred, average='weighted'),
        "recall": recall_score(y_test, lr_pred, average='weighted')
    })
    
    # Log the model with signature and input_example (REQUIRED for Unity Catalog)
    mlflow.sklearn.log_model(
        sk_model=lr_model,
        artifact_path="model",
        registered_model_name=lr_model_name,
        signature=signature,
        input_example=input_example
    )
    
    lr_run_id = run.info.run_id
    print(f"  ✓ Run ID: {lr_run_id}")
    print(f"  ✓ Registered to Unity Catalog: {lr_model_name}")

print("\n" + "="*80)
print("✓ MODELS LOGGED SUCCESSFULLY!")
print("="*80)
print(f"\n✓ Random Forest: {rf_model_name}")
print(f"✓ Logistic Regression: {lr_model_name}")
print("\n📊 View runs in MLflow UI: Workspace → Machine Learning → Experiments")
print("📦 Models registered to Unity Catalog and ready for serving endpoints")
print("\n" + "="*80)
print("NEXT STEPS: Create Serving Endpoints")
print("="*80)
print("\n1. Via UI:")
print("   • Go to 'Serving' in left sidebar")
print("   • Click 'Create Serving Endpoint'")
print(f"   • Select model: {rf_model_name} (or {lr_model_name})")
print("   • Choose endpoint name (e.g., 'loan-approval-api')")
print("   • Select compute size (Small recommended for testing)")
print("   • Click 'Create'")
print("\n2. Via Python API (run in next cell):")
print("   from databricks.sdk import WorkspaceClient")
print("   from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput")
print("   ")
print("   w = WorkspaceClient()")
print("   w.serving_endpoints.create(")
print("       name='loan-approval-api',")
print(f"       config=EndpointCoreConfigInput(")
print(f"           served_entities=[ServedEntityInput(")
print(f"               entity_name='{rf_model_name}',")
print(f"               entity_version='1',")
print(f"               workload_size='Small',")
print(f"               scale_to_zero_enabled=True")
print(f"           )]")
print(f"       )")
print(f"   )")
print("\n3. Test endpoint in Flask:")
print("   import requests")
print("   ")
print("   response = requests.post(")
print("       'https://<workspace-url>/serving-endpoints/loan-approval-api/invocations',")
print("       headers={'Authorization': 'Bearer <token>'},")
print("       json={'dataframe_records': [")
print("           {'_education': 1, '_self_employed': 0, '_income_annum': 5000000, ...}")
print("       ]}")
print("   )")
print("   prediction = response.json()")

# COMMAND ----------

# DBTITLE 1,Create Serving Endpoint (Python SDK)
# Uncomment and run this cell to programmatically create a serving endpoint

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# # Initialize Databricks client
# w = WorkspaceClient()

# # Define endpoint configuration
# endpoint_name = "loan-approval-api"
# model_name = f"{catalog}.{schema}.loan_approval_rf"  # Use Random Forest (best model)

# print(f"Creating serving endpoint: {endpoint_name}")
# print(f"Model: {model_name}\n")

# try:
#     endpoint = w.serving_endpoints.create(
#         name=endpoint_name,
#         config=EndpointCoreConfigInput(
#             served_entities=[
#                 ServedEntityInput(
#                     entity_name=model_name,
#                     entity_version="1",  # Version 1 of the registered model
#                     workload_size="Small",  # Options: Small, Medium, Large
#                     scale_to_zero_enabled=True  # Auto-scale to zero when idle
#                 )
#             ]
#         )
#     )
#     
#     print(f"✓ Endpoint '{endpoint_name}' created successfully!")
#     print(f"\nEndpoint URL: https://{w.config.host}/serving-endpoints/{endpoint_name}/invocations")
#     print("\n⌛ Endpoint is deploying... Check status in Serving UI")
#     print("   (typically takes 5-10 minutes for first deployment)")
    
# except Exception as e:
#     if "already exists" in str(e):
#         print(f"⚠️ Endpoint '{endpoint_name}' already exists")
#         print("   You can update it via the Serving UI or delete and recreate")
#     else:
#         print(f"❌ Error creating endpoint: {e}")

print("📝 Uncomment the code above to create the endpoint programmatically")
print("\nOr create it via UI: Serving → Create Serving Endpoint")

# COMMAND ----------

# DBTITLE 1,Flask Integration Example
# Flask Backend Integration Example
# Copy this code to your Flask application (app.py)

flask_code = '''
from flask import Flask, request, jsonify
import requests
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Databricks configuration
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")  # e.g., "https://dbc-xxx.cloud.databricks.com"
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")  # Your personal access token
ENDPOINT_NAME = "loan-approval-api"

# Endpoint URL
ENDPOINT_URL = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"

# Initialize preprocessors (train these once and save, or load from saved state)
scaler = StandardScaler()
label_encoders = {}

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict loan approval for a given application.
    
    Expected JSON input:
    {
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 5000000,
        "loan_amount": 10000000,
        "loan_term": 12,
        "cibil_score": 750,
        "residential_assets_value": 5000000,
        "commercial_assets_value": 3000000,
        "luxury_assets_value": 8000000,
        "bank_asset_value": 2000000,
        "no_of_dependents": 2
    }
    """
    try:
        # Parse input
        data = request.get_json()
        
        # Preprocess input (encode categorical variables)
        processed_data = {
            "_education": label_encoders["_education"].transform([data["education"]])[0],
            "_self_employed": label_encoders["_self_employed"].transform([data["self_employed"]])[0],
            "_no_of_dependents": data["no_of_dependents"],
            "_income_annum": data["income_annum"],
            "_loan_amount": data["loan_amount"],
            "_loan_term": data["loan_term"],
            "_cibil_score": data["cibil_score"],
            "_residential_assets_value": data["residential_assets_value"],
            "_commercial_assets_value": data["commercial_assets_value"],
            "_luxury_assets_value": data["luxury_assets_value"],
            "_bank_asset_value": data["bank_asset_value"]
        }
        
        # Create DataFrame and scale
        input_df = pd.DataFrame([processed_data])
        input_scaled = scaler.transform(input_df)
        
        # Call Databricks serving endpoint
        response = requests.post(
            ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "dataframe_records": [processed_data]  # or use input_scaled.tolist()[0]
            }
        )
        
        # Check response
        if response.status_code == 200:
            prediction = response.json()
            
            # Parse prediction (0=Rejected, 1=Approved)
            loan_status = "Approved" if prediction["predictions"][0] == 1 else "Rejected"
            confidence = prediction.get("probabilities", [[0.5, 0.5]])[0][1]  # Probability of approval
            
            return jsonify({
                "status": "success",
                "loan_status": loan_status,
                "confidence": f"{confidence * 100:.1f}%",
                "recommendation": get_recommendation(loan_status, data)
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Model prediction failed: {response.text}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def get_recommendation(status, data):
    """Generate actionable recommendations based on prediction."""
    if status == "Approved":
        return "Your loan application is likely to be approved. Please proceed with documentation."
    else:
        suggestions = []
        if data["cibil_score"] < 650:
            suggestions.append("Improve your CIBIL score above 650")
        if data["income_annum"] < 3000000:
            suggestions.append("Consider applying for a smaller loan amount")
        if data["loan_amount"] / data["income_annum"] > 3:
            suggestions.append("Your loan-to-income ratio is high. Reduce loan amount or increase income.")
        
        return "Your loan may be rejected. Suggestions: " + "; ".join(suggestions)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "loan-approval-api"})

if __name__ == "__main__":
    # Load preprocessors (label encoders and scaler) from saved state
    # For demo purposes, initialize them here
    # In production, load from pickle files trained on your dataset
    
    app.run(debug=True, host="0.0.0.0", port=5000)
'''

print("\n" + "="*80)
print("FLASK BACKEND INTEGRATION CODE")
print("="*80)
print(flask_code)

print("\n" + "="*80)
print("TESTING THE ENDPOINT")
print("="*80)
print("""
# Test endpoint using curl:
curl -X POST https://<workspace-url>/serving-endpoints/loan-approval-api/invocations \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [{
      "_education": 1,
      "_self_employed": 0,
      "_no_of_dependents": 2,
      "_income_annum": 5000000,
      "_loan_amount": 10000000,
      "_loan_term": 12,
      "_cibil_score": 750,
      "_residential_assets_value": 5000000,
      "_commercial_assets_value": 3000000,
      "_luxury_assets_value": 8000000,
      "_bank_asset_value": 2000000
    }]
  }'

# Test from Python:
import requests

response = requests.post(
    "https://<workspace-url>/serving-endpoints/loan-approval-api/invocations",
    headers={"Authorization": "Bearer <token>"},
    json={
        "dataframe_records": [{
            "_education": 1,
            "_self_employed": 0,
            "_no_of_dependents": 2,
            "_income_annum": 5000000,
            "_loan_amount": 10000000,
            "_loan_term": 12,
            "_cibil_score": 750,
            "_residential_assets_value": 5000000,
            "_commercial_assets_value": 3000000,
            "_luxury_assets_value": 8000000,
            "_bank_asset_value": 2000000
        }]
    }
)

print(response.json())
""")

print("\n🔑 To get your Databricks token:")
print("   1. Click your profile icon (top right)")
print("   2. Settings → Developer → Access Tokens")
print("   3. Generate New Token")
print("   4. Copy and use in Flask app as environment variable")

print("\n💾 Save preprocessors (scaler, label_encoders):")
print("   import pickle")
print("   pickle.dump(scaler, open('scaler.pkl', 'wb'))")
print("   pickle.dump(label_encoders, open('encoders.pkl', 'wb'))")
print("\n   # Load in Flask:")
print("   scaler = pickle.load(open('scaler.pkl', 'rb'))")
print("   label_encoders = pickle.load(open('encoders.pkl', 'rb'))")

# COMMAND ----------

# MAGIC %pip install groq

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# DBTITLE 1,Cell 15
# 1. Install and import groq for LLM inference
%pip install groq

import groq
import os

# 2. Set Groq API key (replace with your actual API key)


# 3. Prepare context: feature importances and CIBIL score calculation parameters/weights
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

cibil_params = {
    "Yield Weight": 0.35,
    "Soil Quality Weight": 0.20,
    "Farm Size Weight": 0.15,
    "Rainfall Weight": 0.15,
    "Sunlight Weight": 0.10,
    "Fertilizer Weight": 0.05,
    "Score Mapping": "CIBIL range 300-900, nonlinear mapping"
}

# COMMAND ----------

feature_importance_df

# COMMAND ----------

# DBTITLE 1,Restart Python to Clear Cache
# 4. Compose LLM prompt to assist farmer in getting a loan
prompt = f"""
You are an expert agricultural loan advisor. Here are the feature importances from a Random Forest model predicting loan approval:
{feature_importance_df.head(10).to_string(index=False)}

CIBIL score calculation uses these parameters and weights:
{cibil_params}

Given a farmer's data, provide actionable advice to maximize their synthetic CIBIL score and improve loan approval chances. Suggest practical improvements based on the most important features.

Farmer Data Example:
- Rainfall (mm): 1200
- Soil Quality Index: 7
- Farm Size (hectares): 50
- Sunlight Hours: 9
- Fertilizer (kg): 1500
- Crop Yield: 300

Respond with a concise and brief, step-by-step plan for the farmer.
"""

# 5. Run LLM inference using bharatgenai/Param-1-2.9B-Instruct
client = groq.Client()
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=512
)

# 6. Display LLM advice
print("\n=== LLM Advice for Farmer Loan Assistance ===")
print(response.choices[0].message.content)

# COMMAND ----------

