# Databricks notebook source
# Notebook 1: delta_setup.py
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from delta.tables import DeltaTable

spark = SparkSession.builder.getOrCreate()

# Create catalog/schema
spark.sql("CREATE CATALOG IF NOT EXISTS kisancredit")
spark.sql("CREATE SCHEMA IF NOT EXISTS kisancredit.agri")

# Farmer profiles table
spark.sql("""
  CREATE TABLE IF NOT EXISTS kisancredit.agri.farmer_profiles (
    farmer_id STRING, location STRING, land_size_ha DOUBLE,
    fert_level STRING, irrig_type STRING, last_yield DOUBLE,
    created_at TIMESTAMP
  ) USING DELTA TBLPROPERTIES ('delta.enableChangeDataFeed'='true')
""")

# Loan applications + score results
spark.sql("""
  CREATE TABLE IF NOT EXISTS kisancredit.agri.loan_applications (
    loan_id STRING, farmer_id STRING, income_annum BIGINT,
    loan_amount BIGINT, cibil_score INT, verdict STRING,
    max_eligible BIGINT, submitted_at TIMESTAMP
  ) USING DELTA
""")

# Chat history for KisanBot
spark.sql("""
  CREATE TABLE IF NOT EXISTS kisancredit.agri.chat_history (
    session_id STRING, farmer_id STRING, role STRING,
    message STRING, ts TIMESTAMP
  ) USING DELTA
""")