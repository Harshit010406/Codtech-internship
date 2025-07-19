# ETL Pipeline with Pandas and Scikit-learn

## Overview
This repository contains a Python script that performs automated ETL (Extract, Transform, Load) for tabular data using `pandas` and `scikit-learn`.

## Features
- Extracts data from CSV
- Cleans missing values
- Encodes categorical features
- Scales numerical features
- Saves cleaned dataset to new CSV

## File Structure
```
etl_project/
├── data/
│   └── raw_data.csv
├── etl_pipeline.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the Pipeline

```bash
python etl_pipeline.py
```