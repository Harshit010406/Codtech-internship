
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# ----------------------------
# CONFIGURATION
# ----------------------------

RAW_DATA_PATH = "data/raw_data.csv"
CLEAN_DATA_PATH = "data/cleaned_data.csv"
TARGET_COLUMN = "target"

# ----------------------------
# STEP 1: LOAD DATA
# ----------------------------

def load_data(path):
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

# ----------------------------
# STEP 2: BUILD PIPELINE
# ----------------------------

def build_pipeline(df):
    print("Building preprocessing pipeline...")

    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.drop(TARGET_COLUMN)
    categorical_features = df.select_dtypes(include=["object", "category"]).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

# ----------------------------
# STEP 3: TRANSFORM DATA
# ----------------------------

def transform_data(preprocessor, df):
    print("Transforming data...")
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    X_processed = preprocessor.fit_transform(features)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = None

    return X_processed, target, feature_names

# ----------------------------
# STEP 4: SAVE CLEANED DATA
# ----------------------------

def save_cleaned_data(X, y, feature_names, output_path):
    print(f"Saving cleaned data to {output_path}...")
    if feature_names is not None:
        X_df = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X, columns=feature_names)
    else:
        X_df = pd.DataFrame(X)
    df_cleaned = pd.concat([X_df, y.reset_index(drop=True)], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    print("Cleaned data saved.")

# ----------------------------
# MAIN FUNCTION
# ----------------------------

def main():
    df = load_data(RAW_DATA_PATH)
    preprocessor = build_pipeline(df)
    X_processed, y, feature_names = transform_data(preprocessor, df)
    save_cleaned_data(X_processed, y, feature_names, CLEAN_DATA_PATH)

if __name__ == "__main__":
    main()
