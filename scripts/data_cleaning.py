import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# 🛠 Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_config():
    logging.info("🔄 Loading configuration file.")
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"❌ config.yaml not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"✅ Configuration loaded from {config_path}")
    return config, project_root

def find_excel_file(raw_dir, pattern):
    logging.info(f"🔍 Searching for Excel file in {raw_dir} with pattern '{pattern}'.")
    for f in os.listdir(raw_dir):
        if pattern in f and f.endswith(".xlsx"):
            file_path = os.path.join(raw_dir, f)
            logging.info(f"✅ Found matching Excel file: {file_path}")
            return file_path
    raise FileNotFoundError("❌ No matching Excel file found.")

def load_and_prepare_data(file_path, pcos_label):
    logging.info(f"📖 Loading data from Excel file: {file_path}")
    xls = pd.ExcelFile(file_path)
    sheet_name = next((s for s in xls.sheet_names if "Data" in s or "PCOS" in s), xls.sheet_names[-1])
    logging.info(f"✅ Loaded sheet: {sheet_name}")
    df = pd.read_excel(xls, sheet_name=sheet_name)

    logging.info("🔄 Cleaning column names.")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    logging.info(f"🔍 Searching for PCOS column containing 'Y' or 'N'.")
    possible_pcos_cols = [col for col in df.columns if "pcos" in col and ("y" in col or "n" in col)]
    if not possible_pcos_cols:
        raise KeyError("❌ PCOS (Y/N) column not found.")

    logging.info(f"✅ Renaming PCOS column to '{pcos_label}'")
    df.rename(columns={possible_pcos_cols[0]: pcos_label}, inplace=True)

    logging.info(f"✅ Data loaded and prepared with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def clean_data(df, config):
    logging.info(f"🔄 Cleaning data based on missing value threshold of {config['cleaning']['missing_threshold']}")
    pcos_label = config["columns"]["pcos_label"]
    threshold = config["cleaning"]["missing_threshold"]

    logging.info(f"🔄 Removing columns with '_bin_' in their name.")
    df = df.loc[:, ~df.columns.str.contains("_bin_", regex=False)]

    logging.info(f"🔄 Converting object columns to numeric where possible.")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logging.info(f"🔄 Removing columns with more than {threshold * 100}% missing values.")
    df = df.loc[:, df.isnull().mean() < threshold]

    logging.info(f"🔄 Filling missing values.")
    for col in df.columns:
        if df[col].dtype == "object":
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    logging.info(f"✅ Data cleaning complete. Remaining data shape: {df.shape[0]} rows and {df.shape[1]} columns.")
    df = df[[pcos_label] + [col for col in df.columns if col != pcos_label]]
    return df

def is_discrete_numeric_category(series, min_unique=4, max_unique=20):
    if not pd.api.types.is_numeric_dtype(series):
        return False
    unique_vals = series.dropna().unique()
    return (min_unique <= len(unique_vals) <= max_unique) and np.all(unique_vals == unique_vals.astype(int))

def bin_numeric_columns(df, config):
    logging.info("🔄 Binning numeric columns.")
    exclude = config["cleaning"]["exclude_cols_from_binning"]
    pcos_label = config["columns"]["pcos_label"]
    binned_columns = {}
    ml_df = pd.DataFrame()
    num_bins = config["cleaning"].get("number_of_bins", 5)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude and col != pcos_label]
    logging.info(f"🔍 Identified {len(numeric_cols)} numeric columns to bin or encode.")

    for col in numeric_cols:
        logging.info(f"🔄 Processing column: {col}")

        if df[col].nunique() <= 2:
            logging.warning(f"⚠️ Skipping '{col}' due to insufficient unique values for binning.")
            ml_df[col] = df[col]
            continue

        if is_discrete_numeric_category(df[col]):
            logging.info(f"⚠️ Treating '{col}' as discrete numeric category.")
            dummies = pd.get_dummies(df[col].astype(int), prefix=col)
            ml_df = pd.concat([ml_df, dummies.astype(int)], axis=1)
            binned_columns[col] = list(dummies.columns)
            continue

        unique_vals = np.unique(df[col].dropna())
        bins = min(num_bins, len(unique_vals) - 1)
        if bins < 1:
            logging.warning(f"⚠️ Skipping '{col}' due to insufficient unique values for binning.")
            continue

        bin_edges = np.linspace(df[col].min(), df[col].max(), bins + 1)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            logging.warning(f"⚠️ Skipping '{col}' due to invalid bin edges.")
            continue

        ranges = []
        for i in range(len(bin_edges) - 1):
            start = int(round(bin_edges[i]))
            end = int(round(bin_edges[i + 1]))
            if i > 0:
                start += 1
            ranges.append(f"{start}-{end}")

        labels = list(dict.fromkeys([f"{col} ({r})" for r in ranges]))
        if len(labels) != len(bin_edges) - 1:
            logging.warning(f"⚠️ Skipping '{col}' due to bin label mismatch.")
            continue

        cut_data = pd.cut(df[col], bins=bin_edges, labels=labels, include_lowest=True, ordered=False)
        cut_data = pd.Categorical(cut_data, categories=labels, ordered=False)
        dummies = pd.get_dummies(cut_data)
        dummies.columns = labels
        ml_df = pd.concat([ml_df, dummies.astype(int)], axis=1)
        binned_columns[col] = labels

    logging.info(f"✅ Processed {len(binned_columns)} numeric columns (binned or dummified).")
    return ml_df, binned_columns

def encode_categoricals(df, ml_df, config):
    logging.info(f"🔄 Encoding categorical columns.")
    pcos_label = config["columns"]["pcos_label"]
    categoricals = df.select_dtypes(exclude=["number"]).columns

    for col in categoricals:
        logging.info(f"🔄 Encoding column: {col}")
        df[col] = df[col].astype(str).str.strip().replace({"Yes": 1, "No": 0, "Y": 1, "N": 0})
        if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1}):
            ml_df[col] = df[col].astype(int)
        else:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
            ml_df = pd.concat([ml_df, dummies], axis=1)

    logging.info(f"✅ Encoded {len(categoricals)} categorical columns.")
    ml_df[pcos_label] = df[pcos_label].astype(int)
    ml_df = ml_df[[pcos_label] + [col for col in ml_df.columns if col != pcos_label]]
    return ml_df

def save_outputs(df, ml_df, binned_columns, config, base_dir):
    logging.info(f"🔄 Saving cleaned and transformed data.")
    processed = base_dir / config["data_paths"]["processed_data"]
    processed.mkdir(parents=True, exist_ok=True)

    df.to_csv(processed / config["files"]["cleaned_csv"], index=False)
    logging.info(f"✅ Saved cleaned data to {processed / config['files']['cleaned_csv']}")

    ml_df.to_csv(processed / config["files"]["transformed_csv"], index=False)
    logging.info(f"✅ Saved transformed data to {processed / config['files']['transformed_csv']}")

    with open(processed / config["files"]["binning_metadata"], "w") as f:
        json.dump(binned_columns, f, indent=2)
    logging.info(f"✅ Saved binning metadata to {processed / config['files']['binning_metadata']}")

    pcos_label = config["columns"]["pcos_label"]
    df_pcos = ml_df[ml_df[pcos_label] == 1]
    df_no_pcos = ml_df[ml_df[pcos_label] == 0]

    df_pcos.to_csv(processed / config["files"]["patients_with_pcos"], index=False)
    df_no_pcos.to_csv(processed / config["files"]["patients_without_pcos"], index=False)
    logging.info(f"✅ Saved PCOS cohort to {processed / config['files']['patients_with_pcos']}")
    logging.info(f"✅ Saved non-PCOS cohort to {processed / config['files']['patients_without_pcos']}")

def main():
    config, base_dir = load_config()
    raw_dir = base_dir / config["data_paths"]["raw_data"]
    file_path = find_excel_file(raw_dir, config["files"]["excel_pattern"])
    logging.info(f"📁 Using dataset: {file_path}")

    df = load_and_prepare_data(file_path, config["columns"]["pcos_label"])
    df = clean_data(df, config)

    ml_df, binned_columns = bin_numeric_columns(df, config)
    ml_df = encode_categoricals(df, ml_df, config)

    save_outputs(df, ml_df, binned_columns, config, base_dir)
    logging.info("✅ All outputs saved successfully.")

if __name__ == "__main__":
    main()
