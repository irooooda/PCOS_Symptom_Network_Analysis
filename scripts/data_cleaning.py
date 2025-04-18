import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import re
from colorama import init, Fore, Style

# 🎛️ Setup robust + styled logging
init(autoreset=True)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="• %(message)s",
    handlers=[logging.StreamHandler()]
)

# 🎨 Highlight content inside curly braces
def highlight_braces(text, color=Fore.CYAN):
    return re.sub(r"\{(.*?)\}", lambda m: "{" + color + m.group(1) + Style.RESET_ALL + "}", text)

# 🧾 Wrapper for logging
def log(msg):
    logging.info(highlight_braces(msg))


def load_config():
    log("Loading configuration settings...")
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    log("Configuration loaded ✓")
    return config, project_root


def find_excel_file(raw_dir, pattern):
    log(f"Looking for an Excel file in {{ {raw_dir} }} matching pattern '{{ {pattern} }}'...")
    for filename in os.listdir(raw_dir):
        if pattern in filename and filename.endswith(".xlsx"):
            file_path = os.path.join(raw_dir, filename)
            log(f"Found file → {{ {file_path} }}")
            return file_path
    raise FileNotFoundError("No matching Excel file found.")


def load_and_prepare_data(file_path, pcos_label):
    log(f"Reading Excel file → {{ {file_path} }}")
    xls = pd.ExcelFile(file_path)
    sheet_name = next(
        (s for s in xls.sheet_names if "data" in s.lower() or "pcos" in s.lower()),
        xls.sheet_names[-1]
    )
    log(f"Using sheet: {{ {sheet_name} }}")
    df = pd.read_excel(xls, sheet_name=sheet_name)

    if df.empty:
        raise ValueError(f"Sheet '{sheet_name}' is empty.")

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    log("Looking for the PCOS indicator column...")
    pcos_cols = []
    for column in df.columns:
        if "pcos" in column.lower():
            values = df[column].astype(str).str.lower().unique()
            if any(v in ["y", "n", "yes", "no", "1", "0"] for v in values):
                pcos_cols.append(column)

    if not pcos_cols:
        raise KeyError("Could not locate a valid PCOS (Y/N) column.")

    selected_col = pcos_cols[0]
    log(f"PCOS column found → Renaming '{{ {selected_col} }}' → '{{ {pcos_label} }}'")
    df.rename(columns={selected_col: pcos_label}, inplace=True)

    log(f"Data loaded ✓ Shape: {{ {df.shape[0]} }} rows × {{ {df.shape[1]} }} columns")
    return df


def clean_data(df, config):
    threshold = config["cleaning"]["missing_threshold"]
    pcos_label = config["columns"]["pcos_label"]

    log(f"Cleaning data (missing threshold: {{ {threshold * 100:.0f}% }} )...")
    log("Removing '_bin_' columns...")
    df = df.loc[:, ~df.columns.str.contains("_bin_", regex=False)]

    log("Converting text-based columns to numeric (where possible)...")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dropped = df.columns[df.isnull().mean() >= threshold].tolist()
    if dropped:
        log(f"Dropping {{ {len(dropped)} }} column(s) with too many missing values: {{ {dropped} }}")

    columns_to_keep = df.columns[df.isnull().mean() < threshold].tolist()
    if pcos_label not in columns_to_keep:
        columns_to_keep.append(pcos_label)
    df = df[columns_to_keep]

    log("Filling missing values...")
    for col in df.columns:
        if df[col].dtype == "object":
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    df = df[[pcos_label] + [col for col in df.columns if col != pcos_label]]
    log(f"Cleaning complete ✓ Final shape: {{ {df.shape[0]} }} rows × {{ {df.shape[1]} }} columns")
    return df


def is_discrete_numeric_category(series, min_unique=4, max_unique=20):
    if not pd.api.types.is_numeric_dtype(series):
        return False
    unique_vals = series.dropna().unique()
    return (min_unique <= len(unique_vals) <= max_unique) and np.all(unique_vals == unique_vals.astype(int))


def bin_numeric_columns(df, config):
    log("Binning numeric features...")

    exclude = config["cleaning"]["exclude_cols_from_binning"]
    pcos_label = config["columns"]["pcos_label"]
    binned_columns = {}
    ml_df = pd.DataFrame()
    num_bins = config["cleaning"].get("number_of_bins", 5)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude and col != pcos_label]
    log(f"Found {{ {len(numeric_cols)} }} numeric column(s) for binning/encoding")

    for feature in numeric_cols:
        log(f"Processing → {{ {feature} }}")
        if df[feature].nunique() <= 2:
            log(f"Skipping '{{ {feature} }}' — not enough unique values")
            ml_df[feature] = df[feature]
            continue

        if is_discrete_numeric_category(df[feature]):
            log(f"Treating '{{ {feature} }}' as discrete category")
            dummies = pd.get_dummies(df[feature].astype(int), prefix=feature)
            ml_df = pd.concat([ml_df, dummies.astype(int)], axis=1)
            binned_columns[feature] = list(dummies.columns)
            continue

        unique_vals = np.unique(df[feature].dropna())
        bins = min(num_bins, len(unique_vals) - 1)
        if bins < 1:
            log(f"Skipping '{{ {feature} }}' — not enough values to bin")
            continue

        bin_edges = np.linspace(df[feature].min(), df[feature].max(), bins + 1)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            log(f"Skipping '{{ {feature} }}' — invalid bin edges")
            continue

        ranges = []
        for i in range(len(bin_edges) - 1):
            start = int(round(bin_edges[i]))
            end = int(round(bin_edges[i + 1]))
            if i > 0:
                start += 1
            ranges.append(f"{start}-{end}")

        labels = list(dict.fromkeys([f"{feature} ({r})" for r in ranges]))
        if len(labels) != len(bin_edges) - 1:
            log(f"Skipping '{{ {feature} }}' — label mismatch")
            continue

        binned = pd.cut(df[feature], bins=bin_edges, labels=labels, include_lowest=True, ordered=False)
        binned = pd.Categorical(binned, categories=labels, ordered=False)
        dummies = pd.get_dummies(binned)
        dummies.columns = labels
        ml_df = pd.concat([ml_df, dummies.astype(int)], axis=1)
        binned_columns[feature] = labels

    log(f"Binning complete ✓ Processed {{ {len(binned_columns)} }} feature(s)")
    return ml_df, binned_columns


def encode_categoricals(df, ml_df, config):
    log("Encoding categorical features...")

    pcos_label = config["columns"]["pcos_label"]
    categoricals = df.select_dtypes(exclude=["number"]).columns

    for col in categoricals:
        log(f"Encoding → {{ {col} }}")
        df[col] = df[col].astype(str).str.strip().replace({"Yes": 1, "No": 0, "Y": 1, "N": 0})
        if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1}):
            ml_df[col] = df[col].astype(int)
        else:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
            ml_df = pd.concat([ml_df, dummies], axis=1)

    log(f"Encoding complete ✓ {{ {len(categoricals)} }} column(s) encoded")
    ml_df[pcos_label] = df[pcos_label].astype(int)
    ml_df = ml_df[[pcos_label] + [col for col in ml_df.columns if col != pcos_label]]
    return ml_df


def save_outputs(df, ml_df, binned_columns, config, base_dir):
    log("Saving files to disk...")
    processed = base_dir / config["data_paths"]["processed_data"]
    processed.mkdir(parents=True, exist_ok=True)

    df.to_csv(processed / config["files"]["cleaned_csv"], index=False)
    log("✔ Cleaned data saved")

    ml_df.to_csv(processed / config["files"]["transformed_csv"], index=False)
    log("✔ Transformed data saved")

    with open(processed / config["files"]["binning_metadata"], "w") as f:
        json.dump(binned_columns, f, indent=2)
    log("✔ Binning metadata saved")

    pcos_label = config["columns"]["pcos_label"]
    df_pcos = ml_df[ml_df[pcos_label] == 1]
    df_no_pcos = ml_df[ml_df[pcos_label] == 0]

    df_pcos.to_csv(processed / config["files"]["patients_with_pcos"], index=False)
    df_no_pcos.to_csv(processed / config["files"]["patients_without_pcos"], index=False)
    log("✔ PCOS cohort files saved")


def main():
    config, base_dir = load_config()
    raw_dir = base_dir / config["data_paths"]["raw_data"]
    file_path = find_excel_file(raw_dir, config["files"]["excel_pattern"])
    log(f"Working with data file: {{ {file_path} }}")

    df = load_and_prepare_data(file_path, config["columns"]["pcos_label"])
    df = clean_data(df, config)

    ml_df, binned_columns = bin_numeric_columns(df, config)
    ml_df = encode_categoricals(df, ml_df, config)

    save_outputs(df, ml_df, binned_columns, config, base_dir)
    log("🎉 All done! Data pipeline complete ✓")


if __name__ == "__main__":
    main()
