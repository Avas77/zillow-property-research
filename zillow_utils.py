import os
import requests
import pandas as pd

def download_csv(url: str, dest_path: str) -> str:
  os.makedirs(os.path.dirname(dest_path), exist_ok=True)
  r = requests.get(url)
  r.raise_for_status()

  with open(dest_path, "wb") as f:
    f.write(r.content)

  print(f"Downloaded: {dest_path}")
  return dest_path

def clean_csv(csv_path):
  df = pd.read_csv(csv_path)
  df.iloc[:, 5:] = df.iloc[:, 5:].interpolate(method='linear', axis=1).fillna(method='bfill', axis=1)
  return df


def wide_to_long(df: pd.DataFrame, id_vars=("RegionID", "RegionName", "StateName"), value_name="Value") -> pd.DataFrame:
  df = df[df["RegionType"] == "msa"]
  df_clean = df.drop(["SizeRank", "RegionType"], axis=1)
  df_long = df_clean.melt(id_vars=list(id_vars), var_name="Date", value_name=value_name)
  df_long["Date"] = pd.to_datetime(df_long["Date"])
  return df_long


def download_and_process(
  url: str,
  dest_csv: str,
  value_name: str,
  save_parquet: str = None
) -> pd.DataFrame:
  csv_path = download_csv(url, dest_csv)
  df = clean_csv(csv_path)
  df_long = wide_to_long(df, id_vars=("RegionID", "RegionName", "StateName"), value_name=value_name)
  if save_parquet:
    os.makedirs(os.path.dirname(save_parquet), exist_ok=True)
    df_long.to_parquet(save_parquet)
    print(f"Saved parquet: {save_parquet}")
  return df_long