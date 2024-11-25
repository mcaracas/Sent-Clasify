import os
import pandas as pd


def load_comments(filepath: str) -> pd.DataFrame:
    """Load comments from an Excel file into a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    return pd.read_excel(filepath)


def save_results(df: pd.DataFrame, output_filepath: str):
    """Save DataFrame results to an Excel file."""
    df.to_excel(output_filepath, index=False)
    print(f"Results written to {output_filepath}")
