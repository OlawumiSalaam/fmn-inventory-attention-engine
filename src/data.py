"""Data loading and preparation for the FMN Inventory Attention Engine.

The module implements deterministic preparation rules derived from the project
EDA and data preparation specification. It does not contain forecasting,
inventory risk scoring, or UI logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "date",
    "sku_id",
    "category",
    "units_sold",
    "units_received",
    "closing_stock",
    "lead_time_days",
)


@dataclass(frozen=True)
class PreparationAudit:
    """Summary of deterministic transformations applied to the source data."""

    source_rows: int
    rows_after_deduplication: int
    duplicate_rows_removed: int
    missing_units_sold_before: int
    units_sold_reconstructed: int
    missing_units_sold_after: int
    missing_closing_stock: int
    logical_category_count: int
    sku_count: int
    established_sku_count: int
    new_sku_count: int


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the assessment CSV and validate its required columns.

    Args:
        path: Path to the source CSV file.

    Returns:
        DataFrame containing the source observations.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path)

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def validate_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Convert fields to expected types without silently filling invalid values."""
    result = df.copy()

    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["sku_id"] = result["sku_id"].astype("string")
    result["category"] = result["category"].astype("string")

    numeric_columns = [
        "units_sold",
        "units_received",
        "closing_stock",
        "lead_time_days",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if result["date"].isna().any():
        raise ValueError("Invalid date values were found.")

    return result


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate source rows and return the removal count."""
    duplicate_count = int(df.duplicated().sum())
    return df.drop_duplicates().copy(), duplicate_count


def normalise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise category whitespace and casing."""
    result = df.copy()
    result["category"] = result["category"].str.strip().str.title()
    return result


def add_stock_balance_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Add stock balance diagnostics without altering source stock values."""
    result = df.sort_values(["sku_id", "date"]).copy()
    result["prev_closing_stock"] = result.groupby("sku_id")["closing_stock"].shift(1)

    result["expected_closing_stock"] = (
        result["prev_closing_stock"]
        + result["units_received"]
        - result["units_sold"]
    )

    comparable = result[
        [
            "prev_closing_stock",
            "units_received",
            "units_sold",
            "closing_stock",
        ]
    ].notna().all(axis=1)

    result["stock_balance_valid"] = False
    result.loc[comparable, "stock_balance_valid"] = (
        (
            result.loc[comparable, "expected_closing_stock"]
            - result.loc[comparable, "closing_stock"]
        ).abs()
        < 1e-9
    )

    result["stock_balance_difference"] = (
        result["expected_closing_stock"] - result["closing_stock"]
    )

    result["stock_balance_clipped"] = (
        comparable
        & result["closing_stock"].eq(0)
        & result["expected_closing_stock"].lt(0)
    )

    return result


def reconstruct_missing_units_sold(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct missing demand only where stock balance is demonstrably valid.

    Rows at zero stock are excluded from exact reconstruction because stock
    clipping means the observed balance does not identify true demand.
    """
    result = df.copy()
    result["units_sold_original"] = result["units_sold"]
    result["units_sold_source"] = "observed"

    reconstructed = (
        result["prev_closing_stock"]
        + result["units_received"]
        - result["closing_stock"]
    )

    # When units_sold is missing, stock_balance_valid cannot be evaluated
    # from the observed demand column. Instead, require the stock inputs to
    # exist, closing stock to be positive (avoiding zero-stock clipping), and
    # the implied demand to be non-negative.
    eligible = (
        result["units_sold"].isna()
        & result["prev_closing_stock"].notna()
        & result["units_received"].notna()
        & result["closing_stock"].notna()
        & result["closing_stock"].gt(0)
        & reconstructed.ge(0)
    )

    result.loc[eligible, "units_sold"] = reconstructed.loc[eligible]
    result.loc[eligible, "units_sold_source"] = "stock_balance_reconstructed"

    return result


def classify_skus(df: pd.DataFrame) -> pd.DataFrame:
    """Classify SKUs by available history and add limited-history indicators."""
    result = df.copy()

    history = result.groupby("sku_id")["date"].transform("nunique")
    result["history_days"] = history

    result["sku_status"] = result["history_days"].apply(
        lambda days: "new" if days < 30 else "established"
    )
    result["limited_history"] = result["sku_status"].eq("new")

    return result


def add_data_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level data quality flags used downstream."""
    result = df.copy()

    result["closing_stock_missing"] = result["closing_stock"].isna()
    result["lead_time_missing"] = result["lead_time_days"].isna()
    result["stockout"] = result["closing_stock"].eq(0)

    lead_time_stats = result.groupby("sku_id")["lead_time_days"].transform("nunique")
    result["lead_time_inconsistent"] = lead_time_stats.gt(1)

    return result


def prepare_data(path: str | Path) -> tuple[pd.DataFrame, PreparationAudit]:
    """Run the complete deterministic preparation workflow."""
    raw = load_data(path)
    source_rows = len(raw)

    result = validate_and_cast(raw)
    result, duplicate_count = remove_exact_duplicates(result)
    result = normalise_categories(result)
    result = add_stock_balance_diagnostics(result)

    missing_before = int(result["units_sold"].isna().sum())
    result = reconstruct_missing_units_sold(result)
    missing_after = int(result["units_sold"].isna().sum())

    result = classify_skus(result)
    result = add_data_quality_flags(result)
    result = result.sort_values(["sku_id", "date"]).reset_index(drop=True)

    audit = PreparationAudit(
        source_rows=source_rows,
        rows_after_deduplication=len(result),
        duplicate_rows_removed=duplicate_count,
        missing_units_sold_before=missing_before,
        units_sold_reconstructed=int(
            (result["units_sold_source"] == "stock_balance_reconstructed").sum()
        ),
        missing_units_sold_after=missing_after,
        missing_closing_stock=int(result["closing_stock"].isna().sum()),
        logical_category_count=int(result["category"].nunique()),
        sku_count=int(result["sku_id"].nunique()),
        established_sku_count=int(
            result.loc[result["sku_status"].eq("established"), "sku_id"].nunique()
        ),
        new_sku_count=int(
            result.loc[result["sku_status"].eq("new"), "sku_id"].nunique()
        ),
    )

    return result, audit


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare FMN Project 1 data.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/clean_daily_data.parquet"),
    )
    args = parser.parse_args()

    prepared, audit = prepare_data(args.input_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(args.output, index=False)

    print(audit)
