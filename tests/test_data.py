"""Tests for the FMN Project 1 data preparation pipeline."""

from pathlib import Path

from src.data import prepare_data


def test_assessment_dataset_preparation() -> None:
    """Verify known dataset diagnostics after deterministic preparation."""
    path = Path("data/raw/project1_supply_chain_demand.csv")
    if not path.exists():
        return

    prepared, audit = prepare_data(path)

    assert audit.source_rows == 4551
    assert audit.rows_after_deduplication == 4536
    assert audit.duplicate_rows_removed == 15
    assert audit.missing_units_sold_before == 90
    assert audit.units_sold_reconstructed == 84
    assert audit.missing_closing_stock == 45
    assert audit.logical_category_count == 5
    assert audit.sku_count == 28
    assert audit.established_sku_count == 25
    assert audit.new_sku_count == 3
    assert len(prepared) == 4536
