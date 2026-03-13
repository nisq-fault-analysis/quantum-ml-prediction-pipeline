"""Raw-data preparation for the Random Forest baseline.

This module is intentionally conservative:
- it never edits the raw CSV in place
- it records which rows were dropped and why
- it keeps data cleaning assumptions explicit
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.schema import DataConfig
from src.data.dataset import prepare_research_table, read_tabular_dataset, validate_required_columns


@dataclass(slots=True)
class PreparedDataset:
    """Bundle the outputs of the raw-data preparation step."""

    cleaned_frame: pd.DataFrame
    invalid_rows: pd.DataFrame
    validation_summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NumericConstraint:
    """Describe one numeric validation rule in a reusable, reportable way."""

    column: str
    min_value: float | None = None
    max_value: float | None = None
    inclusive_min: bool = True
    inclusive_max: bool = True

    @property
    def issue_name(self) -> str:
        return f"out_of_range_{self.column}"

    def is_valid(self, value: float) -> bool:
        """Return whether a numeric value satisfies the configured constraint."""

        if self.min_value is not None:
            if self.inclusive_min and value < self.min_value:
                return False
            if not self.inclusive_min and value <= self.min_value:
                return False

        if self.max_value is not None:
            if self.inclusive_max and value > self.max_value:
                return False
            if not self.inclusive_max and value >= self.max_value:
                return False

        return True

    def expected_condition(self) -> str:
        """Return a human-readable description of the allowed range."""

        lower_bound = ""
        upper_bound = ""

        if self.min_value is not None:
            operator = ">=" if self.inclusive_min else ">"
            lower_bound = f"{operator} {self.min_value}"

        if self.max_value is not None:
            operator = "<=" if self.inclusive_max else "<"
            upper_bound = f"{operator} {self.max_value}"

        if lower_bound and upper_bound:
            return f"{lower_bound} and {upper_bound}"
        if lower_bound:
            return lower_bound
        if upper_bound:
            return upper_bound
        return "any numeric value"


def build_numeric_constraints() -> list[NumericConstraint]:
    """Centralize numeric validation rules so reports can reference them directly."""

    return [
        NumericConstraint(column="gate_depth", min_value=0.0, inclusive_min=True),
        NumericConstraint(
            column="error_rate_gate",
            min_value=0.0,
            max_value=1.0,
            inclusive_min=True,
            inclusive_max=True,
        ),
        NumericConstraint(column="t1_time", min_value=0.0, inclusive_min=False),
        NumericConstraint(column="t2_time", min_value=0.0, inclusive_min=False),
        NumericConstraint(
            column="readout_error",
            min_value=0.0,
            max_value=1.0,
            inclusive_min=True,
            inclusive_max=True,
        ),
        NumericConstraint(column="shots", min_value=0.0, inclusive_min=False),
        NumericConstraint(
            column="fidelity",
            min_value=0.0,
            max_value=1.0,
            inclusive_min=True,
            inclusive_max=True,
        ),
    ]


def _json_ready_value(value: Any) -> Any:
    """Convert pandas and timestamp-like objects into JSON-friendly values."""

    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _append_issue_detail(
    issue_details: list[dict[str, Any]],
    *,
    issue: str,
    column: str,
    expected_condition: str,
    observed_value: Any,
    rule_type: str,
) -> None:
    """Store structured issue metadata for row-level and summary-level reporting."""

    issue_details.append(
        {
            "issue": issue,
            "column": column,
            "expected_condition": expected_condition,
            "observed_value": _json_ready_value(observed_value),
            "rule_type": rule_type,
        }
    )


def _build_validation_rule_book(config: DataConfig) -> dict[str, dict[str, Any]]:
    """Describe the active validation rules in a machine-readable report section."""

    rule_book: dict[str, dict[str, Any]] = {}

    for constraint in build_numeric_constraints():
        rule_book[constraint.issue_name] = {
            "column": constraint.column,
            "expected_condition": constraint.expected_condition(),
            "rule_type": "numeric_range",
        }

    for column in [config.id_column, config.gate_sequence_column, config.label_column]:
        rule_book[f"missing_{column}"] = {
            "column": column,
            "expected_condition": "non-empty string",
            "rule_type": "required_value",
        }

    rule_book["invalid_qubit_count"] = {
        "column": config.qubit_count_column,
        "expected_condition": "> 0",
        "rule_type": "numeric_range",
    }
    rule_book[f"invalid_{config.timestamp_column}"] = {
        "column": config.timestamp_column,
        "expected_condition": "valid ISO8601 timestamp",
        "rule_type": "format",
    }
    rule_book[f"{config.bitstring_column}:bitstring_longer_than_qubit_count"] = {
        "column": config.bitstring_column,
        "expected_condition": "bitstring length <= qubit_count",
        "rule_type": "length",
    }
    rule_book[f"{config.ideal_bitstring_column}:bitstring_longer_than_qubit_count"] = {
        "column": config.ideal_bitstring_column,
        "expected_condition": "bitstring length <= qubit_count",
        "rule_type": "length",
    }

    return rule_book


def _summarize_issue_details(
    all_issue_details: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate invalid values so reports show both expected and observed ranges."""

    summaries: dict[str, dict[str, Any]] = {}
    for detail in all_issue_details:
        issue = str(detail["issue"])
        summary = summaries.setdefault(
            issue,
            {
                "column": detail["column"],
                "expected_condition": detail["expected_condition"],
                "rule_type": detail["rule_type"],
                "count": 0,
                "observed_min": None,
                "observed_max": None,
                "sample_observed_values": [],
            },
        )
        summary["count"] += 1

        observed_value = detail.get("observed_value")
        if isinstance(observed_value, (int, float)) and not isinstance(observed_value, bool):
            if not math.isnan(float(observed_value)):
                current_min = summary["observed_min"]
                current_max = summary["observed_max"]
                summary["observed_min"] = (
                    float(observed_value)
                    if current_min is None
                    else min(float(current_min), float(observed_value))
                )
                summary["observed_max"] = (
                    float(observed_value)
                    if current_max is None
                    else max(float(current_max), float(observed_value))
                )
        elif observed_value is not None:
            samples = summary["sample_observed_values"]
            observed_text = str(observed_value)
            if observed_text not in samples and len(samples) < 5:
                samples.append(observed_text)

    return summaries


def normalize_bitstring(
    bitstring: object,
    expected_length: int | None,
    *,
    align_short_strings: bool = True,
) -> tuple[str | None, bool, str | None]:
    """Normalize a bitstring and optionally left-pad it to the qubit count.

    TODO: Verify with the dataset documentation whether shorter bitstrings truly
    indicate omitted leading zeros. The public CSV strongly suggests that pattern,
    so zero-padding is a reasonable baseline assumption for now.
    """

    if bitstring is None or pd.isna(bitstring):
        return None, False, "missing_bitstring"

    text = str(bitstring).strip()
    if not text:
        return None, False, "empty_bitstring"

    if expected_length is None or expected_length <= 0:
        return None, False, "invalid_qubit_count"

    if any(character not in {"0", "1"} for character in text):
        return None, False, "non_binary_bitstring"

    if len(text) > expected_length:
        return None, False, "bitstring_longer_than_qubit_count"

    if len(text) < expected_length:
        if not align_short_strings:
            return None, False, "bitstring_shorter_than_qubit_count"
        return text.zfill(expected_length), True, None

    return text, False, None


def coerce_dataset_types(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Apply lightweight, explicit type coercion to the raw dataset."""

    table = prepare_research_table(frame, config)
    table = table.copy()
    table["raw_row_index"] = table.index

    string_columns = [
        config.id_column,
        config.gate_sequence_column,
        config.bitstring_column,
        config.ideal_bitstring_column,
        config.timestamp_column,
        config.label_column,
        *config.categorical_columns,
    ]
    for column in string_columns:
        if column in table.columns:
            table[column] = table[column].astype("string").str.strip()

    for column in config.numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    if config.timestamp_column in table.columns:
        table[config.timestamp_column] = pd.to_datetime(
            table[config.timestamp_column],
            errors="coerce",
            utc=True,
            format="ISO8601",
        )

    return table


def _build_row_issues(
    row: pd.Series, config: DataConfig
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    """Collect data-quality issues for one row."""

    issues: list[str] = []
    details: dict[str, Any] = {}
    issue_details: list[dict[str, Any]] = []

    qubit_count_value = row.get(config.qubit_count_column)
    qubit_count = int(qubit_count_value) if pd.notna(qubit_count_value) else None
    if qubit_count is None or qubit_count <= 0:
        issues.append("invalid_qubit_count")
        _append_issue_detail(
            issue_details,
            issue="invalid_qubit_count",
            column=config.qubit_count_column,
            expected_condition="> 0",
            observed_value=qubit_count_value,
            rule_type="numeric_range",
        )

    for column in [config.id_column, config.gate_sequence_column, config.label_column]:
        value = row.get(column)
        if value is None or pd.isna(value) or not str(value).strip():
            issues.append(f"missing_{column}")
            _append_issue_detail(
                issue_details,
                issue=f"missing_{column}",
                column=column,
                expected_condition="non-empty string",
                observed_value=value,
                rule_type="required_value",
            )

    for constraint in build_numeric_constraints():
        value = row.get(constraint.column)
        if pd.isna(value):
            issues.append(f"missing_or_invalid_{constraint.column}")
            _append_issue_detail(
                issue_details,
                issue=f"missing_or_invalid_{constraint.column}",
                column=constraint.column,
                expected_condition=f"numeric and {constraint.expected_condition()}",
                observed_value=value,
                rule_type="numeric_range",
            )
            continue
        numeric_value = float(value)
        if not constraint.is_valid(numeric_value):
            issues.append(constraint.issue_name)
            _append_issue_detail(
                issue_details,
                issue=constraint.issue_name,
                column=constraint.column,
                expected_condition=constraint.expected_condition(),
                observed_value=numeric_value,
                rule_type="numeric_range",
            )

    timestamp = row.get(config.timestamp_column)
    if pd.isna(timestamp):
        issues.append(f"invalid_{config.timestamp_column}")
        _append_issue_detail(
            issue_details,
            issue=f"invalid_{config.timestamp_column}",
            column=config.timestamp_column,
            expected_condition="valid ISO8601 timestamp",
            observed_value=timestamp,
            rule_type="format",
        )

    observed_bitstring, observed_padded, observed_issue = normalize_bitstring(
        row.get(config.bitstring_column),
        qubit_count,
        align_short_strings=config.align_short_bitstrings_to_qubit_count,
    )
    ideal_bitstring, ideal_padded, ideal_issue = normalize_bitstring(
        row.get(config.ideal_bitstring_column),
        qubit_count,
        align_short_strings=config.align_short_bitstrings_to_qubit_count,
    )

    if observed_issue:
        issues.append(f"{config.bitstring_column}:{observed_issue}")
        _append_issue_detail(
            issue_details,
            issue=f"{config.bitstring_column}:{observed_issue}",
            column=config.bitstring_column,
            expected_condition=f"binary string with length <= qubit_count ({qubit_count})",
            observed_value=row.get(config.bitstring_column),
            rule_type="bitstring",
        )
    if ideal_issue:
        issues.append(f"{config.ideal_bitstring_column}:{ideal_issue}")
        _append_issue_detail(
            issue_details,
            issue=f"{config.ideal_bitstring_column}:{ideal_issue}",
            column=config.ideal_bitstring_column,
            expected_condition=f"binary string with length <= qubit_count ({qubit_count})",
            observed_value=row.get(config.ideal_bitstring_column),
            rule_type="bitstring",
        )

    details["bitstring_aligned"] = observed_bitstring
    details["ideal_bitstring_aligned"] = ideal_bitstring
    details["bitstring_was_padded"] = observed_padded
    details["ideal_bitstring_was_padded"] = ideal_padded

    return issues, details, issue_details


def prepare_raw_dataset(frame: pd.DataFrame, config: DataConfig) -> PreparedDataset:
    """Validate, normalize, and split the raw dataset into valid and invalid rows."""

    validate_required_columns(frame, config.required_columns)
    coerced = coerce_dataset_types(frame, config)

    prepared_rows: list[dict[str, Any]] = []
    invalid_row_records: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    all_issue_details: list[dict[str, Any]] = []

    for _, row in coerced.iterrows():
        issues, details, issue_details = _build_row_issues(row, config)
        issue_text = ";".join(issues)
        expected_text = ";".join(
            f"{detail['column']}: {detail['expected_condition']}" for detail in issue_details
        )

        prepared_row = row.to_dict()
        prepared_row.update(details)
        prepared_row["validation_issues"] = issue_text
        prepared_row["validation_expected_conditions"] = expected_text
        prepared_row["validation_issue_details"] = json.dumps(issue_details)
        prepared_row["is_valid_for_modeling"] = not issues

        if issues:
            invalid_record = prepared_row.copy()
            invalid_row_records.append(invalid_record)
            for issue in issues:
                reason_counts[issue] = reason_counts.get(issue, 0) + 1
            all_issue_details.extend(issue_details)

        prepared_rows.append(prepared_row)

    prepared_frame = pd.DataFrame(prepared_rows)
    if config.drop_invalid_rows:
        cleaned_frame = prepared_frame.loc[prepared_frame["is_valid_for_modeling"]].copy()
    else:
        cleaned_frame = prepared_frame.copy()

    invalid_rows = pd.DataFrame(invalid_row_records)
    validation_summary = {
        "rows_read": int(len(coerced)),
        "rows_valid": int(prepared_frame["is_valid_for_modeling"].sum()),
        "rows_invalid": int((~prepared_frame["is_valid_for_modeling"]).sum()),
        "missing_values_by_column": coerced.isna().sum().astype(int).to_dict(),
        "invalid_reason_counts": reason_counts,
        "validation_rules": _build_validation_rule_book(config),
        "invalid_issue_details": _summarize_issue_details(all_issue_details),
        "bitstring_padding_counts": {
            "bitstring_was_padded": int(prepared_frame["bitstring_was_padded"].sum()),
            "ideal_bitstring_was_padded": int(prepared_frame["ideal_bitstring_was_padded"].sum()),
        },
        "cleaned_columns": cleaned_frame.columns.tolist(),
    }

    return PreparedDataset(
        cleaned_frame=cleaned_frame,
        invalid_rows=invalid_rows,
        validation_summary=validation_summary,
    )


def load_and_prepare_raw_dataset(config: DataConfig) -> PreparedDataset:
    """High-level helper used by the CLI script."""

    raw_frame = read_tabular_dataset(config)
    return prepare_raw_dataset(raw_frame, config)
