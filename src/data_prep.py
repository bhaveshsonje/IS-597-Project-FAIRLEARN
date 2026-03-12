"""
data_prep.py — Load and clean OULAD dataset tables, merge into a single student-level dataframe.
"""

import pandas as pd


EARLY_WEEKS = 4  # number of early weeks (each week = 7 days) to consider for VLE activity


def load_tables(data_dir: str) -> dict:
    """Load all relevant OULAD CSV files."""
    tables = {}
    for name in ["studentInfo", "studentAssessment", "studentVle", "assessments"]:
        tables[name] = pd.read_csv(f"{data_dir}/{name}.csv")
    return tables


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert final_result to binary: 1 = success, 0 = risk."""
    mapping = {"Pass": 1, "Distinction": 1, "Fail": 0, "Withdrawn": 0}
    df = df.copy()
    df["target"] = df["final_result"].map(mapping)
    return df


def aggregate_vle(vle: pd.DataFrame, cutoff_day: int = EARLY_WEEKS * 7) -> pd.DataFrame:
    """Aggregate early VLE activity per student (first cutoff_day days)."""
    early = vle[vle["date"] <= cutoff_day]
    agg = (
        early.groupby("id_student")
        .agg(
            total_clicks=("sum_click", "sum"),
            num_activities=("id_site", "nunique"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )
    return agg


def aggregate_assessments(student_assess: pd.DataFrame, assessments: pd.DataFrame,
                           cutoff_day: int = EARLY_WEEKS * 7) -> pd.DataFrame:
    """Aggregate early assessment scores per student."""
    merged = student_assess.merge(assessments[["id_assessment", "date"]], on="id_assessment", how="left")
    early = merged[merged["date"] <= cutoff_day]
    agg = (
        early.groupby("id_student")
        .agg(
            avg_early_score=("score", "mean"),
            num_assessments_taken=("id_assessment", "count"),
        )
        .reset_index()
    )
    return agg


def build_dataset(data_dir: str) -> pd.DataFrame:
    """Full pipeline: load → merge → clean → return student-level dataframe."""
    tables = load_tables(data_dir)

    info = binarize_target(tables["studentInfo"])
    vle_agg = aggregate_vle(tables["studentVle"])
    assess_agg = aggregate_assessments(tables["studentAssessment"], tables["assessments"])

    df = (
        info
        .merge(vle_agg, on="id_student", how="left")
        .merge(assess_agg, on="id_student", how="left")
    )

    # Fill missing VLE / assessment features with 0 (no early activity recorded)
    fill_cols = ["total_clicks", "num_activities", "active_days",
                 "avg_early_score", "num_assessments_taken"]
    df[fill_cols] = df[fill_cols].fillna(0)

    # Drop rows where target is missing
    df = df.dropna(subset=["target"])

    return df
