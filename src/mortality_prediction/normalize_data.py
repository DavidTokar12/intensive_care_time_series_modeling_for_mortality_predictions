import copy
import logging
import math
import os

import pandas as pd

from mortality_prediction.dataloader import Gender
from mortality_prediction.dataloader import ParamType
from mortality_prediction.dataloader import Patient
from mortality_prediction.dataloader import ScalerParams
from mortality_prediction.dataloader import ScalerType
from mortality_prediction.utils import DATA_DIR


logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Hour grid: 1, 2, ..., 48  (ceiling buckets; bucket N covers (N-1, N] hours)
HOURS = list(range(1, 49))

# Ordered time-series column names (one per ParamType, in enum definition order)
TS_COLS: list[str] = [p.value for p in ParamType]

STATIC_COLS: list[str] = ["age", "height_cm", "weight_kg", "gender"]

# Fixed column order for the output parquet
COLUMN_ORDER: list[str] = ["PatientID", "timestamp", *TS_COLS, *STATIC_COLS, "in_hospital_death"]

_GENDER_FLOAT: dict[Gender, float] = {Gender.MALE: 1.0, Gender.FEMALE: 0.0}


def collect_ts_values(patients: dict[str, Patient], param: ParamType) -> list[float]:
    """Collect all observed raw values for *param* across every patient's timeseries."""
    vals: list[float] = []
    for patient in patients.values():
        for tp in patient.timeseries:
            for m in tp.measurements:
                if m.param == param:
                    vals.append(float(m.value.value if hasattr(m.value, "value") else m.value))
    return vals


def collect_static_values(patients: dict[str, Patient], field: str) -> list[float]:
    """Collect all non-null values of static field *field* across every patient."""
    vals: list[float] = []
    for patient in patients.values():
        v = getattr(patient.static, field)
        if v is not None:
            vals.append(float(v))
    return vals


def scale_value(param_name: str, value: float, params: ScalerParams) -> float:
    """Apply the fitted scaler for *param_name* to a single raw *value*."""
    if params.scaler_type == ScalerType.ROBUST:
        return (value - params.center) / params.scale
    else:  # LOG_STANDARD
        return (math.log1p(value) - params.mean) / params.std


def scale_patients(
    patients: dict[str, Patient],
    norm_params: dict[str, ScalerParams],
) -> dict[str, Patient]:
    """
    Return a deep copy of *patients* with every scalable measurement replaced
    by its scaled value.  Params absent from *norm_params* (e.g. GCS, MechVent,
    gender, icu_type) are left untouched.
    """
    scaled = copy.deepcopy(patients)
    for patient in scaled.values():
        # Static continuous fields
        for field in ("age", "height_cm", "weight_kg"):
            if field in norm_params:
                v = getattr(patient.static, field)
                if v is not None:
                    setattr(patient.static, field, scale_value(field, float(v), norm_params[field]))
        # Time-series measurements
        for tp in patient.timeseries:
            for m in tp.measurements:
                if m.param == ParamType.GCS:
                    m.value = (15.0 - float(m.value)) / 12.0
                elif m.param.value in norm_params:
                    raw = float(m.value.value if hasattr(m.value, "value") else m.value)
                    m.value = scale_value(m.param.value, raw, norm_params[m.param.value])
    return scaled


def _build_patient_df(patient_id: str, patient: Patient) -> pd.DataFrame:
    """
    Return a 48-row DataFrame (one row per hour 1-48) for a single patient.

    Timestamp binning: ceiling to the nearest hour so bucket N covers the
    half-open interval (N-1, N].  A measurement at exactly N hours falls
    into bucket N.  Measurements after hour 48 are dropped.  When multiple
    measurements fall into the same bucket the latest one wins.

    Missing observations are left as NaN / pd.NA.
    """
    # Accumulate (hour -> param_name -> value), processing in time order so
    # later readings in the same bin overwrite earlier ones (last value wins).
    hourly: dict[int, dict[str, float]] = {h: {} for h in HOURS}

    for tp in sorted(patient.timeseries, key=lambda x: x.time):
        hour = math.ceil(tp.time.total_seconds() / 3600)
        if hour < 1 or hour > 48:
            continue
        for m in tp.measurements:
            hourly[hour][m.param.value] = float(
                m.value.value if hasattr(m.value, "value") else m.value
            )

    rows = [{"PatientID": patient_id, "timestamp": h, **hourly[h]} for h in HOURS]
    df = pd.DataFrame(rows)

    # Static features — repeated at every row
    s = patient.static
    df["age"] = float(s.age) if s.age is not None else pd.NA
    df["height_cm"] = float(s.height_cm) if s.height_cm is not None else pd.NA
    df["weight_kg"] = float(s.weight_kg) if s.weight_kg is not None else pd.NA
    df["gender"] = _GENDER_FLOAT.get(s.gender, pd.NA) if s.gender is not None else pd.NA
    df["in_hospital_death"] = (
        float(s.in_hospital_death) if s.in_hospital_death is not None else pd.NA
    )

    return df


def build_dataset(patients: dict[str, Patient]) -> pd.DataFrame:
    """
    Combine all patients into a single DataFrame with the canonical column order.
    Columns not observed for any patient in this set are created as all-NA.
    """
    frames = [_build_patient_df(pid, p) for pid, p in patients.items()]
    df = pd.concat(frames, ignore_index=True)

    # Ensure every expected column exists (some params may be absent in a set)
    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[COLUMN_ORDER]

    df["timestamp"] = df["timestamp"].astype("int16")

    return df


def convert_to_table_format(set_name: str, patients: dict[str, Patient]) -> str:
    """
    Build the 48-row-per-patient hourly table and save it as a parquet file.
    Returns the path of the saved file.
    """
    logger.info(f"Building table format for {set_name} ({len(patients)} patients)...")
    df = build_dataset(patients)

    out_path = os.path.join(DATA_DIR, f"{set_name.replace('-', '_')}_table.parquet")
    df.to_parquet(out_path, index=False)

    rows, cols = df.shape
    logger.info(f"Saved {rows} rows x {cols} columns -> {out_path}")
    return out_path


def _build_patient_vector(patient_id: str, patient: Patient) -> dict:
    """
    Aggregate a single patient's timeseries into one flat feature vector.

    For each time-series param: min / max / last / mean across all observations.
    Static fields and label are appended at the end.
    Missing values are left as pd.NA.
    """
    param_vals: dict[str, list[float]] = {}
    for tp in sorted(patient.timeseries, key=lambda x: x.time):
        for m in tp.measurements:
            val = float(m.value.value if hasattr(m.value, "value") else m.value)
            param_vals.setdefault(m.param.value, []).append(val)

    row: dict = {"PatientID": patient_id}
    for param in ParamType:
        vals = param_vals.get(param.value, [])
        if vals:
            row[f"{param.value}_min"] = min(vals)
            row[f"{param.value}_max"] = max(vals)
            row[f"{param.value}_last"] = vals[-1]
            row[f"{param.value}_mean"] = sum(vals) / len(vals)
        else:
            row[f"{param.value}_min"] = pd.NA
            row[f"{param.value}_max"] = pd.NA
            row[f"{param.value}_last"] = pd.NA
            row[f"{param.value}_mean"] = pd.NA

    s = patient.static
    row["age"] = float(s.age) if s.age is not None else pd.NA
    row["height_cm"] = float(s.height_cm) if s.height_cm is not None else pd.NA
    row["weight_kg"] = float(s.weight_kg) if s.weight_kg is not None else pd.NA
    row["gender"] = _GENDER_FLOAT.get(s.gender, pd.NA) if s.gender is not None else pd.NA
    row["in_hospital_death"] = (
        float(s.in_hospital_death) if s.in_hospital_death is not None else pd.NA
    )
    return row


def convert_to_vector_format(
    set_name: str,
    patients: dict[str, Patient],
    fill_missing: bool = False,
) -> str:
    """
    Build a one-row-per-patient vector table (min/max/last/mean per TS param
    plus statics and label) and save it as a parquet file.

    If *fill_missing* is True, all pd.NA values are replaced with 0.0.
    Returns the path of the saved file.
    """
    logger.info(f"Building vector format for {set_name} ({len(patients)} patients)...")
    df = pd.DataFrame([_build_patient_vector(pid, p) for pid, p in patients.items()])
    if fill_missing:
        df = df.fillna(0.0)

    out_path = os.path.join(DATA_DIR, f"{set_name.replace('-', '_')}_vector.parquet")
    df.to_parquet(out_path, index=False)

    rows, cols = df.shape
    logger.info(f"Saved {rows} rows x {cols} columns -> {out_path}")
    return out_path
