import json
import logging
import os

import pandas as pd

from mortality_prediction.dataloader import Patient
from mortality_prediction.dataloader import ScalerParams
from mortality_prediction.normalize_data import scale_patients
from mortality_prediction.utils import DATA_DIR


logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

INSPECTION_DIR = os.path.join(DATA_DIR, "patient_inspection")
NORM_PARAMS_PATH = os.path.join(DATA_DIR, "set_a_normalization_params.json")

_SEP = "=" * 80


def _load_norm_params() -> dict[str, ScalerParams]:
    with open(NORM_PARAMS_PATH) as f:
        raw = json.load(f)
    return {k: ScalerParams.model_validate(v) for k, v in raw.items()}


def inspect_patient(
    patient_id: str,
    set_name: str = "set-a",
) -> tuple[
    str, Patient, Patient, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Return seven layered representations of one patient.

    1. raw_txt              — verbatim content of the original .txt file
    2. patient              — Patient object from the JSON cache (original)
    3. patient_scaled       — deep copy with all scalable values normalised
    4. table_df             — 48-row hourly table (normalised, no imputation)
    5. table_df_imputed     — same table after forward-fill then zero-fill
    6. vector_df            — single-row aggregate vector (min/max/last/mean, zero-filled)
    7. triplet_df           — one row per measurement event (t, z, v)

    Sections 4-7 are loaded from the pre-built parquet files produced by
    scripts/prepare_data.py; only sections 1-3 are computed on the fly.
    """
    safe_set = set_name.replace("-", "_")

    # --- 1. raw txt ---
    raw_path = os.path.join(DATA_DIR, set_name, f"{patient_id}.txt")
    with open(raw_path) as f:
        raw_txt = f.read()

    # --- 2. original patient from JSON cache ---
    cache_path = os.path.join(DATA_DIR, f"{safe_set}.json")
    with open(cache_path) as f:
        cache = json.load(f)
    patient = Patient.model_validate(cache[patient_id])

    # --- 3. normalised patient (needed only for the JSON view) ---
    norm_params = _load_norm_params()
    patient_scaled = scale_patients({patient_id: patient}, norm_params)[patient_id]

    # --- 4-7. Load DataFrame views from prepared parquet files ---
    def _load(suffix: str) -> pd.DataFrame:
        path = os.path.join(DATA_DIR, f"{safe_set}_{suffix}.parquet")
        df = pd.read_parquet(path)
        return df[df["PatientID"] == patient_id].reset_index(drop=True)

    table_df = _load("table_not_imputed")
    table_df_imputed = _load("table_imputed")
    vector_df = _load("vector_imputed")
    triplet_df = _load("triplet")

    return (
        raw_txt,
        patient,
        patient_scaled,
        table_df,
        table_df_imputed,
        vector_df,
        triplet_df,
    )


def write_inspection(patient_id: str, set_name: str) -> None:
    """Run inspect_patient and write a human-readable report to INSPECTION_DIR."""
    os.makedirs(INSPECTION_DIR, exist_ok=True)

    (
        raw_txt,
        patient,
        patient_scaled,
        table_df,
        table_df_imputed,
        vector_df,
        triplet_df,
    ) = inspect_patient(patient_id, set_name)

    with pd.option_context(
        "display.max_columns", None, "display.width", 200, "display.max_rows", None
    ):
        table_str = table_df.to_string(index=False)
        table_imp_str = table_df_imputed.to_string(index=False)
        vector_str = vector_df.T.to_string(header=False)
        triplet_str = triplet_df.to_string(index=False)

    content = "\n".join(
        [
            f"PATIENT {patient_id}  |  {set_name}",
            _SEP,
            "",
            f"[ 1 / 7 ]  RAW TXT  —  {set_name}/{patient_id}.txt",
            "-" * 80,
            raw_txt,
            _SEP,
            "",
            "[ 2 / 7 ]  ORIGINAL JSON  —  parsed Patient model",
            "-" * 80,
            patient.model_dump_json(indent=2),
            "",
            _SEP,
            "",
            "[ 3 / 7 ]  NORMALISED JSON  —  scaled Patient model",
            "-" * 80,
            patient_scaled.model_dump_json(indent=2),
            "",
            _SEP,
            "",
            "[ 4 / 7 ]  TABLE FORMAT  —  48 hourly rows (normalised, no imputation)",
            "-" * 80,
            table_str,
            "",
            _SEP,
            "",
            "[ 5 / 7 ]  TABLE FORMAT  —  48 hourly rows (forward-fill then zero-fill)",
            "-" * 80,
            table_imp_str,
            "",
            _SEP,
            "",
            "[ 6 / 7 ]  VECTOR FORMAT  —  single-row aggregate (normalised, zero-filled)",
            "-" * 80,
            vector_str,
            "",
            _SEP,
            "",
            "[ 7 / 7 ]  TRIPLET FORMAT  —  one row per measurement event (t, z, v)",
            "-" * 80,
            triplet_str,
            "",
        ]
    )

    out_path = os.path.join(INSPECTION_DIR, f"{patient_id}.txt")
    with open(out_path, "w") as f:
        f.write(content)
    logger.info(f"Inspection written → {out_path}")


if __name__ == "__main__":
    # 133628 —   1 timepoint  (very sparse)
    # 138817 —  71 timepoints (medium, has MechVent + GCS)
    # 135365 — 202 timepoints (dense, has GCS)
    for pid in ["133628", "138817", "135365"]:
        write_inspection(pid, "set-a")
