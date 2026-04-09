import json
import logging
import os

import pandas as pd

from mortality_prediction.dataloader import Patient
from mortality_prediction.dataloader import ScalerParams
from mortality_prediction.normalize_data import _build_patient_vector
from mortality_prediction.normalize_data import build_dataset
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
) -> tuple[str, Patient, Patient, pd.DataFrame, pd.DataFrame]:
    """
    Return five layered representations of one patient:

    1. raw_txt         — verbatim content of the original .txt file
    2. patient         — Patient object from the JSON cache (original)
    3. patient_scaled  — deep copy with all scalable values normalised
    4. table_df        — 48-row hourly table built from the scaled patient
    5. vector_df       — single-row aggregate vector (min/max/last/mean)
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

    # --- 3. normalised patient ---
    norm_params = _load_norm_params()
    patient_scaled = scale_patients({patient_id: patient}, norm_params)[patient_id]

    # --- 4. table format (48 hourly rows) ---
    table_df = build_dataset({patient_id: patient_scaled})

    # --- 5. vector format (single row) ---
    vector_df = pd.DataFrame([_build_patient_vector(patient_id, patient_scaled)])

    return raw_txt, patient, patient_scaled, table_df, vector_df


def write_inspection(patient_id: str, set_name: str) -> None:
    """Run inspect_patient and write a human-readable report to INSPECTION_DIR."""
    os.makedirs(INSPECTION_DIR, exist_ok=True)

    raw_txt, patient, patient_scaled, table_df, vector_df = inspect_patient(
        patient_id, set_name
    )

    with pd.option_context(
        "display.max_columns", None, "display.width", 200, "display.max_rows", None
    ):
        table_str = table_df.to_string(index=False)
        vector_str = vector_df.T.to_string(header=False)

    content = "\n".join(
        [
            f"PATIENT {patient_id}  |  {set_name}",
            _SEP,
            "",
            f"[ 1 / 5 ]  RAW TXT  —  {set_name}/{patient_id}.txt",
            "-" * 80,
            raw_txt,
            _SEP,
            "",
            "[ 2 / 5 ]  ORIGINAL JSON  —  parsed Patient model",
            "-" * 80,
            patient.model_dump_json(indent=2),
            "",
            _SEP,
            "",
            "[ 3 / 5 ]  NORMALISED JSON  —  scaled Patient model",
            "-" * 80,
            patient_scaled.model_dump_json(indent=2),
            "",
            _SEP,
            "",
            "[ 4 / 5 ]  TABLE FORMAT  —  48 hourly rows (normalised)",
            "-" * 80,
            table_str,
            "",
            _SEP,
            "",
            "[ 5 / 5 ]  VECTOR FORMAT  —  single-row aggregate (normalised)",
            "-" * 80,
            vector_str,
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
