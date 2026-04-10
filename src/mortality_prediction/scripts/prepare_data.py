import json
import logging
import os

from mortality_prediction.dataloader import ScalerParams
from mortality_prediction.dataloader import compute_normalization_params
from mortality_prediction.dataloader import get_dataset_a
from mortality_prediction.dataloader import get_dataset_b
from mortality_prediction.dataloader import get_dataset_c
from mortality_prediction.normalize_data import convert_to_table_format
from mortality_prediction.normalize_data import convert_to_triplet_format
from mortality_prediction.normalize_data import convert_to_vector_format
from mortality_prediction.normalize_data import scale_patients
from mortality_prediction.utils import DATA_DIR


logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

_NORM_PARAMS_PATH = os.path.join(DATA_DIR, "set_a_normalization_params.json")

_SETS = [
    ("set-a", "set_a", get_dataset_a),
    ("set-b", "set_b", get_dataset_b),
    ("set-c", "set_c", get_dataset_c),
]

if __name__ == "__main__":
    if os.path.exists(_NORM_PARAMS_PATH):
        with open(_NORM_PARAMS_PATH) as f:
            norm_params = {
                k: ScalerParams.model_validate(v) for k, v in json.load(f).items()
            }
        logger.info(
            f"Loaded {len(norm_params)} normalization params from {_NORM_PARAMS_PATH}"
        )
    else:
        logger.info("Normalization params not found — computing from set-a ...")
        patients_a = get_dataset_a()
        norm_params = compute_normalization_params(patients_a)
        with open(_NORM_PARAMS_PATH, "w") as f:
            json.dump({k: v.model_dump() for k, v in norm_params.items()}, f, indent=2)
        logger.info(
            f"Saved {len(norm_params)} normalization params -> {_NORM_PARAMS_PATH}"
        )

    for set_name, safe_name, loader in _SETS:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Processing {set_name} ...")

        patients = loader()
        scaled = scale_patients(patients, norm_params)

        # ── Normalized exports ────────────────────────────────────────────────
        convert_to_vector_format(safe_name, scaled, fill_missing=True)
        # → set_{a,b,c}_vector_imputed.parquet
        convert_to_vector_format(safe_name, scaled, fill_missing=False)
        # → set_{a,b,c}_vector_not_imputed.parquet
        convert_to_table_format(safe_name, scaled, fill_missing=True)
        # → set_{a,b,c}_table_imputed.parquet
        convert_to_table_format(safe_name, scaled, fill_missing=False)
        # → set_{a,b,c}_table_not_imputed.parquet
        convert_to_triplet_format(safe_name, scaled)
        # → set_{a,b,c}_triplet.parquet

        # ── Raw exports: no normalization, no imputation (for LLM prompts) ─────
        convert_to_vector_format(f"{safe_name}_raw", patients, fill_missing=False)
        # → set_{a,b,c}_raw_vector_not_imputed.parquet
        convert_to_table_format(f"{safe_name}_raw", patients, fill_missing=False)
        # → set_{a,b,c}_raw_table_not_imputed.parquet

    logger.info("\nAll data files prepared successfully.")
