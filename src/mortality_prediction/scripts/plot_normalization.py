import json
import os

from mortality_prediction.dataloader import ScalerParams
from mortality_prediction.dataloader import get_dataset_a
from mortality_prediction.normalize_data import scale_patients
from mortality_prediction.scripts.data_analysis import plot_static_distributions
from mortality_prediction.scripts.data_analysis import plot_timeseries_distributions
from mortality_prediction.utils import DATA_DIR

NORM_PARAMS_PATH = os.path.join(DATA_DIR, "set_a_normalization_params.json")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")

if __name__ == "__main__":
    print("Loading set-a...")
    patients = get_dataset_a()

    with open(NORM_PARAMS_PATH) as f:
        norm_params = {
            k: ScalerParams.model_validate(v) for k, v in json.load(f).items()
        }

    print("Scaling patients...")
    scaled = scale_patients(patients, norm_params)

    print("Plotting...")
    plot_static_distributions("a_scaled", scaled, out_dir=PLOTS_DIR)
    plot_timeseries_distributions("a_scaled", scaled, out_dir=PLOTS_DIR)
