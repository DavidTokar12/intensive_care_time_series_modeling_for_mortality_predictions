import json
import sys

from mortality_prediction.dataloader import compute_normalization_params
from mortality_prediction.dataloader import main
from mortality_prediction.utils import DATA_DIR


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_dataloader.py <input_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]

    patients = main(input_dir)

    set_name = input_dir.rstrip("/\\").split("/")[-1].replace("-", "_")
    out_path = f"{DATA_DIR}/{set_name}.json"
    serialized = {pid: patient.model_dump(mode="json") for pid, patient in patients.items()}
    with open(out_path, "w") as f:
        json.dump(serialized, f)
    print(f"Saved {len(patients)} patients -> {out_path}")

    if set_name == "set_a":
        norm_params = compute_normalization_params(patients)
        norm_path = f"{DATA_DIR}/set_a_normalization_params.json"
        with open(norm_path, "w") as f:
            json.dump({k: v.model_dump() for k, v in norm_params.items()}, f, indent=2)
        print(f"Saved {len(norm_params)} scaler params -> {norm_path}")

    first = next(iter(patients.values()))
    print("\n--- Sample static ---")
    print(first.static.model_dump())
    print("\n--- First timepoint ---")
    print(first.timeseries[0].model_dump())
