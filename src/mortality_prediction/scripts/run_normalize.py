from mortality_prediction.dataloader import get_dataset_a
from mortality_prediction.dataloader import get_dataset_b
from mortality_prediction.dataloader import get_dataset_c
from mortality_prediction.normalize_data import convert_to_table_format
from mortality_prediction.normalize_data import convert_to_vector_format


if __name__ == "__main__":
    for set_name, loader in [
        ("set-a", get_dataset_a),
        ("set-b", get_dataset_b),
        ("set-c", get_dataset_c),
    ]:
        patients = loader()
        convert_to_table_format(set_name, patients)
        convert_to_vector_format(set_name, patients)
