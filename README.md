# Mortality Prediction (Physionet 2012)

## Setup
1. **Dependencies**: Uses **uv**. Run `uv sync`.
2. **Data**: 
   * Download `set-a`, `set-b`, `set-c`, `Outcomes-a.txt`, `Outcomes-b.txt`, and `Outcomes-c.txt` from [PhysioNet](https://physionet.org/content/challenge-2012/1.0.0/).
   * Place these files in the `data/` folder before running scripts.
3. **Preprocessing**:
   * Run `src/mortality_prediction/scripts/prepare_data.py`.
4. **Paths**: Update data paths inside the notebooks to match your specific VM environment.

---

## Report Mapping

| Report Section | Notebook / Implementation |
| :--- | :--- |
| **1. Data Processing** | `dataloader.py`, `normalize_data.py` |
| **2.1 Classic ML** | `classic_ml.ipynb` |
| **2.2 RNNs** | `lstm.ipynb` |
| **2.3a Transformers** |  `lstm.ipynb` | `triplet_transformer.ipynb` |
| **2.3b Tokenized Transformers**| `triplet_transformer.ipynb` |
| **3. Representation Learning** | `contrastive_pretraining.ipynb` |
| **4.1 - 4.2 LLMs** | `foundation_models_llms.ipynb` |
| **4.3 Time-Series Models** | `foundation_models_time_series.ipynb` |