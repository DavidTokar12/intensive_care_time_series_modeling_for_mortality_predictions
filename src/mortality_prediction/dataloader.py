import csv
import json
import logging
import os

from collections import defaultdict
from datetime import timedelta
from enum import Enum
from enum import IntEnum
from enum import StrEnum

import numpy as np

from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from mortality_prediction.utils import DATA_DIR


logger = logging.getLogger(__name__)

STATIC_PARAMS = {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}


class ICUType(IntEnum):
    CORONARY_CARE = 1
    CARDIAC_SURGERY = 2
    MEDICAL = 3
    SURGICAL = 4


class Gender(Enum):
    MALE = "m"
    FEMALE = "f"


class MechVentStatus(IntEnum):
    OFF = 0
    ON = 1


class GCSScore(IntEnum):
    SCORE_3 = 3
    SCORE_4 = 4
    SCORE_5 = 5
    SCORE_6 = 6
    SCORE_7 = 7
    SCORE_8 = 8
    SCORE_9 = 9
    SCORE_10 = 10
    SCORE_11 = 11
    SCORE_12 = 12
    SCORE_13 = 13
    SCORE_14 = 14
    SCORE_15 = 15


class ParamType(StrEnum):
    ALBUMIN = "Albumin"
    ALP = "ALP"
    ALT = "ALT"
    AST = "AST"
    BILIRUBIN = "Bilirubin"
    BUN = "BUN"
    CHOLESTEROL = "Cholesterol"
    CREATININE = "Creatinine"
    DIAS_ABP = "DiasABP"
    FIO2 = "FiO2"
    GCS = "GCS"
    GLUCOSE = "Glucose"
    HCO3 = "HCO3"
    HCT = "HCT"
    HR = "HR"
    K = "K"
    LACTATE = "Lactate"
    MG = "Mg"
    MAP = "MAP"
    MECH_VENT = "MechVent"
    NA = "Na"
    NI_DIAS_ABP = "NIDiasABP"
    NI_MAP = "NIMAP"
    NI_SYS_ABP = "NISysABP"
    PACO2 = "PaCO2"
    PAO2 = "PaO2"
    PH = "pH"
    PLATELETS = "Platelets"
    RESP_RATE = "RespRate"
    SAO2 = "SaO2"
    SYS_ABP = "SysABP"
    TEMP = "Temp"
    TROPONIN_I = "TroponinI"
    TROPONIN_T = "TroponinT"
    URINE = "Urine"
    WBC = "WBC"


# (description, (min, max)) — None means no range validation
PARAM_META: dict[ParamType, tuple[str, tuple[float, float] | None]] = {
    ParamType.ALBUMIN: ("Albumin (g/dL)", None),
    ParamType.ALP: ("Alkaline phosphatase (IU/L)", None),
    ParamType.ALT: ("Alanine transaminase (IU/L)", None),
    ParamType.AST: ("Aspartate transaminase (IU/L)", None),
    ParamType.BILIRUBIN: ("Bilirubin (mg/dL)", None),
    ParamType.BUN: ("Blood urea nitrogen (mg/dL)", None),
    ParamType.CHOLESTEROL: ("Cholesterol (mg/dL)", None),
    ParamType.CREATININE: ("Serum creatinine (mg/dL)", None),
    ParamType.DIAS_ABP: ("Invasive diastolic arterial blood pressure (mmHg)", (0, 200)),
    ParamType.FIO2: ("Fractional inspired O2 (0-1)", (0.21, 1.0)),
    ParamType.GCS: ("Glasgow Coma Score (3-15)", (3, 15)),
    ParamType.GLUCOSE: ("Serum glucose (mg/dL)", None),
    ParamType.HCO3: ("Serum bicarbonate (mmol/L)", None),
    ParamType.HCT: ("Hematocrit (%)", None),
    ParamType.HR: ("Heart rate (bpm)", (0, 300)),
    ParamType.K: ("Serum potassium (mEq/L)", None),
    ParamType.LACTATE: ("Lactate (mmol/L)", None),
    ParamType.MG: ("Serum magnesium (mmol/L)", None),
    ParamType.MAP: ("Invasive mean arterial blood pressure (mmHg)", (0, 300)),
    ParamType.MECH_VENT: ("Mechanical ventilation (0: off, 1: on)", (0, 1)),
    ParamType.NA: ("Serum sodium (mEq/L)", None),
    ParamType.NI_DIAS_ABP: (
        "Non-invasive diastolic arterial blood pressure (mmHg)",
        (0, 200),
    ),
    ParamType.NI_MAP: ("Non-invasive mean arterial blood pressure (mmHg)", (0, 250)),
    ParamType.NI_SYS_ABP: (
        "Non-invasive systolic arterial blood pressure (mmHg)",
        (0, 300),
    ),
    ParamType.PACO2: ("Partial pressure of arterial CO2 (mmHg)", None),
    ParamType.PAO2: ("Partial pressure of arterial O2 (mmHg)", None),
    ParamType.PH: ("Arterial pH", (6.5, 8.0)),
    ParamType.PLATELETS: ("Platelets (cells/nL)", None),
    ParamType.RESP_RATE: ("Respiration rate (bpm)", None),
    ParamType.SAO2: ("O2 saturation in hemoglobin (%)", (0, 100)),
    ParamType.SYS_ABP: ("Invasive systolic arterial blood pressure (mmHg)", (0, 300)),
    ParamType.TEMP: ("Temperature (°C)", (10, 45)),
    ParamType.TROPONIN_I: ("Troponin-I (μg/L)", None),
    ParamType.TROPONIN_T: ("Troponin-T (μg/L)", None),
    ParamType.URINE: ("Urine output (mL)", None),
    ParamType.WBC: ("White blood cell count (cells/nL)", None),
}

# Parameters whose values are discrete categories rather than continuous measurements
CATEGORICAL_PARAMS: frozenset[ParamType] = frozenset(
    {ParamType.GCS, ParamType.MECH_VENT}
)

# --------------------------------------------------------------------------- #
# Normalization                                                                #
# --------------------------------------------------------------------------- #

_ROBUST_TS_PARAMS: frozenset[ParamType] = frozenset({
    # Vitals
    ParamType.HR, ParamType.TEMP, ParamType.RESP_RATE,
    # Invasive blood pressures
    ParamType.DIAS_ABP, ParamType.MAP, ParamType.SYS_ABP,
    # Non-invasive blood pressures
    ParamType.NI_DIAS_ABP, ParamType.NI_MAP, ParamType.NI_SYS_ABP,
    # Electrolytes
    ParamType.NA, ParamType.K, ParamType.MG, ParamType.HCO3,
    # Blood panel
    ParamType.ALBUMIN, ParamType.CHOLESTEROL, ParamType.HCT, ParamType.PLATELETS,
    # Blood gases & O2
    ParamType.PACO2, ParamType.PH, ParamType.FIO2, ParamType.SAO2,
})

_LOG_STANDARD_TS_PARAMS: frozenset[ParamType] = frozenset({
    # Liver & enzymes
    ParamType.ALP, ParamType.ALT, ParamType.AST, ParamType.BILIRUBIN,
    # Kidney & waste
    ParamType.BUN, ParamType.CREATININE, ParamType.URINE,
    # Metabolic & stress
    ParamType.GLUCOSE, ParamType.LACTATE,
    # Cardiac markers
    ParamType.TROPONIN_I, ParamType.TROPONIN_T,
    # Blood counts
    ParamType.WBC,
    # Blood gases
    ParamType.PAO2,
})

_ROBUST_STATIC_PARAMS: frozenset[str] = frozenset({"age", "height_cm", "weight_kg"})


class ScalerType(StrEnum):
    ROBUST = "robust"
    LOG_STANDARD = "log_standard"


class ScalerParams(BaseModel):
    scaler_type: ScalerType
    # RobustScaler params
    center: float | None = None
    scale: float | None = None
    # LogStandardScaler params
    mean: float | None = None
    std: float | None = None


class Measurement(BaseModel):
    param: ParamType
    value: float | MechVentStatus | GCSScore

    @model_validator(mode="after")
    def validate_range_and_categorize(self):
        description, bounds = PARAM_META[self.param]
        raw = float(self.value)
        if bounds is not None:
            low, high = bounds
            if not (low <= raw <= high):
                raise ValueError(
                    f"{self.param.value} value {raw} out of range "
                    f"[{low}, {high}] — {description}"
                )
        if self.param == ParamType.MECH_VENT:
            self.value = MechVentStatus(int(raw))
        elif self.param == ParamType.GCS:
            self.value = GCSScore(int(raw))
        return self

    @property
    def description(self) -> str:
        return PARAM_META[self.param][0]


class TimePoint(BaseModel):
    time: timedelta
    measurements: list[Measurement]


class PatientStatic(BaseModel):
    record_id: str
    age: int | None
    gender: Gender | None
    height_cm: float | None
    weight_kg: float | None
    icu_type: ICUType | None
    in_hospital_death: bool | None

    @field_validator("gender", mode="before")
    @classmethod
    def parse_gender(cls, v):
        if v is None:
            return None
        int_mapping = {0: Gender.FEMALE, 1: Gender.MALE}
        if v in int_mapping:
            return int_mapping[v]
        if isinstance(v, str):
            return Gender(v)  # handles "m"/"f" from JSON cache
        raise ValueError(f"Unexpected gender value: {v}")

    @field_validator("age", mode="before")
    @classmethod
    def parse_age(cls, v):
        if v is None:
            return None
        age = int(v)
        if not (0 < age < 100):
            raise ValueError(f"Age must be between 0 and 100, got {age}")
        return age

    @field_validator("height_cm", mode="before")
    @classmethod
    def parse_height(cls, v):
        if v is None:
            return None
        height = float(v)
        if not (120 < height < 250):
            logger.warning(f"Implausible height {height} cm — treating as missing")
            return None
        return height

    @field_validator("weight_kg", mode="before")
    @classmethod
    def parse_weight(cls, v):
        if v is None:
            return None
        weight = float(v)
        if not (25 < weight < 400):
            logger.warning(f"Implausible weight {weight} kg — treating as missing")
            return None
        return weight


class Patient(BaseModel):
    static: PatientStatic
    timeseries: list[TimePoint]

    @model_validator(mode="after")
    def drop_beyond_48h(self):
        cutoff = timedelta(hours=48)
        self.timeseries = [tp for tp in self.timeseries if tp.time <= cutoff]
        return self


def parse_time(t: str) -> timedelta:
    hours, minutes = t.split(":")
    return timedelta(hours=int(hours), minutes=int(minutes))


def parse_value(v: str) -> float | None:
    try:
        f = float(v)
        return None if f == -1 else f
    except (ValueError, TypeError):
        return None


def load_outcomes(filepath: str) -> dict[str, bool]:
    """Return {record_id: in_hospital_death} from an Outcomes-*.txt file."""
    outcomes: dict[str, bool] = {}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_id = str(int(row["RecordID"]))
            outcomes[record_id] = bool(int(row["In-hospital_death"]))
    return outcomes


def load_patient(filepath: str, death: bool | None = None) -> Patient:
    raw_static: dict = {}
    raw_timeseries: dict[timedelta, dict] = defaultdict(dict)

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time = row["Time"].strip()
            param = row["Parameter"].strip()
            value = parse_value(row["Value"].strip())

            if not param:
                logger.warning(
                    f"Empty parameter name at time {time}, value {value} — skipping in {filepath}"
                )
                continue

            if param in STATIC_PARAMS:
                raw_static[param] = value
            else:
                raw_timeseries[parse_time(time)][param] = value

    static = PatientStatic(
        record_id=str(int(raw_static["RecordID"])),
        age=raw_static.get("Age"),
        gender=raw_static.get("Gender"),
        height_cm=raw_static.get("Height"),
        weight_kg=raw_static.get("Weight"),
        icu_type=raw_static.get("ICUType"),
        in_hospital_death=death,
    )

    timepoints = []
    for time_td, params in sorted(raw_timeseries.items()):
        measurements = []
        for param, val in params.items():
            if val is None:
                logger.warning(
                    f"Dropping None value: {param} at {time_td} in {filepath}"
                )
                continue
            try:
                measurements.append(Measurement(param=ParamType(param), value=val))
            except ValueError as e:
                logger.warning(f"Skipping invalid measurement: {e}")
        if measurements:
            timepoints.append(TimePoint(time=time_td, measurements=measurements))

    return Patient(static=static, timeseries=timepoints)


def main(input_dir: str, outcomes: dict[str, bool] | None = None) -> dict[str, Patient]:
    patients = {}

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".txt"):
            continue

        patient_id = filename.removesuffix(".txt")
        filepath = os.path.join(input_dir, filename)
        death = outcomes.get(patient_id) if outcomes else None

        if outcomes and patient_id not in outcomes:
            logger.warning(f"No outcome label found for patient {patient_id}")

        patient = load_patient(filepath, death=death)
        patients[patient_id] = patient

    return patients


def _load_or_cache_dataset(set_name: str) -> dict[str, Patient]:
    safe_name = set_name.replace("-", "_")
    cache_file = os.path.join(DATA_DIR, safe_name + ".json")
    log_file = os.path.join(DATA_DIR, safe_name + ".log")
    raw_dir = os.path.join(DATA_DIR, set_name)
    outcomes_file = os.path.join(DATA_DIR, f"Outcomes-{set_name[-1]}.txt")

    if os.path.exists(cache_file):
        logger.info(f"Loading {set_name} from cache: {cache_file}")
        with open(cache_file) as f:
            data = json.load(f)
        return {
            pid: Patient.model_validate(patient_data)
            for pid, patient_data in data.items()
        }

    outcomes = load_outcomes(outcomes_file)

    # Capture all warnings produced during parsing into a log file on disk
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter("%(levelname)s — %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    try:
        logger.info(f"Parsing {set_name} from raw files: {raw_dir}")
        patients = main(raw_dir, outcomes=outcomes)
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()

    logger.info(f"Parse warnings written to: {log_file}")

    serialized = {
        pid: patient.model_dump(mode="json") for pid, patient in patients.items()
    }
    with open(cache_file, "w") as f:
        json.dump(serialized, f)

    return patients


def compute_normalization_params(patients: dict[str, Patient]) -> dict[str, ScalerParams]:
    """
    Fit normalization scalers on all observed values across the patient cohort.

    Returns a dict keyed by parameter name (ParamType string value or static
    field name) mapping to a ScalerParams instance that carries everything
    needed to reconstruct the scaler at inference time.

    - RobustScaler  : center (median) + scale (IQR)
    - LogStandard   : mean + std of log1p-transformed values
    """
    ts_values: dict[ParamType, list[float]] = {
        p: [] for p in _ROBUST_TS_PARAMS | _LOG_STANDARD_TS_PARAMS
    }
    static_values: dict[str, list[float]] = {k: [] for k in _ROBUST_STATIC_PARAMS}

    for patient in patients.values():
        s = patient.static
        if s.age is not None:
            static_values["age"].append(float(s.age))
        if s.height_cm is not None:
            static_values["height_cm"].append(s.height_cm)
        if s.weight_kg is not None:
            static_values["weight_kg"].append(s.weight_kg)

        for tp in patient.timeseries:
            for m in tp.measurements:
                if m.param in ts_values:
                    ts_values[m.param].append(
                        float(m.value.value if hasattr(m.value, "value") else m.value)
                    )

    result: dict[str, ScalerParams] = {}

    for param in _ROBUST_TS_PARAMS:
        vals = ts_values[param]
        if not vals:
            continue
        scaler = RobustScaler().fit(np.array(vals).reshape(-1, 1))
        result[param.value] = ScalerParams(
            scaler_type=ScalerType.ROBUST,
            center=float(scaler.center_[0]),
            scale=float(scaler.scale_[0]),
        )

    for param in _LOG_STANDARD_TS_PARAMS:
        vals = ts_values[param]
        if not vals:
            continue
        log_vals = np.log1p(np.array(vals)).reshape(-1, 1)
        scaler = StandardScaler().fit(log_vals)
        result[param.value] = ScalerParams(
            scaler_type=ScalerType.LOG_STANDARD,
            mean=float(scaler.mean_[0]),
            std=float(scaler.scale_[0]),
        )

    for field in _ROBUST_STATIC_PARAMS:
        vals = static_values[field]
        if not vals:
            continue
        scaler = RobustScaler().fit(np.array(vals).reshape(-1, 1))
        result[field] = ScalerParams(
            scaler_type=ScalerType.ROBUST,
            center=float(scaler.center_[0]),
            scale=float(scaler.scale_[0]),
        )

    return result


def get_dataset_a() -> dict[str, Patient]:
    return _load_or_cache_dataset("set-a")


def get_dataset_b() -> dict[str, Patient]:
    return _load_or_cache_dataset("set-b")


def get_dataset_c() -> dict[str, Patient]:
    return _load_or_cache_dataset("set-c")
