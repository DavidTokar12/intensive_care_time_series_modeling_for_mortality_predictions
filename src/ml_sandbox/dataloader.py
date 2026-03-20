import csv
import logging
import os
import sys

from collections import defaultdict
from datetime import timedelta
from enum import Enum
from enum import IntEnum

from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator


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


class ParamType(str, Enum):
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
    ParamType.DIAS_ABP: ("Invasive diastolic arterial blood pressure (mmHg)", None),
    ParamType.FIO2: ("Fractional inspired O2 (0-1)", None),
    ParamType.GCS: ("Glasgow Coma Score (3-15)", None),
    ParamType.GLUCOSE: ("Serum glucose (mg/dL)", None),
    ParamType.HCO3: ("Serum bicarbonate (mmol/L)", None),
    ParamType.HCT: ("Hematocrit (%)", None),
    ParamType.HR: ("Heart rate (bpm)", None),
    ParamType.K: ("Serum potassium (mEq/L)", None),
    ParamType.LACTATE: ("Lactate (mmol/L)", None),
    ParamType.MG: ("Serum magnesium (mmol/L)", None),
    ParamType.MAP: ("Invasive mean arterial blood pressure (mmHg)", None),
    ParamType.MECH_VENT: ("Mechanical ventilation (0: false, 1: true)", None),
    ParamType.NA: ("Serum sodium (mEq/L)", None),
    ParamType.NI_DIAS_ABP: (
        "Non-invasive diastolic arterial blood pressure (mmHg)",
        None,
    ),
    ParamType.NI_MAP: ("Non-invasive mean arterial blood pressure (mmHg)", None),
    ParamType.NI_SYS_ABP: (
        "Non-invasive systolic arterial blood pressure (mmHg)",
        None,
    ),
    ParamType.PACO2: ("Partial pressure of arterial CO2 (mmHg)", None),
    ParamType.PAO2: ("Partial pressure of arterial O2 (mmHg)", None),
    ParamType.PH: ("Arterial pH (0-14)", None),
    ParamType.PLATELETS: ("Platelets (cells/nL)", None),
    ParamType.RESP_RATE: ("Respiration rate (bpm)", None),
    ParamType.SAO2: ("O2 saturation in hemoglobin (%)", None),
    ParamType.SYS_ABP: ("Invasive systolic arterial blood pressure (mmHg)", None),
    ParamType.TEMP: ("Temperature (°C)", None),
    ParamType.TROPONIN_I: ("Troponin-I (μg/L)", None),
    ParamType.TROPONIN_T: ("Troponin-T (μg/L)", None),
    ParamType.URINE: ("Urine output (mL)", None),
    ParamType.WBC: ("White blood cell count (cells/nL)", None),
}


class Measurement(BaseModel):
    param: ParamType
    value: float

    @model_validator(mode="after")
    def validate_range(self):
        description, bounds = PARAM_META[self.param]
        if bounds is None:
            return self
        low, high = bounds
        if not (low <= self.value <= high):
            raise ValueError(
                f"{self.param.value} value {self.value} out of range "
                f"[{low}, {high}] — {description}"
            )
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

    @field_validator("gender", mode="before")
    @classmethod
    def parse_gender(cls, v):
        if v is None:
            return None
        mapping = {0: Gender.FEMALE, 1: Gender.MALE}
        if v in mapping:
            return mapping[v]
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


def parse_time(t: str) -> timedelta:
    hours, minutes = t.split(":")
    return timedelta(hours=int(hours), minutes=int(minutes))


def parse_value(v: str) -> float | None:
    try:
        f = float(v)
        return None if f == -1 else f
    except (ValueError, TypeError):
        return None


def load_patient(filepath: str) -> Patient:
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
    )

    timepoints = []
    for time_td, params in sorted(raw_timeseries.items()):
        measurements = []
        for param, val in params.items():
            if val is None:
                print(f"Dropping None: {param} at {time_td} in {filepath}")
                continue
            try:
                measurements.append(Measurement(param=ParamType(param), value=val))
            except ValueError as e:
                logger.warning(f"Skipping invalid measurement: {e}")
        if measurements:
            timepoints.append(TimePoint(time=time_td, measurements=measurements))

    return Patient(static=static, timeseries=timepoints)


def main(input_dir: str) -> dict[str, Patient]:
    patients = {}

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".txt"):
            continue

        patient_id = filename.removesuffix(".txt")
        filepath = os.path.join(input_dir, filename)

        patient = load_patient(filepath)
        patients[patient_id] = patient

    return patients


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dataloader.py <input_dir>")
        sys.exit(1)

    patients = main(sys.argv[1])

    first = next(iter(patients.values()))
    print("\n--- Sample static ---")
    print(first.static.model_dump())
    print("\n--- First timepoint ---")
    print(first.timeseries[0].model_dump())
