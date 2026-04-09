import logging
import math
import os

from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from mortality_prediction.dataloader import PARAM_META
from mortality_prediction.dataloader import GCSScore
from mortality_prediction.dataloader import Gender
from mortality_prediction.dataloader import ICUType
from mortality_prediction.dataloader import MechVentStatus
from mortality_prediction.dataloader import ParamType
from mortality_prediction.dataloader import Patient
from mortality_prediction.dataloader import get_dataset_a
from mortality_prediction.dataloader import get_dataset_b
from mortality_prediction.dataloader import get_dataset_c


logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "plots")
)

_NUMERIC_PARAMS: list[tuple[str, str]] = [
    ("age", "Age (years)"),
    ("height_cm", "Height (cm)"),
    ("weight_kg", "Weight (kg)"),
]

_ICU_LABELS: dict[ICUType, str] = {
    ICUType.CORONARY_CARE: "Coronary Care",
    ICUType.CARDIAC_SURGERY: "Cardiac Surgery",
    ICUType.MEDICAL: "Medical",
    ICUType.SURGICAL: "Surgical",
}

_GENDER_LABELS: dict[Gender, str] = {
    Gender.MALE: "Male",
    Gender.FEMALE: "Female",
}


def _collect_static(patients: dict[str, Patient]) -> dict:
    """Collect static field values across all patients."""
    result: dict = {field: [] for field, _ in _NUMERIC_PARAMS}
    result["gender"] = []
    result["icu_type"] = []

    for patient in patients.values():
        s = patient.static
        for field, _ in _NUMERIC_PARAMS:
            val = getattr(s, field)
            if val is not None:
                result[field].append(val)
        if s.gender is not None:
            result["gender"].append(s.gender)
        if s.icu_type is not None:
            result["icu_type"].append(s.icu_type)

    return result


def _gender_text(gender_vals: list[Gender]) -> str:
    counts = Counter(gender_vals)
    total = len(gender_vals)
    lines = ["Gender"]
    for g in Gender:
        n = counts.get(g, 0)
        pct = 100 * n / total if total else 0
        lines.append(f"  {_GENDER_LABELS[g]}: {n}  ({pct:.1f}%)")
    lines.append(f"  Missing: {0}")  # already filtered Nones before collection
    return "\n".join(lines)


def _icu_text(icu_vals: list[ICUType]) -> str:
    counts = Counter(icu_vals)
    total = len(icu_vals)
    lines = ["ICU Type"]
    for icu in ICUType:
        n = counts.get(icu, 0)
        pct = 100 * n / total if total else 0
        lines.append(f"  {icu.value} — {_ICU_LABELS[icu]}: {n}  ({pct:.1f}%)")
    return "\n".join(lines)


def plot_static_distributions(
    set_name: str,
    patients: dict[str, Patient],
    out_dir: str = PLOTS_DIR,
) -> str:
    """
    Plot bell-curve distributions for each numeric static param and
    render gender / ICU-type counts as text.  Returns the saved file path.
    """
    os.makedirs(out_dir, exist_ok=True)

    data = _collect_static(patients)

    n_numeric = len(_NUMERIC_PARAMS)

    fig = plt.figure(figsize=(6 * n_numeric, 9))
    fig.suptitle(f"Static parameter distributions — {set_name}", fontsize=14, y=0.98)

    gs = fig.add_gridspec(
        2,
        n_numeric,
        height_ratios=[3, 1],
        hspace=0.45,
        wspace=0.35,
    )

    # --- distribution subplots ---
    for col, (field, label) in enumerate(_NUMERIC_PARAMS):
        ax = fig.add_subplot(gs[0, col])
        vals = data[field]
        if vals:
            sns.histplot(
                vals,
                kde=True,
                ax=ax,
                color="steelblue",
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        n_present = len(vals)
        n_missing = len(patients) - n_present
        ax.text(
            0.97,
            0.97,
            f"n={n_present}\nmissing={n_missing}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="dimgray",
        )

    # --- text panel ---
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    gender_block = _gender_text(data["gender"])
    icu_block = _icu_text(data["icu_type"])

    ax_text.text(
        0.02,
        0.95,
        gender_block,
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontfamily="monospace",
    )
    ax_text.text(
        0.35,
        0.95,
        icu_block,
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontfamily="monospace",
    )

    safe_name = set_name.replace("-", "_")
    out_path = os.path.join(out_dir, f"{safe_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def _draw_param_distribution(ax: plt.Axes, param: ParamType, vals: list[float]) -> None:
    """Fill a single axes with the value distribution for one ParamType."""
    description, _ = PARAM_META[param]

    if param == ParamType.MECH_VENT:
        counts = Counter(int(v) for v in vals)
        total = sum(counts.values())
        labels = [f"{s.name.title()}\n({s.value})" for s in MechVentStatus]
        heights = [counts.get(s.value, 0) for s in MechVentStatus]
        ax.bar(labels, heights, color="steelblue", edgecolor="white")
        if total:
            stats_line = "   ".join(
                f"{s.name.title()}: {counts.get(s.value, 0)} ({100 * counts.get(s.value, 0) / total:.1f}%)"
                for s in MechVentStatus
            )
        else:
            stats_line = "no data"

    elif param == ParamType.GCS:
        counts = Counter(int(v) for v in vals)
        scores = [s.value for s in GCSScore]
        ax.bar(
            scores,
            [counts.get(s, 0) for s in scores],
            color="steelblue",
            edgecolor="white",
        )
        ax.set_xticks(scores)
        total = sum(counts.values())
        if total:
            mode_score = max(counts, key=lambda k: counts[k])
            stats_line = f"mode: {mode_score}   n={total}"
        else:
            stats_line = "no data"

    else:
        if vals:
            mean_v = sum(vals) / len(vals)
            min_v = min(vals)
            max_v = max(vals)
            sns.histplot(
                vals,
                kde=True,
                ax=ax,
                color="steelblue",
                edgecolor="white",
                linewidth=0.3,
            )
            stats_line = f"mean {mean_v:.2f}   min {min_v:.2f}   max {max_v:.2f}"
        else:
            stats_line = "no data"

    ax.set_title(f"{description}\n{stats_line}", fontsize=6.5, pad=4)
    ax.set_xlabel("Value", fontsize=6.5)
    ax.set_ylabel("Count", fontsize=6.5)
    ax.tick_params(labelsize=6)


def plot_timeseries_distributions(
    set_name: str,
    patients: dict[str, Patient],
    out_dir: str = PLOTS_DIR,
) -> str:
    """
    For every ParamType, collect all values across all patients and:
      - Save one combined overview grid.
      - Save a standalone file per param named after its value
        (e.g. HR.png) inside a per-set subdirectory.

    Returns the path to the combined overview file.
    """
    os.makedirs(out_dir, exist_ok=True)

    param_vals: dict[ParamType, list[float]] = defaultdict(list)
    for patient in patients.values():
        for tp in patient.timeseries:
            for m in tp.measurements:
                param_vals[m.param].append(m.value)

    params = sorted(ParamType, key=lambda p: p.value)
    n_cols = 6
    n_rows = math.ceil(len(params) / n_cols)

    # --- combined overview grid ---
    fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
    fig.suptitle(f"Timeseries value distributions — {set_name}", fontsize=14, y=1.001)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.75, wspace=0.4)

    for idx, param in enumerate(params):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        _draw_param_distribution(ax, param, param_vals.get(param, []))

    for idx in range(len(params), n_rows * n_cols):
        fig.add_subplot(gs[idx // n_cols, idx % n_cols]).axis("off")

    safe_name = set_name.replace("-", "_")
    combined_path = os.path.join(out_dir, f"{safe_name}_ts_distributions.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {combined_path}")

    # --- individual files ---
    individual_dir = os.path.join(out_dir, f"{safe_name}_ts_distributions")
    os.makedirs(individual_dir, exist_ok=True)

    for param in params:
        fig, ax = plt.subplots(figsize=(6, 4))
        _draw_param_distribution(ax, param, param_vals.get(param, []))
        fig.tight_layout()
        out_path = os.path.join(individual_dir, f"{param.value}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Saved {len(params)} individual files to: {individual_dir}")
    return combined_path


def plot_patient(
    patient: Patient,
    patient_id: str = "",
    out_path: str | None = None,
) -> str:
    """
    Plot all timeseries measurements for a single patient.

    Layout:
      - Top bar: static demographics as text.
      - Grid of subplots (one per unique parameter), x = time in hours,
        y = measured value, each point annotated with its value.

    Returns the saved file path.
    """
    # --- collect (hours, value) per param ---
    param_series: dict[ParamType, list[tuple[float, float]]] = defaultdict(list)
    for tp in patient.timeseries:
        hours = tp.time.total_seconds() / 3600
        for m in tp.measurements:
            param_series[m.param].append((hours, m.value))

    for pts in param_series.values():
        pts.sort(key=lambda x: x[0])

    params_sorted = sorted(param_series.keys(), key=lambda p: p.value)
    n_params = len(params_sorted)

    if n_params == 0:
        raise ValueError(
            f"Patient {patient_id or patient.static.record_id} has no timeseries data."
        )

    n_cols = 5
    n_ts_rows = math.ceil(n_params / n_cols)

    # Row 0 = static text (short), rows 1..n_ts_rows = timeseries subplots
    fig = plt.figure(figsize=(5 * n_cols, 1.8 + 3.5 * n_ts_rows))
    label = patient_id or patient.static.record_id
    fig.suptitle(f"Patient {label}", fontsize=13, y=0.995)

    gs = fig.add_gridspec(
        1 + n_ts_rows,
        n_cols,
        height_ratios=[0.35] + [1.0] * n_ts_rows,
        hspace=0.7,
        wspace=0.4,
    )

    # --- static text bar ---
    ax_static = fig.add_subplot(gs[0, :])
    ax_static.axis("off")
    s = patient.static
    gender_str = _GENDER_LABELS.get(s.gender, "Unknown") if s.gender else "Unknown"
    icu_str = _ICU_LABELS.get(s.icu_type, "Unknown") if s.icu_type else "Unknown"
    static_text = (
        f"Record ID: {s.record_id}   |   Age: {s.age or '?'}   |   "
        f"Gender: {gender_str}   |   "
        f"Height: {f'{s.height_cm:.0f} cm' if s.height_cm else '?'}   |   "
        f"Weight: {f'{s.weight_kg:.1f} kg' if s.weight_kg else '?'}   |   "
        f"ICU Type: {icu_str}"
    )
    ax_static.text(
        0.5,
        0.5,
        static_text,
        transform=ax_static.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontfamily="monospace",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#f0f4f8",
            "edgecolor": "#b0bec5",
        },
    )

    # --- timeseries subplots ---
    for idx, param in enumerate(params_sorted):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        times, values = zip(*param_series[param], strict=True)
        description, _ = PARAM_META[param]

        if param == ParamType.MECH_VENT:
            # step plot: binary 0/1 over time
            ax.step(times, values, where="post", color="steelblue", linewidth=1)
            ax.scatter(times, values, color="steelblue", s=18, zorder=3)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(
                [f"{s.name.title()} ({s.value})" for s in MechVentStatus], fontsize=6
            )

        elif param == ParamType.GCS:
            # scatter + line; lock y-axis to the GCS range with integer ticks
            ax.plot(
                times,
                values,
                "o-",
                color="steelblue",
                markersize=4,
                linewidth=1,
                zorder=2,
            )
            ax.set_ylim(2.5, 15.5)
            ax.set_yticks([s.value for s in GCSScore])
            for t, v in zip(times, values, strict=True):
                ax.annotate(
                    str(int(v)),
                    xy=(t, v),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5.5,
                    color="#444444",
                )

        else:
            ax.plot(
                times,
                values,
                "o-",
                color="steelblue",
                markersize=4,
                linewidth=1,
                zorder=2,
            )
            for t, v in zip(times, values, strict=True):
                ax.annotate(
                    f"{v:.1f}",
                    xy=(t, v),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5.5,
                    color="#444444",
                )

        ax.set_title(description, fontsize=6.5, pad=3)
        ax.set_xlabel("Time (h)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    # hide any leftover empty cells
    for idx in range(n_params, n_ts_rows * n_cols):
        row = 1 + idx // n_cols
        col = idx % n_cols
        fig.add_subplot(gs[row, col]).axis("off")

    if out_path is None:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        out_path = os.path.join(PLOTS_DIR, f"patient_{label}.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def _summarize(name: str, patients: dict) -> None:
    sample = next(iter(patients.values()))
    logger.info(
        f"{name}: {len(patients)} patients | "
        f"sample record_id={sample.static.record_id}, "
        f"timepoints={len(sample.timeseries)}"
    )


if __name__ == "__main__":
    datasets = {
        "set-a": get_dataset_a,
        "set-b": get_dataset_b,
        "set-c": get_dataset_c,
    }

    for name, loader in datasets.items():
        logger.info(f"Loading {name}...")
        patients = loader()
        _summarize(name, patients)
        plot_static_distributions(name, patients)
        plot_timeseries_distributions(name, patients)

    # test plot_patient on the first patient from set-a
    logger.info("Plotting sample patient from set-a...")
    sample_patients = get_dataset_a()
    pid, sample = next(iter(sample_patients.items()))
    plot_patient(sample, patient_id=pid)
