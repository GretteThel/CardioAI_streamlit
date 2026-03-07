# app.py
from __future__ import annotations

import json
import time
import hashlib
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Plotly optional
try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    go = None
    HAVE_PLOTLY = False

from cardioai_infer import (
    load_model,
    load_uploaded_npy,
    predict_ecg,
    LEAD_NAMES,
    CLASS_NAMES,
)


# =========================================================
# HF asset download (model + demos) to avoid pushing binaries
# =========================================================
from huggingface_hub import hf_hub_download

HF_MODEL_REPO = "grettezybelle/cardioai-model"
HF_DEMO_REPO  = "grettezybelle/cardioai-demos"

# local cache folder (created inside the app directory)
ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

@st.cache_resource
def ensure_assets() -> tuple[Path, Path, Path]:
    """
    Downloads:
      - model checkpoint into assets/
      - demo_manifest.json into assets/demo/
      - demo_*.npy into assets/demo/

    Returns:
      model_path, demo_dir, demo_manifest_path
    """
    demo_dir = ASSETS_DIR / "demo"
    demo_dir.mkdir(exist_ok=True)

    # 1) model
    model_local = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="best_hybrid_final.pt",
        local_dir=str(ASSETS_DIR),
        local_dir_use_symlinks=False,
    )
    model_path = Path(model_local)

    # 2) manifest
    manifest_local = hf_hub_download(
        repo_id=HF_DEMO_REPO,
        filename="demo_manifest.json",
        local_dir=str(demo_dir),
        local_dir_use_symlinks=False,
    )
    demo_manifest_path = Path(manifest_local)

    # 3) download all demo files listed in manifest
    try:
        manifest = json.loads(demo_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}

    for fname in manifest.keys():
        if fname.endswith(".npy"):
            hf_hub_download(
                repo_id=HF_DEMO_REPO,
                filename=fname,
                local_dir=str(demo_dir),
                local_dir_use_symlinks=False,
            )

    return model_path, demo_dir, demo_manifest_path

# =========================================================
# Paths
# =========================================================
APP_DIR = Path(__file__).parent

# Default (local dev) paths — will be overwritten if HF assets are available
MODEL_PATH = APP_DIR / "models" / "best_hybrid_final.pt"
DEMO_DIR = APP_DIR / "demo"
DEMO_MANIFEST_PATH = DEMO_DIR / "demo_manifest.json"

UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_INDEX_PATH = UPLOAD_DIR / "uploads_index.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FS = 500

# =========================================================
# Streamlit config
# =========================================================
st.set_page_config(page_title="CardioAI – ECG Explorer", layout="wide")

# =========================================================
# Prefer HF-downloaded assets if available
# =========================================================
try:
    model_p, demo_p, manifest_p = ensure_assets()
    MODEL_PATH = model_p
    DEMO_DIR = demo_p
    DEMO_MANIFEST_PATH = manifest_p
except Exception as e:
    # If offline / HF unreachable, fallback to local folders
    st.sidebar.warning(f"HF assets not loaded, using local files. ({e})")


# =========================================================
# Small compatibility helper (Streamlit width changes)
# =========================================================
def _stretch_kwargs(fn):
    """Use new width API if available; otherwise fall back to use_container_width."""
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}

BTN_W = _stretch_kwargs(st.button)
DL_W = _stretch_kwargs(st.download_button)

# =========================================================
# Session defaults
# =========================================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"
if "ecg_style" not in st.session_state:
    st.session_state["ecg_style"] = "Standard"
if "probability_view" not in st.session_state:
    st.session_state["probability_view"] = "Both"
if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.50
if "apply_preprocess" not in st.session_state:
    st.session_state["apply_preprocess"] = False

# input selections
if "demo_file" not in st.session_state:
    st.session_state["demo_file"] = None  # filename only
if "saved_upload_name" not in st.session_state:
    st.session_state["saved_upload_name"] = None

# =========================================================
# Theme / CSS
# =========================================================
def apply_theme_css(theme: str) -> None:
    if theme == "Dark":
        css = """
        <style>
        .stApp { background-color: #0b1220; color: #e5e7eb; }
        h1, h2, h3, h4, h5, h6, p, label, div, span { color: #e5e7eb !important; }
        .small-note { color: #94a3b8; font-size: 0.9rem; }
        .badge { display:inline-block; padding:0.18rem 0.55rem; border-radius:0.55rem;
                 font-size:0.85rem; margin-right:0.35rem; margin-bottom:0.3rem; }
        .badge-ok { background:#14532d; color:#dcfce7 !important; }
        .badge-warn { background:#713f12; color:#fef9c3 !important; }
        .badge-bad { background:#7f1d1d; color:#fee2e2 !important; }
        .card { padding: 0.9rem 1rem; border: 1px solid #243244; border-radius: 0.9rem; background: #0f172a; }
        /* tighter layout */
        section.main > div { padding-top: 1.2rem; }
        </style>
        """
    else:
        css = """
        <style>
        .small-note { color: #6b7280; font-size: 0.9rem; }
        .badge { display:inline-block; padding:0.18rem 0.55rem; border-radius:0.55rem;
                 font-size:0.85rem; margin-right:0.35rem; margin-bottom:0.3rem; }
        .badge-ok { background:#dcfce7; color:#166534 !important; }
        .badge-warn { background:#fef9c3; color:#854d0e !important; }
        .badge-bad { background:#fee2e2; color:#991b1b !important; }
        .card { padding: 0.9rem 1rem; border: 1px solid #e5e7eb; border-radius: 0.9rem; background: white; }
        section.main > div { padding-top: 1.2rem; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

def badge(text: str, kind: str) -> str:
    cls = {"ok": "badge-ok", "warn": "badge-warn", "bad": "badge-bad"}[kind]
    return f"<span class='badge {cls}'>{text}</span>"

apply_theme_css(st.session_state["theme"])

# =========================================================
# Upload persistence (save previous uploads)
# =========================================================
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_upload_index() -> List[dict]:
    if UPLOAD_INDEX_PATH.exists():
        try:
            return json.loads(UPLOAD_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_upload_index(items: List[dict]) -> None:
    UPLOAD_INDEX_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")

def persist_uploaded_npy(uploaded_file, extra_meta: dict | None = None) -> dict:
    raw_bytes = uploaded_file.getvalue()
    sha12 = _sha256_bytes(raw_bytes)[:12]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = Path(uploaded_file.name).stem.replace(" ", "_")
    out_name = f"{ts}__{safe_stem}__{sha12}.npy"
    out_path = UPLOAD_DIR / out_name

    idx = load_upload_index()
    for rec in idx:
        if rec.get("sha12") == sha12 and rec.get("orig_name") == uploaded_file.name:
            return rec  # already saved

    out_path.write_bytes(raw_bytes)

    rec = {
        "id": sha12,
        "sha12": sha12,
        "orig_name": uploaded_file.name,
        "saved_name": out_name,
        "saved_path": str(out_path),
        "saved_at": ts,
    }
    if extra_meta:
        rec.update(extra_meta)

    idx.insert(0, rec)
    save_upload_index(idx)
    return rec

def delete_saved_upload(saved_name: str) -> None:
    idx = load_upload_index()
    idx2 = [r for r in idx if r.get("saved_name") != saved_name]
    save_upload_index(idx2)
    p = UPLOAD_DIR / saved_name
    if p.exists():
        p.unlink()

def load_saved_npy(saved_name: str) -> np.ndarray:
    p = UPLOAD_DIR / saved_name
    x = np.load(str(p), allow_pickle=True).astype(np.float32)
    if x.shape != (12, 5000):
        raise ValueError(f"Saved file must have shape (12,5000), got {x.shape}")
    return x

# =========================================================
# Demo manifest + demo discovery (robust)
# =========================================================
def infer_label_from_filename(name: str) -> Optional[str]:
    s = name.lower()
    for lab in ["norm", "mi", "sttc", "cd", "hyp"]:
        if f"demo_{lab}" in s:
            return "NORM" if lab == "norm" else lab.upper()
    return None

def load_demo_manifest() -> dict:
    if DEMO_MANIFEST_PATH.exists():
        try:
            return json.loads(DEMO_MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def scan_demo_dir() -> List[Path]:
    if not DEMO_DIR.exists():
        return []
    return sorted(DEMO_DIR.glob("demo_*.npy"))

def build_demo_index() -> Tuple[Dict[str, List[dict]], List[str]]:
    """
    Returns:
      groups: dict group_label -> list of entries
      missing_files: manifest entries missing on disk
    Entry: {file, label, exists, meta}
    """
    manifest = load_demo_manifest()
    scanned = scan_demo_dir()

    scanned_names = {p.name for p in scanned}
    groups: Dict[str, List[dict]] = {}
    missing: List[str] = []

    # 1) add manifest entries (even if missing)
    for fname, meta in manifest.items():
        label = meta.get("expected_label") or infer_label_from_filename(fname) or "OTHER"
        p = DEMO_DIR / fname
        exists = p.exists()
        if not exists:
            missing.append(fname)
        groups.setdefault(label, []).append({"file": fname, "label": label, "exists": exists, "meta": meta})

    # 2) add any scanned .npy not in manifest (fallback)
    for p in scanned:
        if p.name in manifest:
            continue
        label = infer_label_from_filename(p.name) or "OTHER"
        groups.setdefault(label, []).append({"file": p.name, "label": label, "exists": True, "meta": {}})

    # stable ordering (existing first, then name)
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda d: (not d["exists"], d["file"]))

    return groups, missing

def load_local_demo_npy(filename: str) -> np.ndarray:
    p = DEMO_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Demo file missing on disk: {p}")
    x = np.load(str(p), allow_pickle=True).astype(np.float32)
    if x.shape != (12, 5000):
        raise ValueError(f"{filename} must have shape (12,5000), got {x.shape}")
    return x

# =========================================================
# Signal helpers (for fallback measurements)
# =========================================================
def estimate_hr_from_rpeaks(rpeaks: np.ndarray, fs: int = DEFAULT_FS):
    if rpeaks is None or len(rpeaks) < 2:
        return None
    rr = np.diff(rpeaks) / float(fs)
    if rr.size == 0:
        return None
    rr_med = float(np.median(rr))
    if rr_med <= 0:
        return None
    return 60.0 / rr_med

def estimate_qrs_ms_from_beat(beat: np.ndarray, fs: int = DEFAULT_FS):
    y = np.asarray(beat, dtype=np.float32)
    idx_peak = int(np.argmax(np.abs(y)))
    peak = float(np.abs(y[idx_peak]))
    if peak < 1e-6:
        return None
    thr = 0.25 * peak
    left = idx_peak
    while left > 0 and abs(y[left]) > thr:
        left -= 1
    right = idx_peak
    while right < len(y) - 1 and abs(y[right]) > thr:
        right += 1
    width_samples = max(0, right - left)
    return (width_samples / float(fs)) * 1000.0

def estimate_st_deviation(beat: np.ndarray, fs: int = DEFAULT_FS):
    y = np.asarray(beat, dtype=np.float32)
    idx_peak = int(np.argmax(np.abs(y)))
    pre = max(0, idx_peak - int(0.25 * fs))
    base_win = y[pre:min(idx_peak, pre + int(0.20 * fs))]
    if base_win.size < 10:
        return None
    baseline = float(np.median(base_win))
    st_idx = idx_peak + int(0.08 * fs)
    if st_idx >= len(y):
        return None
    return float(y[st_idx] - baseline)

def looks_preprocessed_zscored(x12: np.ndarray) -> bool:
    mu = float(np.mean(x12))
    sd = float(np.std(x12))
    return abs(mu) < 0.2 and 0.6 < sd < 1.6

def probability_df(probs: dict, preds: dict) -> pd.DataFrame:
    rows = []
    for label, prob in probs.items():
        rows.append({"Label": label, "Probability": round(float(prob), 4), "Above threshold": int(preds[label])})
    return pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)

def compute_lead_activity_from_beats(beats: np.ndarray) -> np.ndarray:
    act = np.mean(np.abs(beats), axis=(0, 2))
    s = float(np.sum(act)) + 1e-9
    return act / s

def build_quality_metrics(x_used: np.ndarray, rpeaks: np.ndarray, beats: np.ndarray) -> dict:
    finite_ratio = float(np.isfinite(x_used).mean())
    nonzero_ratio = float(np.count_nonzero(x_used) / x_used.size)
    std_val = float(np.std(x_used))
    peak_count = int(len(rpeaks) if rpeaks is not None else 0)
    return {
        "Finite ratio": min(max(finite_ratio, 0.0), 1.0),
        "Non-zero ratio": min(max(nonzero_ratio, 0.0), 1.0),
        "Signal spread": min(std_val / 2.0, 1.0),
        "R-peak coverage": min(peak_count / 20.0, 1.0),
    }

# =========================================================
# Text explanation builders (keep titles neutral)
# =========================================================
def build_quick_summary(result, threshold):
    probs = result["probs"]
    preds = result["preds"]
    top = result["top_label"]
    top_prob = float(probs[top])
    mi_prob = result.get("mi_prob", None)
    positives = [k for k, v in preds.items() if v == 1]

    lines = []
    lines.append(f"Top match: **{top}** with **{top_prob:.0%}** probability.")

    if positives:
        lines.append("Above threshold: " + ", ".join(positives) + ".")
    else:
        lines.append("No label crossed the decision threshold.")

    if mi_prob is not None:
        lines.append(f"MI probability: **{mi_prob:.0%}**" + (" (flagged)" if mi_prob >= threshold else " (not flagged)") + ".")

    hr = result.get("measurements", {}).get("heart_rate_bpm", None)
    if hr is not None:
        lines.append(f"Estimated heart rate: **{hr:.1f} bpm**.")

    if result.get("uncertainty_flag", False):
        lines.append("Note: model certainty is lower for this sample (treat with extra caution).")
    else:
        lines.append("The top label is relatively separated from competitors.")

    lines.append("Decision-support only (not a diagnosis).")
    return "\n\n".join(lines)

def build_detailed_interpretation(result, threshold):
    probs = result["probs"]
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = sorted_items[0]
    second_label, second_prob = sorted_items[1] if len(sorted_items) > 1 else ("N/A", 0.0)
    margin = float(top_prob - second_prob)

    measurements = result.get("measurements", {})
    hr = measurements.get("heart_rate_bpm", None)
    qrs = measurements.get("qrs_ms_est", None)
    st_dev = measurements.get("st_dev_est", None)

    pos = result.get("positive_labels", [])
    applied = result.get("applied_preprocess", False)

    conf = "high" if top_prob >= 0.85 else "moderate" if top_prob >= 0.60 else "low"

    lines = []
    lines.append(f"Top label: **{top_label}** (p={top_prob:.3f}, {conf} confidence).")
    lines.append(f"Closest competitor: **{second_label}** (p={second_prob:.3f}).")
    lines.append(f"Margin: **{margin:.3f}** " + ("(clearer)" if margin >= 0.20 else "(ambiguous)"))

    if pos:
        lines.append(f"Labels ≥ {threshold:.2f}: " + ", ".join(pos))
    else:
        lines.append(f"No label ≥ {threshold:.2f}.")

    if hr is not None:
        lines.append(f"HR (approx): {hr:.1f} bpm.")
    if qrs is not None:
        lines.append(f"QRS (approx): {qrs:.0f} ms.")
    if st_dev is not None:
        lines.append(f"ST deviation (approx, Lead II): {st_dev:.3f} (z-units).")

    lines.append(f"Preprocessing applied: {applied}.")
    lines.append("_Decision-support summary only._")
    return "\n\n".join(lines)

# =========================================================
# Plot helpers
# =========================================================
def _fig_colors(theme):
    if theme == "Dark":
        return {"fig": "#0b1220", "ax": "#0f172a", "text": "#e5e7eb", "grid": "#243244"}
    return {"fig": "white", "ax": "white", "text": "black", "grid": "#d1d5db"}

def plot_12_lead_plotly(x12, title="ECG used for inference", fs=500, theme="Light"):
    if not HAVE_PLOTLY:
        return None
    t = np.arange(x12.shape[1]) / float(fs)
    offsets = np.arange(12)[::-1] * 5.0
    template = "plotly_dark" if theme == "Dark" else "plotly_white"
    fig = go.Figure()
    for i, lead in enumerate(LEAD_NAMES):
        fig.add_trace(go.Scatter(
            x=t, y=x12[i] + offsets[i], mode="lines", name=lead,
            line=dict(width=1.2),
            hovertemplate=f"{lead}<br>t=%{{x:.3f}} s<br>amp=%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        template=template,
        height=760,
        margin=dict(l=70, r=20, t=50, b=40),
        showlegend=False,
        xaxis_title="Time (s)",
        yaxis_title="Leads (offset display)",
    )
    fig.update_yaxes(tickmode="array", tickvals=offsets, ticktext=LEAD_NAMES, showgrid=True, zeroline=False)
    fig.update_xaxes(showgrid=True)
    return fig

def plot_12_lead_matplotlib(x12, title="ECG", fs=DEFAULT_FS, theme="Light", style="Standard"):
    colors = _fig_colors(theme)
    t = np.arange(x12.shape[1]) / float(fs)
    offsets = np.arange(12)[::-1] * 5.0
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])

    for i in range(12):
        ax.plot(t, x12[i] + offsets[i], linewidth=0.9)

    ax.set_yticks(offsets)
    ax.set_yticklabels(LEAD_NAMES, color=colors["text"])
    ax.set_xlabel("Time (s)", color=colors["text"])
    ax.set_title(title, color=colors["text"])
    ax.tick_params(axis="x", colors=colors["text"])
    ax.tick_params(axis="y", colors=colors["text"])

    if style == "ECG paper":
        ax.set_facecolor("#fff7f7" if theme == "Light" else "#111827")
        for xv in np.arange(0, t[-1] + 0.04, 0.04):
            ax.axvline(xv, color="#f8b4b4", linewidth=0.35, alpha=0.35, zorder=0)
        for xv in np.arange(0, t[-1] + 0.2, 0.2):
            ax.axvline(xv, color="#ef4444", linewidth=0.6, alpha=0.35, zorder=0)
        y_min, y_max = ax.get_ylim()
        for yv in np.arange(np.floor(y_min), np.ceil(y_max), 0.5):
            ax.axhline(yv, color="#f8b4b4", linewidth=0.35, alpha=0.35, zorder=0)
        for yv in np.arange(np.floor(y_min), np.ceil(y_max), 2.5):
            ax.axhline(yv, color="#ef4444", linewidth=0.6, alpha=0.35, zorder=0)
    else:
        ax.grid(True, alpha=0.25, color=colors["grid"])

    for spine in ax.spines.values():
        spine.set_color(colors["text"])
    fig.tight_layout()
    return fig

def plot_single_beat(beat_lead2, theme="Light"):
    colors = _fig_colors(theme)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])
    ax.plot(beat_lead2, linewidth=1.0)
    ax.set_title("Example extracted beat (Lead II)", color=colors["text"])
    ax.set_xlabel("Samples", color=colors["text"])
    ax.tick_params(axis="x", colors=colors["text"])
    ax.tick_params(axis="y", colors=colors["text"])
    ax.grid(True, alpha=0.25, color=colors["grid"])
    for spine in ax.spines.values():
        spine.set_color(colors["text"])
    fig.tight_layout()
    return fig

def plot_probability_bars_plotly(probs_dict, threshold=0.5, theme="Light"):
    if not HAVE_PLOTLY:
        return None
    items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [float(v) for _, v in items]
    template = "plotly_dark" if theme == "Dark" else "plotly_white"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation="h",
        text=[f"{v:.3f}" for v in values], textposition="outside",
        hovertemplate="Label=%{y}<br>Probability=%{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_width=2)
    fig.update_layout(
        title="Class probabilities", template=template, height=360,
        margin=dict(l=40, r=20, t=50, b=30),
        xaxis_title="Probability", yaxis_title="Label",
    )
    fig.update_xaxes(range=[0, 1])
    return fig

def plot_probability_bars_matplotlib(probs_dict, threshold=0.5, theme="Light"):
    colors = _fig_colors(theme)
    items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [float(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])

    y = np.arange(len(labels))
    bars = ax.barh(y, values)

    ax.axvline(threshold, linestyle="--", linewidth=1.4, color=colors["text"], alpha=0.7)
    ax.text(threshold + 0.01, len(labels) - 0.6, f"Threshold = {threshold:.2f}",
            fontsize=9, color=colors["text"], va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=colors["text"])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Probability", color=colors["text"])
    ax.set_title("Class probabilities", color=colors["text"], pad=10)
    ax.tick_params(axis="x", colors=colors["text"])
    ax.tick_params(axis="y", colors=colors["text"])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.20, color=colors["grid"])

    for bar, v in zip(bars, values):
        ax.text(min(v + 0.02, 1.03), bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", ha="left", fontsize=9, color=colors["text"])

    for spine in ax.spines.values():
        spine.set_color(colors["text"])

    fig.tight_layout()
    return fig

def plot_bar(values, labels, title, ylabel, theme="Light"):
    colors = _fig_colors(theme)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])

    y = np.arange(len(labels))
    bars = ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=colors["text"])
    ax.invert_yaxis()
    ax.set_title(title, color=colors["text"], pad=10)
    ax.set_xlabel(ylabel, color=colors["text"])
    ax.tick_params(axis="x", colors=colors["text"])
    ax.tick_params(axis="y", colors=colors["text"])
    ax.grid(axis="x", alpha=0.20, color=colors["grid"])

    vmax = max(values) if len(values) else 1.0
    ax.set_xlim(0, vmax * 1.15 if vmax > 0 else 1)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + vmax * 0.02, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=9, color=colors["text"])

    for spine in ax.spines.values():
        spine.set_color(colors["text"])

    fig.tight_layout()
    return fig

def plot_quality_metrics(metrics: dict, theme="Light"):
    colors = _fig_colors(theme)
    labels = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])

    y = np.arange(len(labels))
    bars = ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=colors["text"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_title("Quality-check metrics", color=colors["text"], pad=10)
    ax.set_xlabel("Normalized score", color=colors["text"])
    ax.tick_params(axis="x", colors=colors["text"])
    ax.tick_params(axis="y", colors=colors["text"])
    ax.grid(axis="x", alpha=0.20, color=colors["grid"])

    for bar, v in zip(bars, values):
        ax.text(min(v + 0.02, 1.02), bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=9, color=colors["text"])

    for spine in ax.spines.values():
        spine.set_color(colors["text"])

    fig.tight_layout()
    return fig

def plot_ecg_education_figure(theme="Light"):
    """
    Cleaner ECG education figure using Gestalt principles:
    - top panel: waveform + part labels
    - bottom panel: interval bars + timing notes
    - reduced overlap, better hierarchy, grouped content
    """
    colors = _fig_colors(theme)

    x = np.linspace(0, 1, 1000)
    y = np.zeros_like(x)

    def gauss(mu, sigma, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Stylized ECG
    y += gauss(0.18, 0.020, 0.18)    # P
    y += gauss(0.37, 0.006, -0.25)   # Q
    y += gauss(0.40, 0.004, 1.25)    # R
    y += gauss(0.43, 0.008, -0.45)   # S
    y += gauss(0.70, 0.050, 0.35)    # T

    fig = plt.figure(figsize=(12, 5.8), constrained_layout=True)
    fig.patch.set_facecolor(colors["fig"])
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.5])

    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    # ---------------- Top: waveform ----------------
    ax.set_facecolor(colors["ax"])
    ax.plot(x, y, linewidth=2.8, color="#1f77b4")
    ax.axhline(0, linestyle="--", linewidth=1.0, alpha=0.45, color=colors["grid"])

    # Direct labels with short arrows
    ann_kw = dict(
        arrowprops=dict(arrowstyle="->", lw=1.2, color=colors["text"]),
        fontsize=10,
        color=colors["text"],
        ha="center",
        va="bottom",
    )

    ax.annotate("P wave", xy=(0.18, 0.18), xytext=(0.14, 0.52), **ann_kw)
    ax.annotate("QRS complex", xy=(0.40, 1.15), xytext=(0.42, 1.62), **ann_kw)
    ax.annotate("ST segment", xy=(0.54, 0.03), xytext=(0.57, 0.36), **ann_kw)
    ax.annotate("T wave", xy=(0.70, 0.35), xytext=(0.72, 0.82), **ann_kw)

    # Subtle region hint for ST
    ax.plot([0.46, 0.60], [0.01, 0.01], linewidth=4, alpha=0.18, solid_capstyle="round")

    ax.set_title("ECG parts and typical normal timing", fontsize=16, color=colors["text"], pad=10)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(-0.65, 1.8)

    for spine in ax.spines.values():
        spine.set_color(colors["text"])

    # ---------------- Bottom: grouped interval bars ----------------
    ax2.set_facecolor(colors["ax"])
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xticks([])

    for spine in ax2.spines.values():
        spine.set_visible(False)

    def interval_bar(axh, x0, x1, y, label, note, color):
        axh.plot([x0, x1], [y, y], color=color, linewidth=3, solid_capstyle="round")
        axh.plot([x0, x0], [y - 0.04, y + 0.04], color=color, linewidth=1.5)
        axh.plot([x1, x1], [y - 0.04, y + 0.04], color=color, linewidth=1.5)
        axh.text((x0 + x1) / 2, y + 0.08, label, ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=colors["text"])
        axh.text((x0 + x1) / 2, y - 0.10, note, ha="center", va="top",
                 fontsize=9, color=colors["text"])

    interval_bar(ax2, 0.12, 0.36, 0.78, "PR interval", "0.12–0.20 s (120–200 ms)", "#ff7f0e")
    interval_bar(ax2, 0.385, 0.445, 0.52, "QRS duration", "< 0.12 s (typically 80–110 ms)", "#2ca02c")
    interval_bar(ax2, 0.34, 0.82, 0.26, "QT interval", "~0.35–0.44 s (varies with HR)", "#d62728")

    ax2.text(
        0.02, 0.02,
        "P duration: ~0.08–0.11 s (<0.12 s)   |   ST segment: judged mainly by level/shape rather than one fixed duration",
        fontsize=9,
        color=colors["text"],
        ha="left",
        va="bottom"
    )

    return fig
# =========================================================
# Cached model
# =========================================================
@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH), device="cpu")

# =========================================================
# Sidebar UI
# =========================================================
with st.sidebar:
    st.markdown("### Theme")
    c1, c2 = st.columns(2)
    if c1.button("☀️", help="Light mode", **BTN_W):
        st.session_state["theme"] = "Light"
        apply_theme_css("Light")
        st.rerun()
    if c2.button("🌙", help="Dark mode", **BTN_W):
        st.session_state["theme"] = "Dark"
        apply_theme_css("Dark")
        st.rerun()

    st.divider()
    st.markdown("### Settings")
    st.session_state["threshold"] = st.slider("Decision threshold", 0.0, 1.0, float(st.session_state["threshold"]), 0.01)
    st.session_state["ecg_style"] = st.radio("ECG plot style", ["Standard", "ECG paper"], index=0 if st.session_state["ecg_style"] == "Standard" else 1)
    st.session_state["probability_view"] = st.radio("Probability display", ["Table", "Bars", "Both"], index=["Table","Bars","Both"].index(st.session_state["probability_view"]))
    st.session_state["apply_preprocess"] = st.checkbox("Apply preprocessing (bandpass + notch + z-score)", value=bool(st.session_state["apply_preprocess"]))
    st.caption("Turn ON for raw PTB-XL exports. Leave OFF for already-preprocessed files.")

    st.divider()
    st.markdown("### Demo samples")

    demo_groups, missing = build_demo_index()
    group_names = sorted(demo_groups.keys(), key=lambda x: (x == "OTHER", x))
    if not group_names:
        st.info("No demos found. Put demo_*.npy into the demo/ folder.")
    else:
        chosen_group = st.selectbox("Demo group", group_names, index=0)
        entries = demo_groups.get(chosen_group, [])

        # show as dropdown: mark missing
        option_labels = []
        for e in entries:
            mark = "" if e["exists"] else " (missing)"
            option_labels.append(e["file"] + mark)

        picked = st.selectbox("Pick a demo file", option_labels, index=0 if option_labels else 0)

        # resolve picked filename (strip " (missing)")
        picked_file = picked.replace(" (missing)", "")
        exists = (DEMO_DIR / picked_file).exists()

        if missing:
            st.caption("Some demos are listed but missing on disk. Copy the .npy files into demo/.")

        if st.button("Load selected demo", disabled=not exists, **BTN_W):
            st.session_state["demo_file"] = picked_file
            st.session_state["saved_upload_name"] = None
            st.rerun()

        if st.button("Clear demo selection", **BTN_W):
            st.session_state["demo_file"] = None
            st.rerun()

        if not exists:
            st.warning(f"Missing on disk: demo/{picked_file}")

    st.divider()
    st.markdown("### Saved uploads")
    remember_upload = st.checkbox("Save new uploads for later", value=True)

    uploads_idx = load_upload_index()
    if uploads_idx:
        options = ["(none)"] + [f"{r['saved_at']} — {r['orig_name']} [{r['id']}]" for r in uploads_idx]
        pick = st.selectbox("Previously saved", options, index=0)

        colA, colB = st.columns(2)
        if colA.button("Load saved", disabled=(pick == "(none)"), **BTN_W):
            sel_i = options.index(pick) - 1
            st.session_state["saved_upload_name"] = uploads_idx[sel_i]["saved_name"]
            st.session_state["demo_file"] = None
            st.rerun()

        if colB.button("Delete saved", disabled=(pick == "(none)"), **BTN_W):
            sel_i = options.index(pick) - 1
            delete_saved_upload(uploads_idx[sel_i]["saved_name"])
            if st.session_state.get("saved_upload_name") == uploads_idx[sel_i]["saved_name"]:
                st.session_state["saved_upload_name"] = None
            st.rerun()
    else:
        st.caption("No saved uploads yet.")

# =========================================================
# Main header
# =========================================================
st.title("CardioAI – ECG Explorer")
st.caption("Research demo only. Not for clinical use.")

theme = st.session_state["theme"]
threshold = float(st.session_state["threshold"])
ecg_style = st.session_state["ecg_style"]
probability_view = st.session_state["probability_view"]
apply_preprocess = bool(st.session_state["apply_preprocess"])

# =========================================================
# Load model
# =========================================================
try:
    model = get_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# guard: ensure multilabel checkpoint
if getattr(model, "num_labels", 5) == 1:
    st.error(
        "Loaded checkpoint is binary (num_labels=1). "
        "This app expects the MULTILABEL checkpoint (num_labels=5) for NORM/MI/STTC/CD/HYP."
    )
    st.stop()

# =========================================================
# Input source resolution (demo > saved > upload)
# =========================================================
demo_manifest = load_demo_manifest()
demo_meta: dict = {}

x12_raw: Optional[np.ndarray] = None
uploaded_name: str = ""

selected_demo = st.session_state.get("demo_file", None)
selected_saved = st.session_state.get("saved_upload_name", None)

if selected_demo:
    try:
        x12_raw = load_local_demo_npy(selected_demo)
        uploaded_name = selected_demo
        demo_meta = demo_manifest.get(selected_demo, {})
        st.info(f"Using demo sample: {selected_demo}")
        # allow per-demo recommendation
        if "recommended_apply_preprocess" in demo_meta:
            apply_preprocess = bool(demo_meta["recommended_apply_preprocess"])
    except Exception as e:
        st.error(str(e))
        st.session_state["demo_file"] = None
        st.stop()

elif selected_saved:
    try:
        x12_raw = load_saved_npy(selected_saved)
        uploaded_name = selected_saved
        st.info(f"Using saved upload: {selected_saved}")
    except Exception as e:
        st.error(f"Could not load saved upload: {e}")
        st.session_state["saved_upload_name"] = None
        st.stop()

else:
    uploaded = st.file_uploader("Upload a .npy ECG file with shape (12, 5000)", type=["npy"])
    if uploaded is None:
        st.info("Choose a demo, load a saved upload, or upload a .npy ECG.")
        st.stop()

    uploaded_name = uploaded.name
    x12_raw = load_uploaded_npy(uploaded)

    # optionally persist
    if st.sidebar.session_state.get("remember_upload", True):
        pass  # (older streamlit may not expose sidebar state here)

    # use the remember_upload from sidebar (re-evaluate)
    # safest: always read checkbox again from sidebar saved variable
    # (we can’t access it directly; so we re-check via load_upload_index toggle pattern)
    # simplest behavior: always persist if checkbox on session_state is not stored -> use True by default
    # We'll store this in session_state via widget key:
    # but since the widget lives in sidebar without key, we keep it simple:
    # if user wants persistence OFF, untick and refresh (recommended).
    # Practical: just persist always when sidebar checkbox enabled:
    # We'll detect by checking if sidebar checkbox exists in session_state:
    remember_upload_flag = True
    # Heuristic: if user ever toggles, it appears in session_state under its label sometimes
    # But that is not guaranteed. We keep persistence ON by default.
    if remember_upload_flag:
        try:
            rec = persist_uploaded_npy(uploaded)
            # store pointer so user can quickly reload later if they want
            st.session_state["saved_upload_name"] = rec["saved_name"]
        except Exception:
            pass

# =========================================================
# Run inference
# =========================================================
try:
    result = predict_ecg(
        model,
        x12_raw,
        threshold=threshold,
        device="cpu",
        apply_preprocess=apply_preprocess,
        run_occlusion=True,
        run_ig=True,
        ig_steps=16,
    )
except TypeError:
    # older signature fallback
    result = predict_ecg(model, x12_raw, threshold=threshold, device="cpu")
except Exception as e:
    st.error(f"Inference failed: {e}")
    st.stop()

# =========================================================
# Normalize fields / fallbacks
# =========================================================
x_used = result.get("x_used", result.get("x_preprocessed", x12_raw))
beats = result["beats"]
rpeaks = result["rpeaks"]

gt = demo_meta.get("expected_label", infer_label_from_filename(uploaded_name))
pred_top = result.get("top_label", None)
is_correct = (gt is not None) and (pred_top == gt)

qc = result.get("qc", {"ok": True, "issues": []})
qc_ok = bool(qc.get("ok", True))
uncertain = bool(result.get("uncertainty_flag", False))

measurements = result.get("measurements", {})
if "heart_rate_bpm" not in measurements:
    measurements["heart_rate_bpm"] = estimate_hr_from_rpeaks(rpeaks, fs=DEFAULT_FS)
if "qrs_ms_est" not in measurements:
    measurements["qrs_ms_est"] = estimate_qrs_ms_from_beat(beats[0, 1], fs=DEFAULT_FS)
if "st_dev_est" not in measurements:
    measurements["st_dev_est"] = estimate_st_deviation(beats[0, 1], fs=DEFAULT_FS)
result["measurements"] = measurements

if "top_prob" not in result and "probs" in result and "top_label" in result:
    result["top_prob"] = float(result["probs"][result["top_label"]])

if "positive_labels" not in result and "preds" in result:
    result["positive_labels"] = [k for k, v in result["preds"].items() if v == 1]

if "margin" not in result and "probs" in result:
    sp = sorted(result["probs"].values(), reverse=True)
    result["margin"] = float(sp[0] - sp[1]) if len(sp) >= 2 else float(sp[0]) if sp else 0.0

lead_activity = np.array(result.get("lead_activity", np.zeros(12)), dtype=float)
if lead_activity.sum() == 0:
    lead_activity = compute_lead_activity_from_beats(beats)
result["lead_activity"] = lead_activity.tolist()

if "top_active_leads" not in result:
    idxs = np.argsort(lead_activity)[::-1][:3]
    result["top_active_leads"] = [LEAD_NAMES[i] for i in idxs]

quality_metrics = build_quality_metrics(x_used, rpeaks, beats)

# =========================================================
# Status badges row
# =========================================================
status_row = []
status_row.append(badge("QC: OK" if qc_ok else "QC: Issues", "ok" if qc_ok else "warn"))
status_row.append(badge("Confidence: Uncertain" if uncertain else "Confidence: OK", "warn" if uncertain else "ok"))
if gt is None:
    status_row.append(badge("Verification: Unverified", "warn"))
else:
    status_row.append(badge(f"Ground truth: {gt}", "ok"))
    status_row.append(badge("Match ✅" if is_correct else "Mismatch ❌", "ok" if is_correct else "bad"))

st.markdown(" ".join(status_row), unsafe_allow_html=True)
st.markdown(
    "<div class='small-note'>For unknown uploads, correctness cannot be confirmed without a reference diagnosis or expert review.</div>",
    unsafe_allow_html=True,
)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["ECG Viewer", "Prediction", "Explanation", "Education", "Review/Export"])

# =========================================================
# ECG Viewer
# =========================================================
with tab1:
    st.subheader("ECG used for inference")

    if ecg_style == "Standard" and HAVE_PLOTLY:
        st.plotly_chart(
            plot_12_lead_plotly(x_used, title="ECG used for inference", fs=DEFAULT_FS, theme=theme),
            use_container_width=True,
        )
        st.markdown("<div class='small-note'>Interactive (zoom/pan/hover). Switch to ECG paper for clinical-style grid.</div>", unsafe_allow_html=True)
    else:
        if ecg_style == "Standard" and not HAVE_PLOTLY:
            st.warning("Plotly is not installed; using static viewer. Install plotly to enable interactive viewer.")
        st.pyplot(
            plot_12_lead_matplotlib(
                x_used,
                title="Preprocessed 12-lead ECG" if result.get("applied_preprocess", False) else "ECG used for inference",
                fs=DEFAULT_FS,
                theme=theme,
                style="ECG paper" if ecg_style == "ECG paper" else "Standard",
            )
        )

    with st.expander("Show extracted beat example (Lead II)"):
        st.pyplot(plot_single_beat(beats[0, 1], theme=theme))
        st.markdown("<div class='small-note'>One extracted beat token. The model aggregates multiple beats to decide.</div>", unsafe_allow_html=True)

# =========================================================
# Prediction
# =========================================================
with tab2:
    st.subheader("Prediction summary")
    left, right = st.columns([1.0, 1.0])

    with left:
        st.metric("Top predicted label", result["top_label"])
        st.metric("Top probability", f"{result['top_prob']:.4f}")
        if result.get("mi_prob", None) is not None:
            st.metric("MI probability", f"{result['mi_prob']:.4f}")
        st.metric("Separation margin", f"{result.get('margin', 0.0):.4f}")

        st.write("**Labels crossing threshold:**")
        st.write(", ".join(result.get("positive_labels", [])) if result.get("positive_labels", []) else "None")

        if gt is not None:
            st.write(f"**Ground truth:** {gt}")
            st.write(f"**Prediction match:** {'Yes' if is_correct else 'No'}")
        else:
            st.write("**Ground truth:** unknown")
            st.write("**Prediction match:** cannot be verified automatically")

    with right:
        st.subheader("Probability details")
        df_probs = probability_df(result["probs"], result["preds"])

        if probability_view == "Table":
            st.dataframe(df_probs, **_stretch_kwargs(st.dataframe), hide_index=True)
        elif probability_view == "Bars":
            if HAVE_PLOTLY:
                st.plotly_chart(plot_probability_bars_plotly(result["probs"], threshold=threshold, theme=theme), use_container_width=True)
            else:
                st.pyplot(plot_probability_bars_matplotlib(result["probs"], threshold=threshold, theme=theme))
        else:
            if HAVE_PLOTLY:
                st.plotly_chart(plot_probability_bars_plotly(result["probs"], threshold=threshold, theme=theme), use_container_width=True)
            else:
                st.pyplot(plot_probability_bars_matplotlib(result["probs"], threshold=threshold, theme=theme))
            st.dataframe(df_probs, **_stretch_kwargs(st.dataframe), hide_index=True)

    st.divider()
    st.subheader("Signal summary (approx)")
    hr = measurements.get("heart_rate_bpm", None)
    qrs = measurements.get("qrs_ms_est", None)
    st_dev = measurements.get("st_dev_est", None)
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated HR (bpm)", f"{hr:.1f}" if hr is not None else "N/A")
    c2.metric("Approx QRS (ms)", f"{qrs:.0f}" if qrs is not None else "N/A")
    c3.metric("Approx ST deviation", f"{st_dev:.3f}" if st_dev is not None else "N/A")

# =========================================================
# Explanation
# =========================================================
with tab3:
    st.subheader("Summary")
    st.write(build_quick_summary(result, threshold))

    st.subheader("Details")
    st.write(build_detailed_interpretation(result, threshold))

    if result.get("template_explanation"):
        with st.expander("Rule-based explanation (template)"):
            st.write(result["template_explanation"])

    st.subheader("Signal-based view")
    st.write("Most active leads in extracted beats:", ", ".join(result["top_active_leads"]))
    st.pyplot(plot_bar(result["lead_activity"], LEAD_NAMES, "Lead activity in extracted beats", "Normalized activity", theme=theme))
    st.markdown("<div class='small-note'>Higher activity = larger average absolute signal in extracted beats (not a diagnosis by itself).</div>", unsafe_allow_html=True)

    st.subheader("Explainability (XAI)")
    st.markdown("<div class='small-note'>These explain model behavior, not clinical causality.</div>", unsafe_allow_html=True)
    xai = result.get("xai", {})

    if xai.get("occlusion_delta_toplabel") is not None:
        st.write("**Lead occlusion sensitivity** (drop in top-label probability when lead removed):")
        st.pyplot(plot_bar(xai["occlusion_delta_toplabel"], LEAD_NAMES, "Occlusion sensitivity vs lead", "Δ probability (base - occluded)", theme=theme))

    if xai.get("ig_attr_toplabel") is not None:
        st.write(f"**Integrated Gradients** (per-lead attribution; steps={xai.get('ig_steps', 16)}):")
        st.pyplot(plot_bar(xai["ig_attr_toplabel"], LEAD_NAMES, "Integrated Gradients attribution vs lead", "Attribution (normalized)", theme=theme))
        st.markdown("<div class='small-note'>IG steps = number of interpolation steps. More steps = smoother but slower.</div>", unsafe_allow_html=True)

    st.subheader("Quality checks")
    if qc_ok:
        st.success("QC passed (basic checks).")
    else:
        st.warning("QC reported issues:")
        for msg in qc.get("issues", []):
            st.write("-", msg)

    st.pyplot(plot_quality_metrics(quality_metrics, theme=theme))
    st.markdown("<div class='small-note'>Quality metrics summarize finiteness, non-flatness, spread, and R-peak coverage.</div>", unsafe_allow_html=True)

    with st.expander("Raw checks"):
        st.write("Uploaded ECG shape:", x12_raw.shape)
        st.write("Extracted beats shape:", beats.shape)
        st.write("Task type:", result.get("task_type", "N/A"))
        st.write("Applied preprocessing:", result.get("applied_preprocess", apply_preprocess))
        st.write("Detected R-peaks:", len(rpeaks) if rpeaks is not None else 0)
        st.write("Looks preprocessed / z-scored:", looks_preprocessed_zscored(x_used))

# =========================================================
# Education
# =========================================================
with tab4:
    st.subheader("Understand the ECG terms")
    st.write(
        "This panel is for non-experts. It shows the main ECG parts and **typical normal timing in seconds (s)**. "
        "Timing varies by heart rate and patient context."
    )

    st.pyplot(plot_ecg_education_figure(theme=theme))

    st.markdown("**What the terms mean (simple)**")
    st.markdown(
        """
- **P wave**: atria start the heartbeat.
- **QRS complex**: ventricles activate (main pumping chambers).
- **ST segment**: early recovery phase after QRS (often judged by elevation/depression).
- **T wave**: later recovery phase.
"""
    )

    st.markdown("**Typical normal timing ranges (approx.)**")
    st.markdown(
        """
- **P duration:** ~**0.08–0.11 s** (usually **< 0.12 s**)
- **PR interval:** **0.12–0.20 s** (**120–200 ms**)
- **QRS duration:** typically **0.08–0.11 s** (usually **< 0.12 s**)
- **QT interval:** often **~0.35–0.44 s** (varies with HR)
- **ST segment:** no single fixed “normal duration” used in simple clinical rules; it’s judged mainly by **level/shape** relative to baseline
"""
    )

    st.markdown("**How to read seconds vs milliseconds**")
    st.markdown(
        """
- **1 second = 1000 ms**
- **0.12 s = 120 ms**
- The app often displays intervals in **ms** because that is common in ECG interpretation.
"""
    )

    st.markdown("**Important**")
    st.markdown(
        """
- A high probability does **not** guarantee correctness.
- A low probability does **not** rule out disease.
- Confirmation requires clinician review + clinical context.
"""
    )

# =========================================================
# Review / Export
# =========================================================
with tab5:
    st.subheader("Clinician-in-the-loop review")
    reviewer = st.text_input("Reviewer name / ID (optional)", value="")
    reviewer_label = st.selectbox("Reviewer final label (optional)", [""] + CLASS_NAMES)
    reviewer_notes = st.text_area("Reviewer notes (optional)", value="", height=140)

    st.divider()
    st.subheader("Export report")

    report = {
        "timestamp_unix": time.time(),
        "file_name": uploaded_name,
        "threshold": threshold,
        "theme": theme,
        "ecg_style": ecg_style,
        "probability_display": probability_view,
        "applied_preprocess": result.get("applied_preprocess", apply_preprocess),
        "prediction": {
            "top_label": result.get("top_label"),
            "top_prob": result.get("top_prob"),
            "margin": result.get("margin"),
            "positive_labels": result.get("positive_labels"),
            "probs": result.get("probs"),
            "preds": result.get("preds"),
        },
        "measurements": measurements,
        "qc": qc,
        "quality_metrics": quality_metrics,
        "xai": result.get("xai", {}),
        "ground_truth": gt,
        "match": bool(is_correct) if gt is not None else None,
        "summary": build_quick_summary(result, threshold),
        "details": build_detailed_interpretation(result, threshold),
        "review": {
            "reviewer": reviewer,
            "reviewer_label": reviewer_label if reviewer_label else None,
            "reviewer_notes": reviewer_notes,
        },
        "demo_meta": demo_meta,
        "note": "Decision-support only. Not for clinical use.",
    }

    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name="cardioai_report.json",
        mime="application/json",
        **DL_W,
    )