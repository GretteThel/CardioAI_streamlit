from __future__ import annotations

import hashlib
import inspect
import json
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# Plotly optional
try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    go = None
    HAVE_PLOTLY = False

# Hugging Face optional
try:
    from huggingface_hub import hf_hub_download
    HAVE_HF_HUB = True
except Exception:
    hf_hub_download = None
    HAVE_HF_HUB = False

# PDF support optional
try:
    import fitz  # PyMuPDF
    HAVE_FITZ = True
except Exception:
    fitz = None
    HAVE_FITZ = False

from cardioai_infer import (
    CLASS_NAMES,
    LEAD_NAMES,
    FS,
    load_model,
    load_uploaded_npy,
    predict_ecg,
)

# =========================================================
# App constants
# =========================================================
APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_INDEX_PATH = UPLOAD_DIR / "uploads_index.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_MODEL_PATH = APP_DIR / "assets" / "best_hybrid_final.pt"
PRIMARY_DEMO_DIR = APP_DIR / "assets" / "demo"
SECONDARY_DEMO_DIR = APP_DIR / "demo"
LOCAL_DEMO_MANIFEST_PATH = PRIMARY_DEMO_DIR / "demo_manifest.json"

HF_MODEL_REPO = "grettezybelle/cardioai-model"
HF_DEMO_REPO = "grettezybelle/cardioai-demos"

DEFAULT_FS = FS
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTS = {".pdf"}
SIGNAL_EXTS = {".npy"}

st.set_page_config(page_title="CardioAI – ECG Explorer", layout="wide")


# =========================================================
# UI helpers
# =========================================================
def _stretch_kwargs(fn):
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


BTN_W = _stretch_kwargs(st.button)
DL_W = _stretch_kwargs(st.download_button)


def init_session_state() -> None:
    defaults = {
        "theme": "Light",
        "ecg_style": "Standard",
        "probability_view": "Both",
        "threshold": 0.50,
        "apply_preprocess": False,
        "demo_file": None,
        "saved_upload_name": None,
        "remember_upload": True,
        "use_hf_assets": True,
        "image_layout": "3x4 standard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _fig_colors(theme: str) -> dict:
    if theme == "Dark":
        return {
            "fig": "#0b1220",
            "ax": "#0f172a",
            "text": "#e5e7eb",
            "grid": "#243244",
            "muted": "#94a3b8",
            "accent": "#60a5fa",
        }
    return {
        "fig": "white",
        "ax": "white",
        "text": "#111827",
        "grid": "#d1d5db",
        "muted": "#6b7280",
        "accent": "#2563eb",
    }


def apply_theme_css(theme: str) -> None:
    c = _fig_colors(theme)

    if theme == "Dark":
        css = f"""
        <style>
        :root {{
            --bg-main: #0b1220;
            --bg-card: #0f172a;
            --bg-soft: #111827;
            --border: #243244;
            --text: {c["text"]};
            --muted: {c["muted"]};
            --accent: {c["accent"]};
        }}

        .stApp {{
            background-color: var(--bg-main) !important;
            color: var(--text) !important;
        }}

        section[data-testid="stSidebar"] {{
            background: var(--bg-soft) !important;
            border-right: 1px solid var(--border);
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--text) !important;
        }}

        h1, h2, h3, h4, h5, h6, p, label, div, span {{
            color: var(--text) !important;
        }}

        .small-note {{
            color: var(--muted) !important;
            font-size: 0.9rem;
        }}

        .badge {{
            display:inline-block;
            padding:0.18rem 0.55rem;
            border-radius:0.55rem;
            font-size:0.85rem;
            margin-right:0.35rem;
            margin-bottom:0.3rem;
        }}
        .badge-ok {{
            background:#14532d;
            color:#dcfce7 !important;
        }}
        .badge-warn {{
            background:#713f12;
            color:#fef9c3 !important;
        }}
        .badge-bad {{
            background:#7f1d1d;
            color:#fee2e2 !important;
        }}

        .card {{
            padding: 0.9rem 1rem;
            border: 1px solid var(--border);
            border-radius: 0.9rem;
            background: var(--bg-card);
        }}

        section.main > div {{
            padding-top: 1.2rem;
        }}

        button[data-baseweb="tab"] {{
            color: var(--text) !important;
            background: transparent !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            border-bottom: 2px solid var(--accent) !important;
            color: #ffffff !important;
        }}

        .stButton > button, .stDownloadButton > button {{
            background: var(--bg-card) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0.7rem !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--accent) !important;
            color: #ffffff !important;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }}

        input, textarea {{
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }}

        [data-testid="stFileUploader"] {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0.8rem !important;
            padding: 0.4rem;
        }}
        [data-testid="stFileUploader"] * {{
            color: var(--text) !important;
        }}

        [data-testid="metric-container"] {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0.8rem !important;
            padding: 0.8rem !important;
        }}
        [data-testid="metric-container"] * {{
            color: var(--text) !important;
        }}

        [data-testid="stDataFrame"] {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0.8rem !important;
        }}

        table, thead, tbody, tr, th, td {{
            color: var(--text) !important;
            background-color: transparent !important;
        }}

        details {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 0.8rem !important;
            padding: 0.25rem 0.5rem !important;
        }}
        details summary {{
            color: var(--text) !important;
        }}

        [data-testid="stAlert"] {{
            border-radius: 0.8rem !important;
        }}

        .stRadio label, .stCheckbox label, .stSlider label {{
            color: var(--text) !important;
        }}

        pre, code {{
            background: #111827 !important;
            color: #e5e7eb !important;
        }}

        hr {{
            border-color: var(--border) !important;
        }}
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

        [data-testid="metric-container"] {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 0.8rem;
            padding: 0.8rem;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def badge(text: str, kind: str) -> str:
    cls = {"ok": "badge-ok", "warn": "badge-warn", "bad": "badge-bad"}[kind]
    return f"<span class='badge {cls}'>{text}</span>"


# =========================================================
# Asset loading
# =========================================================
@st.cache_resource
def ensure_assets() -> tuple[Path, Path, Path]:
    if not HAVE_HF_HUB:
        raise ImportError("huggingface_hub is not installed")

    demo_dir = ASSETS_DIR / "demo"
    demo_dir.mkdir(exist_ok=True)

    model_local = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        filename="best_hybrid_final.pt",
        local_dir=str(ASSETS_DIR),
        local_dir_use_symlinks=False,
    )
    model_path = Path(model_local)

    manifest_local = hf_hub_download(
        repo_id=HF_DEMO_REPO,
        repo_type="dataset",
        filename="demo_manifest.json",
        local_dir=str(demo_dir),
        local_dir_use_symlinks=False,
    )
    demo_manifest_path = Path(manifest_local)

    try:
        manifest = json.loads(demo_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}

    for fname in manifest.keys():
        if fname.endswith(".npy"):
            hf_hub_download(
                repo_id=HF_DEMO_REPO,
                repo_type="dataset",
                filename=fname,
                local_dir=str(demo_dir),
                local_dir_use_symlinks=False,
            )

    return model_path, demo_dir, demo_manifest_path


def resolve_assets(use_hf_assets: bool = True):
    model_path = LOCAL_MODEL_PATH
    primary_demo_dir = PRIMARY_DEMO_DIR
    secondary_demo_dir = SECONDARY_DEMO_DIR
    demo_manifest_path = LOCAL_DEMO_MANIFEST_PATH
    asset_message = f"Using local assets from: {ASSETS_DIR}"

    if use_hf_assets:
        try:
            model_path, hf_demo_dir, demo_manifest_path = ensure_assets()
            primary_demo_dir = hf_demo_dir
            asset_message = "Using Hugging Face assets."
        except Exception as e:
            asset_message = f"Using local assets (HF unavailable: {e})"

    return model_path, primary_demo_dir, secondary_demo_dir, demo_manifest_path, asset_message


# =========================================================
# Upload persistence
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


def persist_uploaded_file(uploaded_file, extra_meta: dict | None = None) -> dict:
    raw_bytes = uploaded_file.getvalue()
    sha12 = _sha256_bytes(raw_bytes)[:12]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(uploaded_file.name).suffix.lower()
    safe_stem = Path(uploaded_file.name).stem.replace(" ", "_")
    out_name = f"{ts}__{safe_stem}__{sha12}{ext}"
    out_path = UPLOAD_DIR / out_name

    idx = load_upload_index()
    for rec in idx:
        if rec.get("sha12") == sha12 and rec.get("orig_name") == uploaded_file.name:
            return rec

    out_path.write_bytes(raw_bytes)

    rec = {
        "id": sha12,
        "sha12": sha12,
        "orig_name": uploaded_file.name,
        "saved_name": out_name,
        "saved_path": str(out_path),
        "saved_at": ts,
        "ext": ext,
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


# =========================================================
# Image -> signal digitization helpers
# =========================================================
def resample_1d(sig: np.ndarray, target_len: int) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    if len(sig) == target_len:
        return sig
    x_old = np.linspace(0.0, 1.0, len(sig))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, sig).astype(np.float32)


def validate_signal_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D ECG array, got shape {x.shape}")

    if x.shape[0] != 12 and x.shape[1] == 12:
        x = x.T

    if x.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, got shape {x.shape}")

    if x.shape[1] != 5000:
        x = np.vstack([resample_1d(lead, 5000) for lead in x])

    return x.astype(np.float32)


def crop_border(arr: np.ndarray, frac: float = 0.03) -> np.ndarray:
    h, w = arr.shape[:2]
    y0, y1 = int(h * frac), int(h * (1.0 - frac))
    x0, x1 = int(w * frac), int(w * (1.0 - frac))
    return arr[y0:y1, x0:x1]


def split_ecg_grid(gray: np.ndarray, layout: str = "3x4_standard") -> list[np.ndarray]:
    gray = crop_border(gray, frac=0.03)

    if layout == "3x4_standard":
        nrows, ncols = 3, 4
    elif layout == "4x3_stacked":
        nrows, ncols = 4, 3
    else:
        raise ValueError(f"Unknown layout: {layout}")

    h, w = gray.shape
    row_h = h // nrows
    col_w = w // ncols

    panels = []
    for r in range(nrows):
        for c in range(ncols):
            y0, y1 = r * row_h, (r + 1) * row_h
            x0, x1 = c * col_w, (c + 1) * col_w
            panels.append(gray[y0:y1, x0:x1])

    return panels


def reorder_panels_to_standard_12lead(panels: list[np.ndarray], layout: str) -> list[np.ndarray]:
    if len(panels) != 12:
        raise ValueError(f"Expected 12 lead panels, got {len(panels)}")

    if layout == "4x3_stacked":
        return panels

    if layout == "3x4_standard":
        idx = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        return [panels[i] for i in idx]

    raise ValueError(f"Unknown layout: {layout}")


def trace_panel_to_signal(panel: np.ndarray) -> np.ndarray:
    arr = panel.astype(np.float32)
    arr = 255.0 - arr
    arr = arr / max(arr.max(), 1.0)

    h, w = arr.shape
    y_trace = np.zeros(w, dtype=np.float32)

    for x in range(w):
        col = arr[:, x]
        if col.max() < 0.08:
            y_trace[x] = h / 2.0
        else:
            y_trace[x] = float(np.argmax(col))

    kernel = np.ones(9, dtype=np.float32) / 9.0
    y_trace = np.convolve(y_trace, kernel, mode="same")

    sig = -(y_trace - (h / 2.0)) / max(h / 2.0, 1.0)
    return sig.astype(np.float32)


def digitize_ecg_image(image: Image.Image, layout: str = "3x4_standard") -> np.ndarray:
    gray = ImageOps.autocontrast(image.convert("L"))
    gray_arr = np.asarray(gray)

    panels = split_ecg_grid(gray_arr, layout=layout)
    panels = reorder_panels_to_standard_12lead(panels, layout=layout)

    leads = []
    for panel in panels:
        sig = trace_panel_to_signal(panel)
        sig = resample_1d(sig, 5000)
        leads.append(sig)

    x12 = np.vstack(leads).astype(np.float32)
    return validate_signal_array(x12)


def digitize_ecg_pdf(pdf_source, layout: str = "3x4_standard") -> np.ndarray:
    if not HAVE_FITZ:
        raise ImportError("PDF upload needs PyMuPDF. Install pymupdf.")

    if isinstance(pdf_source, (str, Path)):
        doc = fitz.open(str(pdf_source))
    else:
        doc = fitz.open(stream=pdf_source.getvalue(), filetype="pdf")

    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200, alpha=False)
    image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
    return digitize_ecg_image(image, layout=layout)


def load_saved_input(saved_name: str) -> np.ndarray:
    p = UPLOAD_DIR / saved_name
    ext = p.suffix.lower()

    if ext == ".npy":
        x = np.load(str(p), allow_pickle=True).astype(np.float32)
        return validate_signal_array(x)

    if ext in IMAGE_EXTS:
        image = Image.open(p).convert("RGB")
        return digitize_ecg_image(image, layout="3x4_standard")

    if ext in PDF_EXTS:
        return digitize_ecg_pdf(p, layout="3x4_standard")

    raise ValueError(f"Unsupported saved file type: {ext}")


# =========================================================
# Demo helpers
# =========================================================
def infer_label_from_filename(name: str) -> Optional[str]:
    s = name.lower()
    for lab in ["norm", "mi", "sttc", "cd", "hyp"]:
        if f"demo_{lab}" in s:
            return "NORM" if lab == "norm" else lab.upper()
    return None


def load_demo_manifest(demo_manifest_path: Path) -> dict:
    if demo_manifest_path.exists():
        try:
            return json.loads(demo_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def scan_demo_dirs(primary_demo_dir: Path, secondary_demo_dir: Path) -> List[Path]:
    files = []
    seen = set()

    for d in [primary_demo_dir, secondary_demo_dir]:
        if d.exists():
            for p in sorted(d.glob("demo_*.npy")):
                if p.name not in seen:
                    files.append(p)
                    seen.add(p.name)

    return files


def build_demo_index(primary_demo_dir: Path, secondary_demo_dir: Path, demo_manifest_path: Path) -> Tuple[Dict[str, List[dict]], List[str]]:
    manifest = load_demo_manifest(demo_manifest_path)
    scanned = scan_demo_dirs(primary_demo_dir, secondary_demo_dir)

    groups: Dict[str, List[dict]] = {}
    missing: List[str] = []
    seen = set()

    for fname, meta in manifest.items():
        label = meta.get("expected_label") or infer_label_from_filename(fname) or "OTHER"

        path_primary = primary_demo_dir / fname
        path_secondary = secondary_demo_dir / fname

        if path_primary.exists():
            full_path = path_primary
            exists = True
        elif path_secondary.exists():
            full_path = path_secondary
            exists = True
        else:
            full_path = path_primary
            exists = False
            missing.append(fname)

        groups.setdefault(label, []).append({
            "file": fname,
            "label": label,
            "exists": exists,
            "meta": meta,
            "path": str(full_path),
        })
        seen.add(fname)

    for p in scanned:
        if p.name in seen:
            continue
        label = infer_label_from_filename(p.name) or "OTHER"
        groups.setdefault(label, []).append({
            "file": p.name,
            "label": label,
            "exists": True,
            "meta": {},
            "path": str(p),
        })

    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda d: (not d["exists"], d["file"]))

    return groups, missing


def load_local_demo_npy(primary_demo_dir: Path, secondary_demo_dir: Path, filename: str) -> np.ndarray:
    candidates = [
        primary_demo_dir / filename,
        secondary_demo_dir / filename,
    ]

    for p in candidates:
        if p.exists():
            x = np.load(str(p), allow_pickle=True).astype(np.float32)
            if x.shape != (12, 5000):
                raise ValueError(f"{filename} must have shape (12,5000), got {x.shape}")
            return x

    raise FileNotFoundError(f"Demo file missing in both folders: {filename}")


# =========================================================
# Signal helpers
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
# Explanation builders
# =========================================================
def build_quick_summary(result, threshold):
    probs = result["probs"]
    preds = result["preds"]
    top = result["top_label"]
    top_prob = float(probs[top])
    mi_prob = result.get("mi_prob", None)
    positives = [k for k, v in preds.items() if v == 1]

    lines = []
    lines.append(f"The model's strongest pattern match is **{top}** with **{top_prob:.0%}** confidence.")

    if positives:
        lines.append(f"At the current threshold (**{threshold:.2f}**), the label(s) flagged are: **{', '.join(positives)}**.")
    else:
        lines.append(f"At the current threshold (**{threshold:.2f}**), no class is formally flagged.")

    if mi_prob is not None:
        lines.append(
            f"The **MI** score is **{mi_prob:.0%}**, "
            + ("so it is above the threshold." if mi_prob >= threshold else "so it stays below the threshold.")
        )

    hr = result.get("measurements", {}).get("heart_rate_bpm", None)
    if hr is not None:
        lines.append(f"The estimated heart rate is **{hr:.1f} bpm**.")

    if result.get("uncertainty_flag", False):
        lines.append("Model certainty is lower than usual, so this result should be interpreted more cautiously.")
    else:
        lines.append("The top class is clearly separated from the next most likely class.")

    lines.append("This is a decision-support output only and not a clinical diagnosis.")
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
    clarity = "clear separation" if margin >= 0.20 else "limited separation"

    lines = []
    lines.append(f"**Primary finding:** The strongest model output is **{top_label}** with probability **{top_prob:.3f}** ({conf} confidence).")
    lines.append(f"**Nearest alternative:** **{second_label}** at **{second_prob:.3f}**.")
    lines.append(f"**Confidence gap:** The difference between the top two classes is **{margin:.3f}**, which suggests **{clarity}**.")

    if pos:
        lines.append(f"**Threshold result:** At **{threshold:.2f}**, the class(es) above threshold are **{', '.join(pos)}**.")
    else:
        lines.append(f"**Threshold result:** No class crosses **{threshold:.2f}**.")

    signal_lines = []
    if hr is not None:
        signal_lines.append(f"heart rate ≈ **{hr:.1f} bpm**")
    if qrs is not None:
        signal_lines.append(f"QRS width ≈ **{qrs:.0f} ms**")
    if st_dev is not None:
        signal_lines.append(f"ST deviation in Lead II ≈ **{st_dev:.3f}** z-units")
    if signal_lines:
        lines.append("**Signal cues used for context:** " + "; ".join(signal_lines) + ".")

    if top_label == "CD" and qrs is not None:
        lines.append("**Plain-language interpretation:** A conduction-disturbance prediction can align with broader or slower ventricular conduction patterns. The displayed QRS estimate helps the reader see whether ventricular activation may look wider than usual, but it is only an approximation.")
    elif top_label == "MI":
        lines.append("**Plain-language interpretation:** An MI prediction means the model found a pattern that resembles myocardial infarction examples in training data. This should always be confirmed clinically because model similarity is not the same as diagnosis.")
    elif top_label == "STTC":
        lines.append("**Plain-language interpretation:** An ST/T change prediction means the model sees repolarization-related patterns that may involve ST shape or T-wave shape. These findings depend strongly on context and clinical review.")
    elif top_label == "HYP":
        lines.append("**Plain-language interpretation:** A hypertrophy-related prediction means the model detected a pattern that resembles chamber enlargement or thickening examples seen during training, not a direct anatomical measurement.")
    elif top_label == "NORM":
        lines.append("**Plain-language interpretation:** A normal prediction means the overall waveform looked more similar to normal examples than to the abnormal categories tracked by this model.")

    lines.append(f"**Preprocessing:** {'Applied' if applied else 'Not applied'}.")
    lines.append("_Decision-support summary only._")
    return "\n\n".join(lines)


# =========================================================
# Plot helpers
# =========================================================
def plot_12_lead_plotly(x12, title="ECG used for inference", fs=500, theme="Light"):
    if not HAVE_PLOTLY:
        return None

    colors = _fig_colors(theme)
    t = np.arange(x12.shape[1]) / float(fs)
    offsets = np.arange(12)[::-1] * 5.0
    template = "plotly_dark" if theme == "Dark" else "plotly_white"
    fig = go.Figure()

    for i, lead in enumerate(LEAD_NAMES):
        fig.add_trace(go.Scatter(
            x=t,
            y=x12[i] + offsets[i],
            mode="lines",
            name=lead,
            line=dict(width=1.15, color=colors["accent"]),
            hovertemplate=f"{lead}<br>t=%{{x:.3f}} s<br>amp=%{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        template=template,
        height=760,
        margin=dict(l=70, r=20, t=55, b=40),
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

    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])

    for i in range(12):
        ax.plot(t, x12[i] + offsets[i], linewidth=0.9, color=colors["accent"])

    ax.set_yticks(offsets)
    ax.set_yticklabels(LEAD_NAMES, color=colors["text"])
    ax.set_xlabel("Time (s)", color=colors["text"])
    ax.set_title(title, color=colors["text"], pad=10)
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
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])
    ax.plot(beat_lead2, linewidth=1.15, color=colors["accent"])
    ax.set_title("Example extracted beat (Lead II)", color=colors["text"], pad=8)
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
        x=values,
        y=labels,
        orientation="h",
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        hovertemplate="Label=%{y}<br>Probability=%{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_width=2)
    fig.update_layout(
        title="Class probabilities",
        template=template,
        height=380,
        margin=dict(l=50, r=35, t=55, b=35),
        xaxis_title="Probability",
        yaxis_title="Label",
    )
    fig.update_xaxes(range=[0, 1.08], showgrid=True)
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


def plot_bar(values, labels, title, xlabel, theme="Light"):
    colors = _fig_colors(theme)
    values = np.asarray(list(values), dtype=float)
    labels = list(labels)

    fig_h = max(4.3, 0.42 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(8.9, fig_h))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])

    y = np.arange(len(labels))
    bars = ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=colors["text"])
    ax.invert_yaxis()
    ax.set_title(title, color=colors["text"], pad=10)
    ax.set_xlabel(xlabel, color=colors["text"])
    ax.tick_params(axis="x", colors=colors["text"])
    ax.tick_params(axis="y", colors=colors["text"])
    ax.grid(axis="x", alpha=0.20, color=colors["grid"])

    vmin = float(values.min()) if len(values) else 0.0
    vmax = float(values.max()) if len(values) else 1.0
    span = max(vmax - vmin, 0.05)
    left = min(0.0, vmin - 0.10 * span)
    right = max(0.0, vmax + 0.18 * span)
    ax.set_xlim(left, right)

    for bar, v in zip(bars, values):
        if v >= 0:
            x_txt = v + 0.02 * span
            ha = "left"
        else:
            x_txt = v - 0.02 * span
            ha = "right"
        ax.text(x_txt, bar.get_y() + bar.get_height()/2, f"{v:.2f}",
                va="center", ha=ha, fontsize=9, color=colors["text"])

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
    colors = _fig_colors(theme)

    x = np.linspace(0, 1, 1000)
    y = np.zeros_like(x)

    def gauss(mu, sigma, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    y += gauss(0.18, 0.020, 0.18)
    y += gauss(0.37, 0.006, -0.25)
    y += gauss(0.40, 0.004, 1.25)
    y += gauss(0.43, 0.008, -0.45)
    y += gauss(0.70, 0.050, 0.35)

    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])
    ax.plot(x, y, linewidth=2.8, color=colors["accent"])
    ax.axhline(0, linestyle="--", linewidth=1.0, alpha=0.35, color=colors["grid"])

    ann_kw = dict(
        arrowprops=dict(arrowstyle="->", lw=1.2, color=colors["text"]),
        fontsize=10,
        color=colors["text"],
        ha="center",
        va="bottom",
    )
    ax.annotate("P wave", xy=(0.18, 0.18), xytext=(0.145, 0.52), **ann_kw)
    ax.annotate("QRS complex", xy=(0.40, 1.15), xytext=(0.42, 1.58), **ann_kw)
    ax.annotate("ST segment", xy=(0.54, 0.03), xytext=(0.57, 0.36), **ann_kw)
    ax.annotate("T wave", xy=(0.70, 0.35), xytext=(0.72, 0.78), **ann_kw)

    def interval_bar(x0, x1, y0, label, note, color):
        ax.plot([x0, x1], [y0, y0], color=color, linewidth=2.6, solid_capstyle="round")
        ax.plot([x0, x0], [y0 - 0.045, y0 + 0.045], color=color, linewidth=1.3)
        ax.plot([x1, x1], [y0 - 0.045, y0 + 0.045], color=color, linewidth=1.3)
        ax.text((x0 + x1) / 2, y0 + 0.08, label, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=colors["text"])
        ax.text((x0 + x1) / 2, y0 - 0.09, note, ha="center", va="top",
                fontsize=9, color=colors["text"])

    interval_bar(0.12, 0.36, -0.78, "PR interval", "0.12–0.20 s (120–200 ms)", "#ff7f0e")
    interval_bar(0.385, 0.445, -1.08, "QRS duration", "< 0.12 s (typically 80–110 ms)", "#2ca02c")
    interval_bar(0.34, 0.82, -1.40, "QT interval", "~0.35–0.44 s (varies with HR)", "#d62728")

    ax.text(
        0.02,
        -1.68,
        "P duration: ~0.08–0.11 s (<0.12 s)   |   ST segment: judged mainly by level/shape rather than one fixed duration",
        fontsize=9,
        color=colors["text"],
        ha="left",
        va="bottom",
    )

    ax.set_title("ECG parts and typical normal timing", fontsize=16, color=colors["text"], pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-1.85, 1.8)

    for spine in ax.spines.values():
        spine.set_color(colors["text"])
    fig.tight_layout()
    return fig


def plot_signal_cue_overlay(x_used, rpeaks, measurements, theme="Light", fs=DEFAULT_FS, lead_idx=1):
    colors = _fig_colors(theme)
    sig = np.asarray(x_used[lead_idx], dtype=float)
    t = np.arange(sig.shape[0]) / float(fs)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor(colors["fig"])
    ax.set_facecolor(colors["ax"])
    ax.plot(t, sig, linewidth=1.2, color=colors["accent"])
    ax.set_title("Lead II signal cues used for explanation", color=colors["text"], pad=10)
    ax.set_xlabel("Time (s)", color=colors["text"])
    ax.set_ylabel("Amplitude", color=colors["text"])
    ax.tick_params(axis="x", colors=colors["text"])
    ax.tick_params(axis="y", colors=colors["text"])
    ax.grid(True, alpha=0.20, color=colors["grid"])

    if rpeaks is not None and len(rpeaks) > 0:
        shown = rpeaks[: min(8, len(rpeaks))]
        ax.scatter(shown / float(fs), sig[shown], s=18, zorder=3)
        for rp in shown:
            ax.axvline(rp / float(fs), linestyle="--", linewidth=0.9, alpha=0.35)

        qrs_ms = measurements.get("qrs_ms_est")
        if qrs_ms is not None:
            half_w = int(max(1, round((qrs_ms / 1000.0) * fs / 2)))
            rp = int(shown[len(shown) // 2])
            left = max(0, rp - half_w)
            right = min(len(sig) - 1, rp + half_w)
            ax.axvspan(left / float(fs), right / float(fs), alpha=0.15)
            ax.text((left + right) / 2 / float(fs), ax.get_ylim()[1] * 0.85,
                    "Approx QRS window", ha="center", va="top", fontsize=9, color=colors["text"])

            st_idx = min(len(sig) - 1, rp + int(0.08 * fs))
            ax.scatter([st_idx / float(fs)], [sig[st_idx]], s=28, zorder=4)
            ax.annotate("Approx ST point",
                        xy=(st_idx / float(fs), sig[st_idx]),
                        xytext=(st_idx / float(fs) + 0.18, sig[st_idx]),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color=colors["text"]),
                        fontsize=9, color=colors["text"])

    for spine in ax.spines.values():
        spine.set_color(colors["text"])
    fig.tight_layout()
    return fig


# =========================================================
# Model
# =========================================================
@st.cache_resource
def get_model(model_path: Path):
    return load_model(str(model_path), device="cpu")


# =========================================================
# Sidebar
# =========================================================
def render_sidebar(primary_demo_dir: Path, secondary_demo_dir: Path, demo_manifest_path: Path, asset_message: Optional[str]) -> None:
    with st.sidebar:
        if asset_message:
            st.info(asset_message)

        st.markdown("### Theme")
        c1, c2 = st.columns(2)
        if c1.button("☀️", help="Light mode", **BTN_W):
            st.session_state["theme"] = "Light"
            st.rerun()
        if c2.button("🌙", help="Dark mode", **BTN_W):
            st.session_state["theme"] = "Dark"
            st.rerun()

        st.divider()
        st.markdown("### Settings")
        st.session_state["threshold"] = st.slider(
            "Decision threshold", 0.0, 1.0, float(st.session_state["threshold"]), 0.01
        )
        st.session_state["ecg_style"] = st.radio(
            "ECG plot style",
            ["Standard", "ECG paper"],
            index=0 if st.session_state["ecg_style"] == "Standard" else 1,
        )
        st.session_state["probability_view"] = st.radio(
            "Probability display",
            ["Table", "Bars", "Both"],
            index=["Table", "Bars", "Both"].index(st.session_state["probability_view"]),
        )
        st.session_state["apply_preprocess"] = st.checkbox(
            "Apply preprocessing (bandpass + notch + z-score)",
            value=bool(st.session_state["apply_preprocess"]),
        )
        st.session_state["remember_upload"] = st.checkbox(
            "Save new uploads for later",
            value=bool(st.session_state["remember_upload"]),
        )
        st.session_state["use_hf_assets"] = st.checkbox(
            "Try Hugging Face assets",
            value=bool(st.session_state["use_hf_assets"]),
        )

        st.caption("Turn ON preprocessing for raw PTB-XL exports. Leave OFF for already-preprocessed files.")

        st.divider()
        st.markdown("### Demo samples")

        demo_groups, missing = build_demo_index(primary_demo_dir, secondary_demo_dir, demo_manifest_path)
        group_names = sorted(demo_groups.keys(), key=lambda x: (x == "OTHER", x))

        if not group_names:
            st.info("No demos found. Put demo_*.npy into the demo/ folder.")
        else:
            chosen_group = st.selectbox("Demo group", group_names, index=0)
            entries = demo_groups.get(chosen_group, [])

            option_labels = []
            for e in entries:
                mark = "" if e["exists"] else " (missing)"
                option_labels.append(e["file"] + mark)

            picked = st.selectbox("Pick a demo file", option_labels, index=0 if option_labels else 0)
            picked_file = picked.replace(" (missing)", "")
            exists = (primary_demo_dir / picked_file).exists() or (secondary_demo_dir / picked_file).exists()

            if missing:
                st.caption("Some demos are listed but missing on disk.")

            if st.button("Load selected demo", disabled=not exists, **BTN_W):
                st.session_state["demo_file"] = picked_file
                st.session_state["saved_upload_name"] = None
                st.rerun()

            if st.button("Clear demo selection", **BTN_W):
                st.session_state["demo_file"] = None
                st.session_state["saved_upload_name"] = None
                st.rerun()

            if not exists:
                st.warning(f"Missing on disk: demo/{picked_file}")

        st.divider()
        st.markdown("### Saved uploads")

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

        st.divider()
        if st.button("Clear all selections", **BTN_W):
            st.session_state["demo_file"] = None
            st.session_state["saved_upload_name"] = None
            st.rerun()


# =========================================================
# Input resolution
# =========================================================
def resolve_input_source(primary_demo_dir: Path, secondary_demo_dir: Path, demo_manifest_path: Path):
    demo_manifest = load_demo_manifest(demo_manifest_path)
    demo_meta: dict = {}

    uploaded = st.file_uploader(
        "Upload ECG (.npy, .png, .jpg, .jpeg, .pdf)",
        type=["npy", "png", "jpg", "jpeg", "pdf"],
        key="main_ecg_uploader",
    )

    layout_label = st.selectbox(
        "ECG image layout",
        ["3x4 standard", "4x3 stacked"],
        index=0,
        key="image_layout_select",
    )
    layout = "3x4_standard" if layout_label == "3x4 standard" else "4x3_stacked"

    if uploaded is not None:
        uploaded_name = uploaded.name
        ext = Path(uploaded.name).suffix.lower()

        if ext == ".npy":
            x12_raw = load_uploaded_npy(uploaded)
        elif ext in IMAGE_EXTS:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded ECG image", use_container_width=True)
            x12_raw = digitize_ecg_image(image, layout=layout)
        elif ext in PDF_EXTS:
            x12_raw = digitize_ecg_pdf(uploaded, layout=layout)
        else:
            raise ValueError(f"Unsupported upload type: {ext}")

        x12_raw = validate_signal_array(x12_raw)

        if st.session_state.get("remember_upload", True):
            try:
                rec = persist_uploaded_file(uploaded, extra_meta={"layout": layout})
                st.session_state["saved_upload_name"] = rec["saved_name"]
            except Exception:
                pass

        st.session_state["demo_file"] = None
        return x12_raw, uploaded_name, demo_meta

    selected_demo = st.session_state.get("demo_file", None)
    if selected_demo:
        x12_raw = load_local_demo_npy(primary_demo_dir, secondary_demo_dir, selected_demo)
        uploaded_name = selected_demo
        demo_meta = demo_manifest.get(selected_demo, {})
        st.info(f"Using demo sample: {selected_demo}")
        if "recommended_apply_preprocess" in demo_meta:
            st.session_state["apply_preprocess"] = bool(demo_meta["recommended_apply_preprocess"])
        return x12_raw, uploaded_name, demo_meta

    selected_saved = st.session_state.get("saved_upload_name", None)
    if selected_saved:
        x12_raw = load_saved_input(selected_saved)
        uploaded_name = selected_saved
        st.info(f"Using saved upload: {selected_saved}")
        return x12_raw, uploaded_name, demo_meta

    st.info("Choose a demo, a saved upload, or upload a new ECG file.")
    st.stop()


# =========================================================
# Tab renderers
# =========================================================
def render_explanation_tab(theme, threshold, result, qc, qc_ok, quality_metrics, x12_raw, beats, rpeaks, x_used, apply_preprocess):
    st.subheader("Summary")
    st.write(build_quick_summary(result, threshold))

    st.subheader("Details")
    st.write(build_detailed_interpretation(result, threshold))

    if result.get("template_explanation"):
        with st.expander("Rule-based explanation (template)"):
            st.write(result["template_explanation"])

    st.subheader("Visual signal cues")
    st.pyplot(plot_signal_cue_overlay(x_used, rpeaks, result.get("measurements", {}), theme=theme))
    st.markdown(
        "<div class='small-note'>This overlay marks approximate signal landmarks used to make the explanation easier to follow. It is not a clinically validated abnormality locator.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Signal-based view")
    st.write("Most active leads in extracted beats:", ", ".join(result["top_active_leads"]))
    st.pyplot(plot_bar(result["lead_activity"], LEAD_NAMES, "Lead activity in extracted beats", "Normalized activity", theme=theme))
    st.markdown(
        "<div class='small-note'>Higher activity = larger average absolute signal in extracted beats, not diagnosis.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Explainability (XAI)")
    st.markdown("<div class='small-note'>These explain model behavior, not clinical causality.</div>", unsafe_allow_html=True)
    xai = result.get("xai", {})

    if xai.get("occlusion_delta_toplabel") is not None:
        st.write("**Lead occlusion sensitivity**")
        st.pyplot(
            plot_bar(
                xai["occlusion_delta_toplabel"],
                LEAD_NAMES,
                "Occlusion sensitivity vs lead",
                "Δ probability (base - occluded)",
                theme=theme,
            )
        )
        st.markdown(
            "<div class='small-note'>Bigger positive values mean the model depended more on that lead. Near-zero or negative values mean removing that lead changed the prediction little or made it slightly stronger.</div>",
            unsafe_allow_html=True,
        )

    if xai.get("ig_attr_toplabel") is not None:
        st.write(f"**Integrated Gradients** (per-lead attribution; steps={xai.get('ig_steps', 16)})")
        st.pyplot(
            plot_bar(
                xai["ig_attr_toplabel"],
                LEAD_NAMES,
                "Integrated Gradients attribution vs lead",
                "Attribution (normalized)",
                theme=theme,
            )
        )
        st.markdown(
            "<div class='small-note'>Larger attribution means the model assigned more influence to that lead for the top prediction. More IG steps usually gives smoother attribution but slower runtime.</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Quality checks")
    if qc_ok:
        st.success("QC passed (basic checks).")
    else:
        st.warning("QC reported issues:")
        for msg in qc.get("issues", []):
            st.write("-", msg)

    st.pyplot(plot_quality_metrics(quality_metrics, theme=theme))
    st.markdown(
        "<div class='small-note'>Quality metrics summarize finiteness, non-flatness, spread, and R-peak coverage.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Raw checks"):
        st.write("Uploaded ECG shape:", x12_raw.shape)
        st.write("Extracted beats shape:", beats.shape)
        st.write("Task type:", result.get("task_type", "N/A"))
        st.write("Applied preprocessing:", result.get("applied_preprocess", apply_preprocess))
        st.write("Detected R-peaks:", len(rpeaks) if rpeaks is not None else 0)
        st.write("Looks preprocessed / z-scored:", looks_preprocessed_zscored(x_used))


def render_education_tab(theme):
    st.subheader("Understand the ECG terms")
    st.caption("This panel is for non-experts. It shows the main ECG parts and typical normal timing.")

    st.pyplot(plot_ecg_education_figure(theme=theme))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**P wave**")
        st.write("Atrial activation — the atria begin the heartbeat.")
        st.markdown("**QRS complex**")
        st.write("Ventricular activation — the main pumping chambers contract.")
    with c2:
        st.markdown("**ST segment**")
        st.write("Early recovery phase after QRS; usually judged by level and shape.")
        st.markdown("**T wave**")
        st.write("Later ventricular recovery phase.")

    st.markdown("**Typical normal timing ranges**")
    timing_df = pd.DataFrame(
        {
            "Part": ["P duration", "PR interval", "QRS duration", "QT interval", "ST segment"],
            "Typical value": ["0.08–0.11 s", "0.12–0.20 s", "0.08–0.11 s", "0.35–0.44 s", "Varies"],
            "Note": ["Usually < 0.12 s", "120–200 ms", "Usually < 0.12 s", "Depends on HR", "Judged mainly by level/shape"],
        }
    )
    st.dataframe(timing_df, hide_index=True, **_stretch_kwargs(st.dataframe))

    st.markdown("**How to read seconds vs milliseconds**")
    st.markdown(
        "- **1 second = 1000 ms**\n"
        "- **0.12 s = 120 ms**\n"
        "- ECG intervals are often reported in **ms**"
    )

    st.markdown("**How a general reader can look at an ECG**")
    st.markdown(
        "- Look first at whether the beats are regular or irregular.\n"
        "- Then look at whether the QRS spikes are narrow or broad.\n"
        "- Next, look at whether the ST and T-wave parts return smoothly toward baseline.\n"
        "- Finally, remember that one unusual-looking beat alone is not enough for diagnosis."
    )

    st.markdown("**Important reminder**")
    st.markdown(
        "- A probability score is not the same as a medical diagnosis.\n"
        "- The app compares patterns to training examples; clinicians interpret ECGs with symptoms, history, and other tests.\n"
        "- Heart rate changes, motion, and noise can affect what the waveform looks like."
    )

    st.info("Educational support only. Real ECG interpretation depends on the full clinical context.")


# =========================================================
# Main app
# =========================================================
def main():
    init_session_state()
    apply_theme_css(st.session_state["theme"])

    model_path, primary_demo_dir, secondary_demo_dir, demo_manifest_path, asset_message = resolve_assets(
        use_hf_assets=bool(st.session_state["use_hf_assets"])
    )
    render_sidebar(primary_demo_dir, secondary_demo_dir, demo_manifest_path, asset_message)

    theme = st.session_state["theme"]
    threshold = float(st.session_state["threshold"])
    ecg_style = st.session_state["ecg_style"]
    probability_view = st.session_state["probability_view"]
    apply_preprocess = bool(st.session_state["apply_preprocess"])

    st.title("CardioAI – ECG Explorer")
    st.caption("Research demo only. Not for clinical use.")
    st.info("Image upload is experimental. Best results come from clean 12-lead ECG screenshots or PDF exports with standard layout.")
    st.markdown(
        "<div class='small-note'>Tip: dark mode is best viewed with Plotly installed so the interactive ECG chart also follows the app theme.</div>",
        unsafe_allow_html=True,
    )

    try:
        model = get_model(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    if getattr(model, "num_labels", 5) == 1:
        st.error(
            "Loaded checkpoint is binary (num_labels=1). "
            "This app expects the MULTILABEL checkpoint (num_labels=5) for NORM/MI/STTC/CD/HYP."
        )
        st.stop()

    try:
        x12_raw, uploaded_name, demo_meta = resolve_input_source(primary_demo_dir, secondary_demo_dir, demo_manifest_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

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
        result = predict_ecg(model, x12_raw, threshold=threshold, device="cpu")
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["ECG Viewer", "Prediction", "Explanation", "Education", "Review/Export"])

    with tab1:
        st.subheader("ECG used for inference")

        if ecg_style == "Standard" and HAVE_PLOTLY:
            fig = plot_12_lead_plotly(x_used, title="ECG used for inference", fs=DEFAULT_FS, theme=theme)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "<div class='small-note'>Interactive viewer with direct lead labeling and reduced clutter.</div>",
                unsafe_allow_html=True,
            )
        else:
            if ecg_style == "Standard" and not HAVE_PLOTLY:
                st.warning("Plotly is not installed; using static viewer.")
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
            st.markdown(
                "<div class='small-note'>One extracted beat token. The model aggregates multiple beats to decide.</div>",
                unsafe_allow_html=True,
            )

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

    with tab3:
        render_explanation_tab(
            theme, threshold, result, qc, qc_ok, quality_metrics,
            x12_raw, beats, rpeaks, x_used, apply_preprocess
        )

    with tab4:
        render_education_tab(theme)

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


if __name__ == "__main__":
    main()
