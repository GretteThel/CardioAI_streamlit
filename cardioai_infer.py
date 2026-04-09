import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

try:
    from PIL import Image, ImageOps
    HAVE_PIL = True
except Exception:
    Image = None
    ImageOps = None
    HAVE_PIL = False

try:
    import fitz  # PyMuPDF
    HAVE_FITZ = True
except Exception:
    fitz = None
    HAVE_FITZ = False


# =========================
# Constants
# =========================
FS = 500
SEG_LEN = 10 * FS          # 5000
B = 12
BEAT_LEN = 1250
PRE_R = 500                # 500 before R, 750 after

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTS = {".pdf"}
SIGNAL_EXTS = {".npy"}


# =========================
# "Agent-style" modules (deterministic, not LLM)
# =========================
@dataclass
class QCReport:
    ok: bool
    issues: List[str]
    per_lead_std: List[float]
    per_lead_ptp: List[float]


@dataclass
class Measurements:
    heart_rate_bpm: Optional[float]
    rr_std_ms: Optional[float]
    qrs_ms_est: Optional[float]
    st_dev_mv_est: Optional[float]
    rpeaks_count: int


# =========================
# Generic input helpers
# =========================
def resample_1d(sig: np.ndarray, target_len: int) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    if len(sig) == target_len:
        return sig.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, len(sig), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
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
        x = np.vstack([resample_1d(lead, 5000) for lead in x]).astype(np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x


def load_uploaded_npy(uploaded_file):
    data = np.load(io.BytesIO(uploaded_file.getvalue()), allow_pickle=True)
    data = np.asarray(data, dtype=np.float32)
    return validate_signal_array(data)


def load_npy_bytes(raw_bytes: bytes) -> np.ndarray:
    data = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
    data = np.asarray(data, dtype=np.float32)
    return validate_signal_array(data)


# =========================
# ECG image digitization
# =========================
def crop_border(arr: np.ndarray, frac: float = 0.03) -> np.ndarray:
    h, w = arr.shape[:2]
    y0, y1 = int(h * frac), int(h * (1.0 - frac))
    x0, x1 = int(w * frac), int(w * (1.0 - frac))
    if y1 <= y0 or x1 <= x0:
        return arr
    return arr[y0:y1, x0:x1]


def split_ecg_grid(gray: np.ndarray, layout: str = "3x4_standard") -> List[np.ndarray]:
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


def reorder_panels_to_standard_12lead(panels: List[np.ndarray], layout: str) -> List[np.ndarray]:
    if len(panels) != 12:
        raise ValueError(f"Expected 12 lead panels, got {len(panels)}")

    if layout == "4x3_stacked":
        # assumes row-major:
        # [I, II, III,
        #  aVR, aVL, aVF,
        #  V1, V2, V3,
        #  V4, V5, V6]
        return panels

    if layout == "3x4_standard":
        # assumes row-major:
        # [I, aVR, V1, V4,
        #  II, aVL, V2, V5,
        #  III, aVF, V3, V6]
        idx = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        return [panels[i] for i in idx]

    raise ValueError(f"Unknown layout: {layout}")


def trace_panel_to_signal(panel: np.ndarray) -> np.ndarray:
    arr = panel.astype(np.float32)

    # invert so dark trace becomes large
    arr = 255.0 - arr
    arr = arr / max(float(arr.max()), 1.0)

    h, w = arr.shape
    y_trace = np.zeros(w, dtype=np.float32)

    for x in range(w):
        col = arr[:, x]
        if float(col.max()) < 0.08:
            y_trace[x] = h / 2.0
        else:
            y_trace[x] = float(np.argmax(col))

    kernel = np.ones(9, dtype=np.float32) / 9.0
    y_trace = np.convolve(y_trace, kernel, mode="same")

    # convert pixel row -> centered amplitude
    sig = -(y_trace - (h / 2.0)) / max(h / 2.0, 1.0)
    return sig.astype(np.float32)


def digitize_ecg_image(image: "Image.Image", layout: str = "3x4_standard") -> np.ndarray:
    if not HAVE_PIL:
        raise ImportError("Image upload needs Pillow installed.")

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


def digitize_ecg_pdf_bytes(raw_bytes: bytes, layout: str = "3x4_standard") -> np.ndarray:
    if not HAVE_FITZ:
        raise ImportError("PDF upload needs PyMuPDF installed.")

    if not HAVE_PIL:
        raise ImportError("PDF upload also needs Pillow installed.")

    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200, alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return digitize_ecg_image(image, layout=layout)


def load_uploaded_input(uploaded_file, layout: str = "3x4_standard") -> np.ndarray:
    ext = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    if ext in SIGNAL_EXTS:
        return load_npy_bytes(raw_bytes)

    if ext in IMAGE_EXTS:
        if not HAVE_PIL:
            raise ImportError("Image upload needs Pillow installed.")
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return digitize_ecg_image(image, layout=layout)

    if ext in PDF_EXTS:
        return digitize_ecg_pdf_bytes(raw_bytes, layout=layout)

    raise ValueError(f"Unsupported upload type: {ext}")


def load_input_from_path(path: Union[str, Path], layout: str = "3x4_standard") -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in SIGNAL_EXTS:
        x = np.load(str(path), allow_pickle=True)
        return validate_signal_array(x)

    if ext in IMAGE_EXTS:
        if not HAVE_PIL:
            raise ImportError("Image loading needs Pillow installed.")
        image = Image.open(path).convert("RGB")
        return digitize_ecg_image(image, layout=layout)

    if ext in PDF_EXTS:
        return digitize_ecg_pdf_bytes(path.read_bytes(), layout=layout)

    raise ValueError(f"Unsupported saved file type: {ext}")


# =========================
# Preprocessing
# =========================
def bandpass_filter(x, fs=FS, low=0.5, high=40.0, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def notch_filter(x, fs=FS, f0=50.0, q=30.0):
    b, a = iirnotch(f0 / (fs / 2), q)
    return filtfilt(b, a, x, axis=-1)


def zscore_per_lead(x, eps=1e-8):
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True) + eps
    return (x - mu) / sd


def preprocess_ecg(x12: np.ndarray) -> np.ndarray:
    """
    x12: (12, 5000)
    """
    x12 = np.asarray(x12, dtype=np.float32)
    x12 = validate_signal_array(x12)
    x12 = bandpass_filter(x12)
    x12 = notch_filter(x12)
    x12 = zscore_per_lead(x12)
    x12 = np.nan_to_num(x12).astype(np.float32)
    return x12


# =========================
# QC / validation "agent"
# =========================
def validate_ecg(x12: np.ndarray) -> QCReport:
    issues = []
    x12 = np.asarray(x12, dtype=np.float32)

    if x12.ndim != 2:
        issues.append(f"Wrong ndim: {x12.ndim}, expected 2D array.")
        return QCReport(ok=False, issues=issues, per_lead_std=[], per_lead_ptp=[])

    if x12.shape[0] != 12 and x12.shape[1] == 12:
        x12 = x12.T

    if x12.shape != (12, 5000):
        issues.append(f"Wrong shape: {x12.shape}, expected (12, 5000).")
        return QCReport(ok=False, issues=issues, per_lead_std=[], per_lead_ptp=[])

    if not np.isfinite(x12).all():
        issues.append("Non-finite values found (NaN/Inf).")

    per_std = [float(np.std(x12[i])) for i in range(12)]
    per_ptp = [float(np.ptp(x12[i])) for i in range(12)]

    if np.count_nonzero(x12) < 100:
        issues.append("Signal appears mostly zero/flatline.")
    if np.mean(per_std) < 1e-4:
        issues.append("Very low variance signal (likely flat).")
    if any(s < 1e-5 for s in per_std):
        issues.append("Some leads have extremely low variance (flat).")

    ok = len(issues) == 0
    return QCReport(ok=ok, issues=issues, per_lead_std=per_std, per_lead_ptp=per_ptp)


# =========================
# R-peaks + beat extraction
# =========================
def detect_rpeaks_lead2(x12, fs=FS):
    """
    Lead-II heuristic: slope-energy envelope + peak finding.
    Works reasonably for demo; not clinical-grade.
    """
    lead2 = x12[1]
    d = np.diff(lead2, prepend=lead2[0])
    e = d ** 2

    win = max(1, int(0.150 * fs))
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.convolve(e, kernel, mode="same")

    peaks, _ = find_peaks(env, distance=int(0.25 * fs))
    return peaks.astype(int)


def extract_fixed_beats(x12, rpeaks, B=B, beat_len=BEAT_LEN, pre_r=PRE_R):
    post_r = beat_len - pre_r
    beats = np.zeros((B, 12, beat_len), dtype=np.float32)

    if len(rpeaks) == 0:
        centers = np.linspace(pre_r, SEG_LEN - post_r - 1, B).astype(int)
    elif len(rpeaks) >= B:
        idx = np.linspace(0, len(rpeaks) - 1, B).round().astype(int)
        centers = rpeaks[idx]
    else:
        centers = np.pad(rpeaks, (0, B - len(rpeaks)), mode="edge")

    for i, c in enumerate(centers[:B]):
        s = c - pre_r
        e = c + post_r

        src_s = max(0, s)
        src_e = min(SEG_LEN, e)

        dst_s = src_s - s
        dst_e = dst_s + (src_e - src_s)

        beats[i, :, dst_s:dst_e] = x12[:, src_s:src_e]

    return beats


# =========================
# Model definition
# =========================
class CNN1DBeatEncoder(nn.Module):
    def __init__(self, d_model=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model),
        )

    def forward(self, x):
        return self.proj(self.net(x))


class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class HybridBeatsTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, n_layers=2, dropout=0.3, num_labels=5):
        super().__init__()
        self.num_labels = num_labels
        self.enc = CNN1DBeatEncoder(d_model=d_model, dropout=dropout)
        self.pos = PosEnc(d_model=d_model, max_len=64)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, x):
        bs, B_, C, T = x.shape
        x = x.view(bs * B_, C, T)
        z = self.enc(x).view(bs, B_, -1)
        z = self.tr(self.pos(z)).mean(dim=1)
        z = self.drop(self.norm(z))
        return self.head(z)


# =========================
# Checkpoint helpers
# =========================
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state", "model_state_dict", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    return ckpt


def _strip_prefixes(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned[nk] = v
    return cleaned


def _infer_num_labels(state_dict):
    if "head.weight" not in state_dict:
        raise KeyError("Could not find 'head.weight' in checkpoint state_dict.")
    return int(state_dict["head.weight"].shape[0])


def load_model(ckpt_path="models/best_hybrid_final.pt", device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = _strip_prefixes(_extract_state_dict(ckpt))
    num_labels = _infer_num_labels(sd)

    model = HybridBeatsTransformer(num_labels=num_labels)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


# =========================
# Measurement "agent" (demo-level, approximate)
# =========================
def estimate_hr_rr(rpeaks: np.ndarray, fs=FS) -> Tuple[Optional[float], Optional[float]]:
    if len(rpeaks) < 2:
        return None, None
    rr = np.diff(rpeaks) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) == 0:
        return None, None
    hr = 60.0 / float(np.mean(rr))
    rr_std_ms = float(np.std(rr) * 1000.0)
    return hr, rr_std_ms


def estimate_qrs_ms_from_beats(beats: np.ndarray, fs=FS) -> Optional[float]:
    lead = 1  # Lead II
    widths = []
    for b in range(beats.shape[0]):
        sig = beats[b, lead]
        center = PRE_R
        w = sig[max(0, center - 150):min(len(sig), center + 200)]
        amp = np.max(np.abs(w))
        if amp < 1e-6:
            continue
        thr = 0.5 * amp
        idx = np.where(np.abs(w) >= thr)[0]
        if len(idx) < 2:
            continue
        width_samp = idx[-1] - idx[0]
        widths.append(width_samp)
    if not widths:
        return None
    return float(np.median(widths) / fs * 1000.0)


def estimate_st_deviation(beats: np.ndarray, fs=FS) -> Optional[float]:
    lead = 1
    st_vals = []
    for b in range(beats.shape[0]):
        sig = beats[b, lead]
        r = PRE_R
        b0 = int(r - 0.20 * fs)
        b1 = int(r - 0.10 * fs)
        j = int(r + 0.06 * fs)
        if b0 < 0 or j >= len(sig):
            continue
        baseline = float(np.mean(sig[b0:b1]))
        st = float(sig[j] - baseline)
        st_vals.append(st)
    if not st_vals:
        return None
    return float(np.median(st_vals))


def compute_measurements(beats: np.ndarray, rpeaks: np.ndarray) -> Measurements:
    hr, rr_std = estimate_hr_rr(rpeaks)
    qrs_ms = estimate_qrs_ms_from_beats(beats)
    st_dev = estimate_st_deviation(beats)
    return Measurements(
        heart_rate_bpm=hr,
        rr_std_ms=rr_std,
        qrs_ms_est=qrs_ms,
        st_dev_mv_est=st_dev,
        rpeaks_count=int(len(rpeaks)),
    )


# =========================
# XAI: Lead activity + occlusion + Integrated Gradients
# =========================
def lead_activity_from_beats(beats: np.ndarray) -> np.ndarray:
    lead_energy = np.mean(np.abs(beats), axis=(0, 2))
    s = float(np.sum(lead_energy))
    if s > 0:
        lead_energy = lead_energy / s
    return lead_energy.astype(np.float32)


@torch.no_grad()
def model_probs_from_beats(model: nn.Module, beats_tensor: torch.Tensor) -> np.ndarray:
    logits = model(beats_tensor).squeeze(0)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs


@torch.no_grad()
def lead_occlusion_sensitivity(
    model: nn.Module,
    beats_tensor: torch.Tensor,
    target_label_index: int
) -> np.ndarray:
    base_probs = model_probs_from_beats(model, beats_tensor)
    base = float(base_probs[target_label_index])

    deltas = np.zeros(12, dtype=np.float32)
    for lead in range(12):
        x_occ = beats_tensor.clone()
        x_occ[:, :, lead, :] = 0.0
        p_occ = float(model_probs_from_beats(model, x_occ)[target_label_index])
        deltas[lead] = base - p_occ
    return deltas


def integrated_gradients_beats(
    model: nn.Module,
    beats_tensor: torch.Tensor,
    target_label_index: int,
    steps: int = 16
) -> np.ndarray:
    model.eval()

    x = beats_tensor.detach().clone()
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)

    for i in range(1, steps + 1):
        alpha = i / steps
        xi = baseline + alpha * (x - baseline)
        xi.requires_grad_(True)

        logits = model(xi).squeeze(0)
        target = logits[target_label_index]
        model.zero_grad(set_to_none=True)
        target.backward()

        grad = xi.grad.detach()
        total_grad += grad

    avg_grad = total_grad / steps
    ig = (x - baseline) * avg_grad

    ig_abs = ig.abs().detach().cpu().numpy()
    per_lead = np.sum(ig_abs, axis=(0, 1, 3))
    per_lead = per_lead / (np.sum(per_lead) + 1e-8)
    return per_lead.astype(np.float32)


# =========================
# Explanation templates (rule-based)
# =========================
def template_explanation(
    top_label: str,
    top_prob: float,
    mi_prob: Optional[float],
    positive_labels: List[str],
    measurements: Measurements,
    uncertainty_flag: bool
) -> str:
    hr = measurements.heart_rate_bpm
    qrs = measurements.qrs_ms_est
    st = measurements.st_dev_mv_est

    hr_txt = f"{hr:.1f} bpm" if hr is not None else "unavailable"
    qrs_txt = f"{qrs:.0f} ms" if qrs is not None else "unavailable"
    st_txt = f"{st:.3f} (z-units)" if st is not None else "unavailable"

    flags_txt = ", ".join(positive_labels) if positive_labels else "none"

    caution = ""
    if uncertainty_flag:
        caution = (
            "\n\n**Uncertainty note:** The top two labels are relatively close or probabilities are moderate. "
            "Treat this as lower-confidence decision-support."
        )

    if top_label == "NORM":
        label_txt = (
            "The model’s highest score is **NORM**, which may indicate patterns closer to normal recordings "
            "in the training distribution."
        )
    elif top_label == "MI":
        label_txt = (
            "The model’s highest score is **MI**. This may reflect patterns seen in myocardial infarction cases "
            "in the training set, but it is **not a diagnosis**."
        )
    elif top_label == "STTC":
        label_txt = (
            "The model’s highest score is **STTC** (ST/T changes). This can be associated with multiple conditions "
            "(including ischemia or non-ischemic causes) and should be clinically verified."
        )
    elif top_label == "CD":
        label_txt = (
            "The model’s highest score is **CD** (conduction disturbance). This may align with broader QRS-related "
            "morphology differences; QRS estimate is provided below (approx)."
        )
    elif top_label == "HYP":
        label_txt = (
            "The model’s highest score is **HYP** (hypertrophy-related patterns). This may relate to voltage/morphology "
            "differences often seen in hypertrophy-like cases in the training data."
        )
    else:
        label_txt = f"The model’s highest score is **{top_label}**."

    mi_txt = ""
    if mi_prob is not None:
        mi_txt = f"**MI probability:** {mi_prob:.3f}."

    text = (
        f"{label_txt}\n\n"
        f"**Model confidence:** {top_prob:.3f}.\n\n"
        f"**Labels above threshold:** {flags_txt}.\n\n"
        f"{mi_txt}\n\n"
        f"**Signal summary (approx):** HR {hr_txt}, QRS {qrs_txt}, ST deviation {st_txt}.\n\n"
        f"_This explanation combines model outputs and simple signal measurements. It is decision-support only, "
        f"not a clinical diagnosis._"
        f"{caution}"
    )
    return text


# =========================
# Main pipeline
# =========================
def prepare_input(
    x12_raw: np.ndarray,
    apply_preprocess: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x12_raw = validate_signal_array(x12_raw)
    x_used = preprocess_ecg(x12_raw) if apply_preprocess else x12_raw
    rpeaks = detect_rpeaks_lead2(x_used)
    beats = extract_fixed_beats(x_used, rpeaks)
    return x_used, rpeaks, beats


def predict_ecg(
    model: nn.Module,
    x12_raw: np.ndarray,
    threshold: float = 0.5,
    device: str = "cpu",
    apply_preprocess: bool = False,
    run_occlusion: bool = True,
    run_ig: bool = False,
    ig_steps: int = 16,
) -> Dict:
    x12_raw = validate_signal_array(x12_raw)
    qc = validate_ecg(x12_raw)

    x_used, rpeaks, beats = prepare_input(x12_raw, apply_preprocess=apply_preprocess)
    xb = torch.from_numpy(beats).unsqueeze(0).to(device)

    probs = model_probs_from_beats(model, xb)
    num_labels = int(model.num_labels)

    class_names = CLASS_NAMES[:num_labels]
    probs_dict = {class_names[i]: float(probs[i]) for i in range(num_labels)}
    preds_dict = {k: int(v >= threshold) for k, v in probs_dict.items()}
    positive_labels = [k for k, v in preds_dict.items() if v == 1]
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx]
    top_prob = float(probs[top_idx])

    sorted_probs = sorted(probs, reverse=True)
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = float(top_prob - second)
    uncertainty_flag = (top_prob < 0.60) or (margin < 0.15)

    meas = compute_measurements(beats, rpeaks)

    lead_activity = lead_activity_from_beats(beats)
    top3_leads_idx = np.argsort(lead_activity)[::-1][:3]
    top3_leads = [LEAD_NAMES[i] for i in top3_leads_idx]

    occlusion = None
    ig_attr = None
    if run_occlusion:
        occlusion = lead_occlusion_sensitivity(model, xb, target_label_index=top_idx).tolist()

    if run_ig:
        ig_attr = integrated_gradients_beats(model, xb, target_label_index=top_idx, steps=ig_steps).tolist()

    explanation_text = template_explanation(
        top_label=top_label,
        top_prob=top_prob,
        mi_prob=probs_dict.get("MI", None),
        positive_labels=positive_labels,
        measurements=meas,
        uncertainty_flag=uncertainty_flag
    )

    return {
        "task_type": "multilabel",
        "probs": probs_dict,
        "preds": preds_dict,
        "positive_labels": positive_labels,
        "top_label": top_label,
        "top_prob": top_prob,
        "margin": margin,
        "mi_prob": probs_dict.get("MI", None),
        "uncertainty_flag": uncertainty_flag,

        "rpeaks": rpeaks,
        "beats": beats,
        "x_used": x_used,
        "applied_preprocess": bool(apply_preprocess),

        "qc": {
            "ok": qc.ok,
            "issues": qc.issues,
            "per_lead_std": qc.per_lead_std,
            "per_lead_ptp": qc.per_lead_ptp,
        },
        "measurements": {
            "heart_rate_bpm": meas.heart_rate_bpm,
            "rr_std_ms": meas.rr_std_ms,
            "qrs_ms_est": meas.qrs_ms_est,
            "st_dev_est": meas.st_dev_mv_est,
            "rpeaks_count": meas.rpeaks_count,
        },
        "lead_activity": lead_activity.tolist(),
        "top_active_leads": top3_leads,

        "xai": {
            "occlusion_delta_toplabel": occlusion,
            "ig_attr_toplabel": ig_attr,
            "ig_steps": ig_steps,
        },

        "template_explanation": explanation_text,
    }
