import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch, find_peaks


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
    if x12.shape != (12, 5000):
        raise ValueError(f"Expected shape (12, 5000), got {x12.shape}")

    x12 = bandpass_filter(x12)
    x12 = notch_filter(x12)
    x12 = zscore_per_lead(x12)
    x12 = np.nan_to_num(x12).astype(np.float32)
    return x12


# =========================
# Input loading
# =========================
def load_uploaded_npy(uploaded_file):
    data = np.load(io.BytesIO(uploaded_file.getvalue()), allow_pickle=True)
    data = np.asarray(data, dtype=np.float32)
    if data.shape != (12, 5000):
        raise ValueError(f"Expected uploaded .npy to have shape (12, 5000), got {data.shape}")
    return data


# =========================
# QC / validation "agent"
# =========================
def validate_ecg(x12: np.ndarray) -> QCReport:
    issues = []
    x12 = np.asarray(x12, dtype=np.float32)

    if x12.shape != (12, 5000):
        issues.append(f"Wrong shape: {x12.shape}, expected (12, 5000).")
        return QCReport(ok=False, issues=issues, per_lead_std=[], per_lead_ptp=[])

    if not np.isfinite(x12).all():
        issues.append("Non-finite values found (NaN/Inf).")

    per_std = [float(np.std(x12[i])) for i in range(12)]
    per_ptp = [float(np.ptp(x12[i])) for i in range(12)]

    # Very simple QC thresholds (demo-level)
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
    """
    Rough QRS duration estimate using Lead II beat windows:
    width of samples above 50% of peak abs amplitude around the R region.
    """
    lead = 1  # Lead II
    widths = []
    for b in range(beats.shape[0]):
        sig = beats[b, lead]
        center = PRE_R
        w = sig[max(0, center-150):min(len(sig), center+200)]
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
    """
    Rough ST deviation estimate in Lead II:
    baseline = mean around [-200ms, -100ms] before R
    J point approx = +60ms after R
    Returns in arbitrary units (since signals are z-scored if preprocessing applied).
    Still useful for relative explanation.
    """
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
    logits = model(beats_tensor).squeeze(0)  # (L,)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs


@torch.no_grad()
def lead_occlusion_sensitivity(
    model: nn.Module,
    beats_tensor: torch.Tensor,
    target_label_index: int
) -> np.ndarray:
    """
    For each lead, zero that lead across all beats and compute drop in target prob.
    Returns array of length 12: delta = base_prob - occluded_prob (bigger => more important).
    """
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
    """
    Manual Integrated Gradients on beats input (no extra libs).
    Returns per-lead attribution (sum abs over beats/time).
    """
    model.eval()

    x = beats_tensor.detach().clone()
    baseline = torch.zeros_like(x)

    total_grad = torch.zeros_like(x)

    for i in range(1, steps + 1):
        alpha = i / steps
        xi = baseline + alpha * (x - baseline)
        xi.requires_grad_(True)

        logits = model(xi).squeeze(0)  # (L,)
        target = logits[target_label_index]
        model.zero_grad(set_to_none=True)
        target.backward()

        grad = xi.grad.detach()
        total_grad += grad

    avg_grad = total_grad / steps
    ig = (x - baseline) * avg_grad  # attribution tensor

    # aggregate to per-lead score
    ig_abs = ig.abs().detach().cpu().numpy()  # (1,B,12,T)
    per_lead = np.sum(ig_abs, axis=(0, 1, 3))  # (12,)
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
    """
    Template-based medical-style explanation.
    Uses cautious language: "may be consistent with" and "decision-support only".
    """
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

    # label-specific language (still cautious)
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
# Main pipeline (front-end orchestration)
# =========================
def prepare_input(
    x12_raw: np.ndarray,
    apply_preprocess: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_used: (12,5000) after optional preprocessing
      rpeaks: (N,)
      beats:  (B,12,1250)
    """
    x12_raw = np.asarray(x12_raw, dtype=np.float32)
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
    """
    Full inference + QC + measurements + optional XAI.
    """
    qc = validate_ecg(x12_raw)

    x_used, rpeaks, beats = prepare_input(x12_raw, apply_preprocess=apply_preprocess)
    xb = torch.from_numpy(beats).unsqueeze(0).to(device)  # (1,B,12,1250)

    probs = model_probs_from_beats(model, xb)
    num_labels = int(model.num_labels)

    # multilabel
    class_names = CLASS_NAMES[:num_labels]
    probs_dict = {class_names[i]: float(probs[i]) for i in range(num_labels)}
    preds_dict = {k: int(v >= threshold) for k, v in probs_dict.items()}
    positive_labels = [k for k, v in preds_dict.items() if v == 1]
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx]
    top_prob = float(probs[top_idx])

    # uncertainty heuristic
    sorted_probs = sorted(probs, reverse=True)
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = float(top_prob - second)
    uncertainty_flag = (top_prob < 0.60) or (margin < 0.15)

    meas = compute_measurements(beats, rpeaks)

    # Lead activity always (cheap)
    lead_activity = lead_activity_from_beats(beats)
    top3_leads_idx = np.argsort(lead_activity)[::-1][:3]
    top3_leads = [LEAD_NAMES[i] for i in top3_leads_idx]

    # XAI: occlusion + IG for TOP label
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