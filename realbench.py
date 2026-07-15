"""Zero-shot evaluation of trained in-context SysID models on real measured
plants from nonlinearbenchmark.org: Silverbox, Cascaded_Tanks, and
WienerHammerBenchMark via the `nonlinear_benchmarks` package (imported
lazily, inside functions, so this module only requires it when actually
run); Bouc-Wen via a direct fetch from its official source (4TU.ResearchData,
Noel & Schoukens 2020, CC BY-SA 4.0 -- see _boucwen_fetch), since it is not
exposed by `nonlinear_benchmarks`'s public API.

    python -m plantforge.realbench
"""
from __future__ import annotations

import pathlib

import numpy as np
import scipy.signal
import torch

from .corpus import RATES
from .evaluate import InContextSysID, _norm, T_CTX, D, DEV, CKPT_DIR

WINDOW_CAP = 8

BOUCWEN_URL = "https://data.4tu.nl/ndownloader/items/7060f9bc-8289-411e-8d32-57bef2740d32/versions/1"
BOUCWEN_CACHE = pathlib.Path.home() / ".nonlinear_benchmarks" / "BoucWen"
BOUCWEN_DT = 1.0 / 750.0   # native sampling rate, 750 Hz exactly (benchmark spec PDF)


def decimate_to_factor(x: np.ndarray, q: int) -> np.ndarray:
    """Anti-aliased decimation by integer factor q. Chains calls in steps of
    at most 10 when q factors evenly into such steps (scipy.signal.decimate
    recommends q<=13 per call); when q (or what remains of it) has no
    divisor <=10 other than 1 -- e.g. a prime like 13, 17, 23 -- decimates
    by the remaining factor in a single call instead of looping forever
    looking for a clean split that doesn't exist."""
    factors = []
    remaining = q
    while remaining > 1:
        step = None
        for f in range(min(remaining, 10), 1, -1):
            if remaining % f == 0:
                step = f
                break
        if step is None:
            step = remaining
        factors.append(step)
        remaining //= step
    out = x
    for f in factors:
        out = scipy.signal.decimate(out, f, ftype="iir", zero_phase=True)
    return out


def best_decimation_factor(native_dt: float, target_dts=RATES):
    """Integer decimation factor q>=1 that brings native_dt down to the
    FINEST (smallest dt) trained rate reachable by decimating. Finer
    retains more of the real signal's bandwidth than coarser, so among
    reachable targets the finest is preferred outright -- picking by raw
    abs(dt) error instead favors coarse targets by round-off coincidence
    for some native rates (e.g. native_dt=0.0016384s lands closer in raw
    error to the 0.10s target than to the 0.02s target, despite 0.02s
    being reachable and far more informative). Returns None if native_dt
    is already coarser than every target (nothing to decimate to -- the
    Cascaded_Tanks case).

    Edge case (not hit by any dataset used in this project): if native_dt
    is coarser than the finest target but still finer than the coarsest
    (e.g. native_dt=0.06 against target_dts=[0.10, 0.05, 0.02]),
    round(finest / native_dt) can round to 0, which is clamped to q=1 --
    i.e. "no decimation applied" -- even though the achieved_dt returned
    (== native_dt) doesn't actually land near any target in target_dts.
    Callers passing a native_dt in that middle range should not rely on
    this function's "reachable target" guarantee."""
    if native_dt >= max(target_dts):
        return None
    finest = min(target_dts)
    q = max(1, round(finest / native_dt))
    achieved = native_dt * q
    return q, achieved


def make_windows(u: np.ndarray, y: np.ndarray, cap: int = WINDOW_CAP):
    """Slice 1-D (u, y) arrays into non-overlapping length-D windows, stacked
    as (B, D) float32 tensors on DEV. Returns None if fewer than 1 full
    window is available."""
    n_windows = min(len(u), len(y)) // D
    if n_windows < 1:
        return None
    n_windows = min(n_windows, cap)
    u_win = np.stack([u[i * D:(i + 1) * D] for i in range(n_windows)])
    y_win = np.stack([y[i * D:(i + 1) * D] for i in range(n_windows)])
    return (torch.tensor(u_win, dtype=torch.float32, device=DEV),
            torch.tensor(y_win, dtype=torch.float32, device=DEV))


def pooled_windows(records, cap: int = WINDOW_CAP):
    """Slice windows from each (u, y) record and concatenate across records,
    stopping once `cap` total windows are collected."""
    u_all, y_all = [], []
    remaining = cap
    for u, y in records:
        if remaining <= 0:
            break
        w = make_windows(u, y, cap=remaining)
        if w is None:
            continue
        u_all.append(w[0]); y_all.append(w[1])
        remaining -= w[0].shape[0]
    if not u_all:
        return None
    return torch.cat(u_all, dim=0), torch.cat(y_all, dim=0)


def nmse_on_windows(model, u: torch.Tensor, y: torch.Tensor) -> float:
    """Same nMSE formula as evaluate.nmse: query-horizon MSE normalized by
    query-horizon variance, computed over a fixed batch of (u, y) windows."""
    u_n, y_n = _norm(u, y)
    with torch.no_grad():
        pred = model(u_n, y_n)
    return (((pred[:, T_CTX:] - y_n[:, T_CTX:]) ** 2).mean()
            / (y_n[:, T_CTX:] ** 2).mean()).item()


def load_model(ckpt_name: str):
    """Load a trained InContextSysID checkpoint in eval mode, or None if the
    checkpoint file doesn't exist (e.g. only one of corpus/wh_only has been
    trained so far)."""
    ck_path = CKPT_DIR / ckpt_name
    if not ck_path.exists():
        return None
    model = InContextSysID().to(DEV)
    ck = torch.load(ck_path, map_location=DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def silverbox_windows():
    """Load Silverbox's test records, decimate to the nearest trained rate,
    and pool windows across all test records. Returns
    (windows_or_None, achieved_dt, decimation_factor)."""
    import nonlinear_benchmarks as nb
    _, test = nb.Silverbox()
    native_dt = test[0].sampling_time
    result = best_decimation_factor(native_dt, RATES)
    if result is None:
        raise RuntimeError(
            f"Silverbox native_dt={native_dt} is already coarser than every "
            f"trained rate {RATES} -- cannot decimate down to any of them"
        )
    q, achieved_dt = result
    records = []
    for rec in test:
        u = decimate_to_factor(np.asarray(rec.u, dtype=np.float64), q)
        y = decimate_to_factor(np.asarray(rec.y, dtype=np.float64), q)
        records.append((u, y))
    return pooled_windows(records), achieved_dt, q


def cascaded_tanks_windows():
    """Load Cascaded_Tanks' test record at its native rate (already coarser
    than every trained rate, so no decimation is applied). Returns
    (windows_or_None, native_dt)."""
    import nonlinear_benchmarks as nb
    _, test = nb.Cascaded_Tanks()
    native_dt = test.sampling_time
    u = np.asarray(test.u, dtype=np.float64)
    y = np.asarray(test.y, dtype=np.float64)
    return make_windows(u, y), native_dt


def wienerhammer_status():
    """WienerHammerBenchMark's total physical duration is shorter than one
    context window at any trained rate -- report why it's skipped rather
    than evaluate it. Returns (duration_seconds, min_seconds_needed)."""
    import nonlinear_benchmarks as nb
    _, test = nb.WienerHammerBenchMark()
    native_dt = test.sampling_time
    duration = len(test.u) * native_dt
    min_needed = D * min(RATES)
    return duration, min_needed


def _boucwen_fetch() -> pathlib.Path:
    """Download+cache the official Bouc-Wen test signals (CC BY-SA 4.0,
    Noel & Schoukens 2020, data.4tu.nl DOI 10.4121/12967592) if not already
    cached. Returns the directory containing the four .mat test-signal
    files. Uses only the standard library (urllib, zipfile) -- no new
    dependency for a one-off fetch of a single small dataset."""
    inner = BOUCWEN_CACHE / "BoucWenFiles" / "Test signals" / "Validation signals"
    if (inner / "uval_multisine.mat").exists():
        return inner
    import io
    import urllib.request
    import zipfile
    print("dataset not found, downloading Bouc-Wen from data.4tu.nl ...")
    BOUCWEN_CACHE.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BOUCWEN_URL) as resp:
        outer_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer_zip:
        outer_zip.extract("BoucWenFiles.zip", BOUCWEN_CACHE)
    inner_zip_path = BOUCWEN_CACHE / "BoucWenFiles.zip"
    with zipfile.ZipFile(inner_zip_path) as inner_zip:
        # inner_zip's own members are already prefixed with "BoucWenFiles/",
        # so extract into BOUCWEN_CACHE directly -- extracting into
        # BOUCWEN_CACHE / "BoucWenFiles" would double that prefix and produce
        # .../BoucWenFiles/BoucWenFiles/Test signals/... instead of `inner`.
        inner_zip.extractall(BOUCWEN_CACHE)
    return inner


def boucwen_windows():
    """Load Bouc-Wen's multisine + sinesweep test records (noiseless,
    750 Hz native), decimate to the nearest trained rate, and pool windows
    across both. Returns (windows_or_None, achieved_dt, decimation_factor)."""
    import scipy.io as sio
    d = _boucwen_fetch()
    q, achieved_dt = best_decimation_factor(BOUCWEN_DT, RATES)
    records = []
    for name in ("multisine", "sinesweep"):
        u = sio.loadmat(d / f"uval_{name}.mat")[f"uval_{name}"].ravel().astype(np.float64)
        y = sio.loadmat(d / f"yval_{name}.mat")[f"yval_{name}"].ravel().astype(np.float64)
        u = decimate_to_factor(u, q)
        y = decimate_to_factor(y, q)
        records.append((u, y))
    return pooled_windows(records), achieved_dt, q


def _report_dataset(header: str, windows, skip_reason: str, models):
    """Print `header`, then either a SKIPPED line (if windows is None) or,
    for each non-None model, its nMSE on the given windows. Shared by the
    Silverbox, Cascaded_Tanks, and Bouc-Wen blocks in main()."""
    print(header)
    if windows is None:
        print(f"    SKIPPED -- {skip_reason}")
        return
    u, y = windows
    for name, model in models.items():
        if model is None:
            continue
        v = nmse_on_windows(model, u, y)
        print(f"    {name} model: nMSE={v:.4f}  (n={u.shape[0]} windows)")


def main():
    print("=== real-plant zero-shot transfer (in-context nMSE) ===")
    models = {name: load_model(f"eval_{name}_s0.pt") for name in ("wh_only", "corpus")}
    missing = [name for name, m in models.items() if m is None]
    if missing:
        print(f"  (checkpoint(s) missing, skipped: {', '.join(missing)} -- "
              f"run `python -m plantforge.evaluate <mode>` first)")

    sb_windows, sb_dt, sb_q = silverbox_windows()
    _report_dataset(
        f"  Silverbox (decimated {sb_q}x -> dt={sb_dt:.4f}s, ~50Hz-like):",
        sb_windows, "no full window available after decimation", models)

    ct_windows, ct_dt = cascaded_tanks_windows()
    _report_dataset(
        f"  Cascaded_Tanks (native dt={ct_dt:.2f}s -- "
        f"{ct_dt / max(RATES):.0f}x coarser than trained range, EXTRAPOLATION):",
        ct_windows, "record too short for one context window", models)

    bw_windows, bw_dt, bw_q = boucwen_windows()
    _report_dataset(
        f"  Bouc-Wen (decimated {bw_q}x -> dt={bw_dt:.4f}s, exact 50Hz match):",
        bw_windows, "no full window available after decimation", models)

    wh_duration, wh_min_needed = wienerhammer_status()
    print(f"  WienerHammerBenchMark: SKIPPED -- record duration "
          f"({wh_duration:.2f}s) shorter than one context window at any "
          f"trained rate (min {wh_min_needed:.2f}s) -- not evaluable zero-shot.")


if __name__ == "__main__":
    main()
