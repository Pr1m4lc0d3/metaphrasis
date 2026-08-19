"""Audio feature extraction for metaphrasis.

Turns an audio file into a compact, cache-backed feature record: tempo and beat
grid, musical key, perceptual loudness, timbre descriptors, an energy contour,
and ranked splice points.

The deep drum/event analysis deliberately lives elsewhere. For beat-syncing a
video, use music-to-video's analyze-beatgrid.py — this module exists to make a
LIBRARY legible and to find clean cut points, not to replace that analyzer.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import warnings
from pathlib import Path

import librosa
import numpy as np
import scipy.stats

warnings.filterwarnings("ignore")

SR = 22050
CACHE_VERSION = 6

# Krumhansl-Schmuckler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# The band a spoken voice occupies. Music with a lot of energy here fights narration.
SPEECH_LO, SPEECH_HI = 300.0, 3400.0


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def _cache_path(audio: Path) -> Path:
    cache_dir = audio.parent / ".metaphrasis"
    stat = audio.stat()
    stamp = f"{audio.name}:{stat.st_size}:{int(stat.st_mtime)}:{CACHE_VERSION}"
    digest = hashlib.sha1(stamp.encode()).hexdigest()[:16]
    return cache_dir / f"{audio.stem}.{digest}.json"


def load_cached(audio: Path):
    path = _cache_path(audio)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def save_cached(audio: Path, record: dict) -> None:
    path = _cache_path(audio)
    path.parent.mkdir(exist_ok=True)
    # Drop records for this track from earlier cache versions or edits, so the
    # directory holds exactly one entry per file instead of growing forever.
    for stale in path.parent.glob(f"{audio.stem}.*.json"):
        if stale != path:
            stale.unlink(missing_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------
# individual measurements
# --------------------------------------------------------------------------

def measure_loudness(audio: Path) -> dict:
    """Integrated loudness and loudness range via ffmpeg's EBU R128 meter.

    LUFS is a perceptual model, unlike raw peak or RMS, so it is the honest
    number for 'how loud will this feel under a voice'.
    """
    proc = subprocess.run(
        ["ffmpeg", "-nostats", "-i", str(audio), "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    tail = proc.stderr[-2000:]
    integrated = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", tail)
    lra = re.search(r"LRA:\s*(-?[\d.]+)\s*LU", tail)
    peak = re.search(r"Peak:\s*(-?[\d.]+)\s*dBFS", tail)
    return {
        "lufs": float(integrated.group(1)) if integrated else None,
        "range_lu": float(lra.group(1)) if lra else None,
        "true_peak_db": float(peak.group(1)) if peak else None,
    }


def detect_key(y: np.ndarray, sr: int) -> dict:
    """Estimate key by correlating the average chroma against both profiles.

    `margin` is the gap to the runner-up. A small margin means the track is
    tonally ambiguous — drones and static pads often are — and callers should
    not trust the key for harmonic matching.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    total = chroma.sum()
    if total <= 0:
        return {"key": None, "mode": None, "confidence": 0.0, "margin": 0.0}
    chroma = chroma / total

    scored = []
    for i in range(12):
        rotated = np.roll(chroma, -i)
        for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            r = np.corrcoef(rotated, profile / profile.sum())[0, 1]
            scored.append((float(r), _PITCHES[i], mode))
    scored.sort(reverse=True)

    top, runner = scored[0], scored[1]
    return {
        "key": top[1],
        "mode": top[2],
        "confidence": round(top[0], 3),
        "margin": round(top[0] - runner[0], 3),
    }


def describe_timbre(y: np.ndarray, sr: int) -> dict:
    """Spectral shape, plus plain words for it.

    The words are the point: 'centroid 684 Hz' means nothing at a glance,
    'dark' does.
    """
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())

    if centroid < 800:
        brightness = "dark"
    elif centroid < 1800:
        brightness = "warm"
    elif centroid < 3200:
        brightness = "bright"
    else:
        brightness = "harsh"

    # Flatness separates tonal material (pads, strings) from noisy material
    # (cymbals, distortion, texture beds).
    texture = "smooth" if flatness < 0.01 else ("textured" if flatness < 0.05 else "noisy")
    body = "thin" if bandwidth > 2600 else ("full" if bandwidth > 1200 else "narrow")

    return {
        "centroid_hz": round(centroid),
        "rolloff85_hz": round(rolloff),
        "flatness": round(flatness, 5),
        "bandwidth_hz": round(bandwidth),
        "words": [brightness, texture, body],
    }


def speech_band_share(y: np.ndarray, sr: int) -> float:
    """Fraction of spectral energy sitting where a spoken voice lives."""
    spec = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= SPEECH_LO) & (freqs <= SPEECH_HI)
    total = spec.sum()
    if total <= 0:
        return 0.0
    return float(spec[band].sum() / total)


def energy_contour(y: np.ndarray, sr: int, buckets: int = 40) -> list:
    """Normalised RMS energy resampled to a fixed number of buckets.

    Fixed width is what makes tracks of different lengths comparable at a
    glance in a scan table.
    """
    rms = librosa.feature.rms(y=y).flatten()
    if rms.size == 0:
        return [0.0] * buckets
    edges = np.linspace(0, rms.size, buckets + 1).astype(int)
    out = []
    for i in range(buckets):
        chunk = rms[edges[i]:edges[i + 1]]
        out.append(float(chunk.mean()) if chunk.size else 0.0)
    peak = max(out) or 1.0
    return [round(v / peak, 4) for v in out]


# Onset-envelope kurtosis below this reads as a genuine repeating pulse.
# Calibrated on 42 instrumental tracks where the intended style was known from
# the generation prompt: everything written as a groove scored under 11, every
# ambient bed scored over 13, with nothing in between. Material very unlike
# that (live drums, sparse acoustic) may want a different line.
PULSE_KURTOSIS_MAX = 12.0


def pulse_clarity(onset_env: np.ndarray) -> float:
    """Kurtosis of the onset envelope — low means a steady repeating pulse.

    librosa's tracker always returns a tempo. On a sustained pad it returns a
    metronome it invented, and anchoring cuts to that makes a video twitch
    against nothing. Measuring where the beats landed cannot catch this, because
    the tracker places beats on envelope peaks by construction — that test was
    tried and scored the most static pad in the library highest.

    The shape of the envelope does separate them. A groove produces many
    similar peaks, so the distribution is broad and kurtosis is low. A pad is
    mostly flat with a few swells, which is the textbook heavy-tailed shape.
    """
    if onset_env.size < 8:
        return float("inf")
    return float(scipy.stats.kurtosis(onset_env))


def find_splice_points(y: np.ndarray, sr: int, tempo: float, beats_sec: list) -> list:
    """Rank moments where a cut is least likely to be audible.

    Two regimes, because they need opposite logic:

    - Rhythmic material: cut on a downbeat, so the meter survives the join.
    - Ambient material: there is no meter to protect, so cut in a trough where
      little is sounding and a crossfade can hide the seam.
    """
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y).flatten()
    times = librosa.frames_to_time(np.arange(rms.size), sr=sr)
    peak = rms.max() or 1.0
    norm = rms / peak

    candidates = []
    rhythmic = tempo > 0 and len(beats_sec) > 8

    if rhythmic:
        # Every 4th beat approximates a bar line for common time. Spacing them
        # out keeps the list a shortlist rather than a transcript of the grid.
        last_kept = -99.0
        for i in range(0, len(beats_sec), 4):
            t = beats_sec[i]
            if t < 1.0 or t > duration - 1.0:
                continue
            if t - last_kept < 4.0:
                continue
            last_kept = t
            idx = int(np.argmin(np.abs(times - t)))
            candidates.append({
                "sec": round(float(t), 3),
                "kind": "downbeat",
                "energy": round(float(norm[idx]), 3),
                "score": round(1.0 - float(norm[idx]) * 0.35, 3),
            })
    else:
        # Local minima of the energy envelope, smoothed so we find real troughs
        # rather than sample-level noise.
        window = max(3, int(sr / 512 * 0.75))
        kernel = np.ones(window) / window
        smooth = np.convolve(norm, kernel, mode="same")
        for i in range(window, smooth.size - window):
            if smooth[i] <= smooth[i - window] and smooth[i] <= smooth[i + window]:
                t = float(times[i])
                if t < 1.0 or t > duration - 1.0:
                    continue
                if candidates and t - candidates[-1]["sec"] < 2.0:
                    continue
                candidates.append({
                    "sec": round(t, 3),
                    "kind": "trough",
                    "energy": round(float(smooth[i]), 3),
                    "score": round(1.0 - float(smooth[i]), 3),
                })

    candidates.sort(key=lambda c: -c["score"])
    return candidates[:60]


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

def contour_volatility(contour: list) -> float:
    """Mean absolute change between adjacent buckets — how much a track lurches.

    Distinct from loudness range, and the distinction matters. LRA is a
    statistical spread over the whole piece; a track can sit inside a modest
    range while still jumping constantly within it. That restlessness is what
    pulls an ear off a voice, and only this measure sees it.
    """
    if len(contour) < 2:
        return 0.0
    steps = [abs(contour[i + 1] - contour[i]) for i in range(len(contour) - 1)]
    return round(sum(steps) / len(steps), 4)


def voiceover_verdict(loudness: dict, speech_share: float, range_lu, volatility: float = 0.0) -> dict:
    """Judge whether a track can sit under narration, and say why.

    Deliberately conservative: the cost of a bed that fights the voice is a
    re-edit, the cost of a false rejection is auditioning one more track.

    This is triage, not a ruling. It reliably catches loud, busy and restless
    material; it cannot tell you a track is beautiful.
    """
    reasons = []
    score = 100

    if volatility > 0.25:
        score -= 30
        reasons.append(f"restless — energy jumps {volatility:.2f} per step")
    elif volatility > 0.19:
        score -= 12
        reasons.append(f"somewhat restless ({volatility:.2f} per step)")

    if speech_share > 0.36:
        score -= 40
        reasons.append(f"{speech_share:.0%} of energy in the speech band")
    elif speech_share > 0.28:
        score -= 15
        reasons.append(f"{speech_share:.0%} in the speech band is borderline")

    lufs = loudness.get("lufs")
    if lufs is not None:
        if lufs > -14:
            score -= 25
            reasons.append(f"{lufs} LUFS is loud for a bed")
        elif lufs > -17:
            score -= 10
            reasons.append(f"{lufs} LUFS needs pulling down")

    if range_lu is not None:
        if range_lu > 11:
            score -= 20
            reasons.append(f"{range_lu} LU swing will duck in and out under speech")
        elif range_lu < 3:
            reasons.append("very static, easy to sit under a voice")

    score = max(0, score)
    grade = "good" if score >= 75 else ("ok" if score >= 50 else "poor")
    return {"grade": grade, "score": score, "reasons": reasons}


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

def analyse(audio: Path, use_cache: bool = True) -> dict:
    """Full feature record for one file."""
    audio = Path(audio)
    if use_cache:
        cached = load_cached(audio)
        if cached:
            return cached

    y, sr = librosa.load(str(audio), sr=SR, mono=True)
    duration = len(y) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beats_sec = [round(float(t), 3) for t in librosa.frames_to_time(beat_frames, sr=sr)]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    onset_rate = len(onsets) / duration if duration else 0.0

    loudness = measure_loudness(audio)
    key = detect_key(y, sr)
    timbre = describe_timbre(y, sr)
    share = speech_band_share(y, sr)

    # A beat grid on ambient material is a metronome the tracker invents.
    clarity = pulse_clarity(onset_env)
    rhythmic = clarity < PULSE_KURTOSIS_MAX and len(beats_sec) > 8

    record = {
        "file": audio.name,
        "path": str(audio),
        "duration_sec": round(duration, 2),
        "tempo_bpm": round(tempo, 1),
        "rhythmic": bool(rhythmic),
        "pulse_clarity": round(clarity, 3),
        "onset_rate": round(onset_rate, 2),
        "beats_sec": beats_sec,
        "key": key,
        "loudness": loudness,
        "timbre": timbre,
        "speech_band_share": round(share, 4),
        "contour": energy_contour(y, sr),
        "splice_points": find_splice_points(y, sr, tempo if rhythmic else 0.0, beats_sec),
    }
    record["volatility"] = contour_volatility(record["contour"])
    record["voiceover"] = voiceover_verdict(
        loudness, share, loudness.get("range_lu"), record["volatility"]
    )

    if use_cache:
        save_cached(audio, record)
    return record
