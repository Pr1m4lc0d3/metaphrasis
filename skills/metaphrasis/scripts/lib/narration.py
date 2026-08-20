"""Narration analysis for metaphrasis.

Transcribes the voiceover in a video, then measures it against the music under
it. The two halves only mean something together: a transcript says what was
said, and the levels say whether a viewer could hear it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np

warnings.filterwarnings("ignore")

SKILL_ROOT = Path(__file__).resolve().parents[2]

# Whisper.net writes these when it hears music or noise rather than speech.
NON_SPEECH = re.compile(r"^\s*[\[\(][^\])]*[\]\)]\s*$")

# Dialogue is normally mixed several dB above its bed. Below this the voice
# starts competing with the music instead of sitting on top of it.
GOOD_HEADROOM_DB = 8.0
TIGHT_HEADROOM_DB = 4.0

DEFAULT_MODEL = Path(r"D:\Programs\Deliberon AI\Data\whisper-models\ggml-base.en.bin")

# The band a spoken voice occupies, shared with the music analysis so both
# halves of the skill judge "competing with speech" the same way.
from .features import SPEECH_HI, SPEECH_LO  # noqa: E402


def find_whisper_cli() -> Path | None:
    env = os.environ.get("WHISPER_CLI")
    if env and Path(env).exists():
        return Path(env)
    matches = sorted((SKILL_ROOT / "tools" / "WhisperCli" / "bin").rglob("whisper-cli.exe"))
    return matches[-1] if matches else None


def find_model() -> Path | None:
    env = os.environ.get("WHISPER_MODEL")
    if env and Path(env).exists():
        return Path(env)
    return DEFAULT_MODEL if DEFAULT_MODEL.exists() else None


def extract_audio(source: Path, dest: Path, sample_rate: int = 16000) -> Path:
    """Whisper wants 16 kHz mono PCM; give it exactly that."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source),
         "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dest)],
        check=True, capture_output=True,
    )
    return dest


# Seeds the decoder with spellings it would otherwise invent. Whisper returned
# "Deliveron" and "Box CPM2" on the setup walkthrough without this.
VOCABULARY = (
    "Deliberon, INTEGRA, VoxCPM2, CODE A.I., Binary Gold, Chatterbot 2000, "
    "Aletheia, Peitho, Janus, chairman, council, roster, convention, specialist, "
    "vault, loadout, provider, BYO model, Anthropic, OpenAI, DeepSeek, OpenRouter."
)


def transcribe(wav: Path, vocabulary: str | None = VOCABULARY) -> dict:
    cli, model = find_whisper_cli(), find_model()
    if cli is None:
        raise RuntimeError(
            "whisper-cli not built. Run: dotnet build -c Release in "
            f"{SKILL_ROOT / 'tools' / 'WhisperCli'}"
        )
    if model is None:
        raise RuntimeError(
            "no Whisper model found. Set WHISPER_MODEL, or install "
            f"{DEFAULT_MODEL}"
        )
    cmd = [str(cli), str(model), str(wav)]
    if vocabulary:
        cmd.append(vocabulary)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"whisper-cli failed: {proc.stderr.strip()[:500]}")
    return json.loads(proc.stdout)


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, 1e-10))


def measure_voice_against_bed(wav: Path, segments: list | None = None) -> dict:
    """How far the voice rises above the music, in the band where they compete.

    Deliberately does NOT use Whisper's segment boundaries. Those move with
    decoder settings — adding a vocabulary prompt turned tight phrase segments
    into long contiguous chunks, which erased the gaps this measurement samples
    and swung the answer by 9 dB. A number that changes when the transcriber's
    settings change is measuring the transcriber, not the mix.

    Instead, detect speech from the audio itself. A voice concentrates energy
    between 300 and 3400 Hz, so the ratio of in-band to total energy rises
    sharply when someone talks over a dark music bed. Comparing in-band level
    when that ratio is high against when it is low gives the separation a
    listener actually experiences.

    Still an estimate from a finished mix rather than from stems: treat it as a
    flag, not a spec. It also under-reports separation on mixes that duck the
    music under the voice, since the bed is sampled where the voice is absent.
    """
    y, sr = librosa.load(str(wav), sr=16000, mono=True)
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

    band = (freqs >= SPEECH_LO) & (freqs <= SPEECH_HI)
    band_energy = stft[band].sum(axis=0)
    total_energy = stft.sum(axis=0) + 1e-10
    ratio = band_energy / total_energy

    # Ignore near-silence: it belongs to neither voice nor bed and would drag
    # both estimates down.
    band_db = _db(band_energy)
    audible = band_db > (np.percentile(band_db, 95) - 40)
    if audible.sum() < 10:
        return {"error": "file is effectively silent"}

    # Split on the ratio rather than a fixed threshold, so the measurement
    # adapts to how dark or bright the bed happens to be.
    high, low = np.percentile(ratio[audible], 75), np.percentile(ratio[audible], 25)
    if high - low < 0.02:
        return {"error": "no clear speech/music separation — is there narration?"}

    # Two different jobs, two different masks.
    #
    # Levels are read from the top and bottom quartiles: those are clean
    # exemplars of "voice present" and "music only", away from the boundary.
    #
    # Coverage needs an actual decision boundary, not a quantile — a quartile is
    # 25% of frames by construction, so reporting it as "how much of the video
    # has narration" would always say 25% no matter what the file contains. That
    # error made a 147 wpm read look like 328 wpm.
    voice_mask = audible & (ratio >= high)
    bed_mask = audible & (ratio <= low)
    if not voice_mask.any() or not bed_mask.any():
        return {"error": "could not isolate speech and music passages"}

    threshold = (high + low) / 2.0
    speech_frames = audible & (ratio >= threshold)
    coverage = float(speech_frames.sum() / max(1, ratio.size))

    voice_db = float(np.median(band_db[voice_mask]))
    bed_db = float(np.median(band_db[bed_mask]))
    headroom = round(voice_db - bed_db, 1)

    if headroom >= GOOD_HEADROOM_DB:
        verdict, note = "clear", "voice sits well above the bed"
    elif headroom >= TIGHT_HEADROOM_DB:
        verdict, note = "tight", "voice is only just above the bed"
    else:
        verdict, note = "buried", "the bed is competing with the voice"

    return {
        "voice_db": round(voice_db, 1),
        "bed_db": round(bed_db, 1),
        "headroom_db": headroom,
        "verdict": verdict,
        "note": note,
        "method": "speech-band ratio, independent of the transcript",
        "speech_coverage": round(coverage, 3),
    }


def find_dead_air(segments: list, duration: float, threshold: float = 3.0) -> list:
    """Gaps between spoken lines longer than the threshold."""
    spoken = [s for s in segments if s["text"].strip() and not NON_SPEECH.match(s["text"])]
    gaps, previous = [], 0.0
    for seg in spoken:
        if seg["start"] - previous >= threshold:
            gaps.append({"start": round(previous, 1), "end": round(seg["start"], 1),
                         "length": round(seg["start"] - previous, 1)})
        previous = max(previous, seg["end"])
    if duration - previous >= threshold:
        gaps.append({"start": round(previous, 1), "end": round(duration, 1),
                     "length": round(duration - previous, 1)})
    return gaps


def pacing(segments: list, speaking_seconds: float | None = None) -> dict:
    """Words per minute overall and per line, to catch rushed or dragging reads.

    Overall rate prefers the speaking time measured from the audio, because
    segment spans stretch to fill gaps when the decoder chunks coarsely and that
    makes the rate look far slower than the read actually is.
    """
    spoken = [s for s in segments if s["text"].strip() and not NON_SPEECH.match(s["text"])]
    if not spoken:
        return {"wpm": None, "rushed": [], "slow": [], "per_line": False}

    total_words = sum(len(s["text"].split()) for s in spoken)
    spans = [max(0.1, s["end"] - s["start"]) for s in spoken]
    total_time = speaking_seconds if speaking_seconds else sum(spans)
    overall = total_words / total_time * 60 if total_time else 0

    # Per-line rates are only meaningful when the decoder produced phrase-sized
    # segments. Long chunks average several sentences together and would report
    # every line as comfortable.
    phrase_level = float(np.median(spans)) <= 8.0
    if not phrase_level:
        return {"wpm": round(overall), "words": total_words,
                "rushed": [], "slow": [], "per_line": False}

    rushed, slow = [], []
    for s in spoken:
        span = max(0.1, s["end"] - s["start"])
        words = len(s["text"].split())
        if words < 4:
            continue
        rate = words / span * 60
        # Comfortable narration runs roughly 130-170 wpm.
        if rate > 195:
            rushed.append({"start": s["start"], "wpm": round(rate), "text": s["text"][:70]})
        elif rate < 95:
            slow.append({"start": s["start"], "wpm": round(rate), "text": s["text"][:70]})

    return {"wpm": round(overall), "words": total_words,
            "rushed": rushed, "slow": slow, "per_line": True}


# Whisper's own habits rather than the narrator's. Stripping them stops a take being blamed for
# how the transcriber punctuated a pause.
_FILLER = {"uh", "um", "mm", "hmm", "ah"}


def _script_words(text: str) -> list:
    out = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return [w for w in out if w not in _FILLER]


def compare_to_script(transcript: str, script: str) -> dict:
    """Did the narrator say what it was given?

    This exists because a synthesiser will occasionally invent a word. A take went out reading
    "And not a list of ingredients" where the script said "Not a list of ingredients", and every
    other measure here was happy: the mix was clean, the pacing was normal, the transcript was
    printed in full and read perfectly well on its own. Nothing compares it to what was ASKED
    for, so nothing could see the extra word. A human ear caught it.

    ADDED and DROPPED words are faults. SUBSTITUTED words are not reported as faults, because
    Whisper hears "would" as "wood" and "she is" as "she has" on takes that are perfectly fine.
    Treating those as errors would condemn good audio; an insertion or a deletion is a real
    difference in what was spoken.
    """
    want, got = _script_words(script), _script_words(transcript)
    ratio = SequenceMatcher(None, want, got).ratio()

    # A comparison that cannot line up must never come back clean. Handed the wrong file, this
    # first reported "faithful" at a 2% match, because when nothing corresponds every difference
    # lands in a `replace` and the two categories it does report stay empty. A reassuring verdict
    # from a broken comparison is worse than no check at all.
    if ratio < 0.6:
        return {"added": [], "dropped": [], "similarity": round(ratio, 3),
                "faithful": False,
                "error": "script and audio do not correspond (%.0f%% match). Wrong file, or the "
                         "narration is not this script." % (ratio * 100)}

    added, dropped = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, want, got).get_opcodes():
        if tag == "insert":
            added.append(" ".join(got[j1:j2]))
        elif tag == "delete":
            dropped.append(" ".join(want[i1:i2]))
        elif tag == "replace":
            pass  # homophones and near-misses; see the docstring
    return {
        "added": added,
        "dropped": dropped,
        "similarity": round(ratio, 3),
        "faithful": not added and not dropped,
    }


def analyse(source: Path, script: str | None = None) -> dict:
    """Full narration report for a video or audio file.

    Pass `script` (the text the narrator was given) to also check that it said it. Without one,
    every measure here is about how the narration SOUNDS, and none is about whether it is right.
    """
    source = Path(source)
    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_audio(source, Path(tmp) / "audio.wav")
        result = transcribe(wav)
        segments = result.get("segments", [])
        duration = librosa.get_duration(path=str(wav))
        mix = measure_voice_against_bed(wav)
        # Speaking time from the audio, not from segment spans.
        speaking_seconds = duration * mix.get("speech_coverage", 0) or None
        transcript = result.get("text", "") or " ".join(
            s["text"].strip() for s in segments)
        report = {
            "file": source.name,
            "duration_sec": round(duration, 1),
            "transcript": transcript,
            "segments": segments,
            "mix": mix,
            "pacing": pacing(segments, speaking_seconds),
            "dead_air": find_dead_air(segments, duration),
        }
        if script:
            report["script_check"] = compare_to_script(transcript, script)
    return report


def render_report(report: dict) -> str:
    mix, pace = report["mix"], report["pacing"]
    minutes = int(report["duration_sec"] // 60)
    seconds = int(report["duration_sec"] % 60)

    lines = [
        f"{report['file']}  ·  {minutes}:{seconds:02d}  ·  "
        f"{pace.get('words', 0)} words at {pace.get('wpm', '?')} wpm",
        "",
        "MIX",
    ]
    if "error" in mix:
        lines.append(f"  {mix['error']}")
    else:
        lines.append(f"  voice {mix['voice_db']} dB · bed {mix['bed_db']} dB · "
                     f"headroom {mix['headroom_db']} dB")
        lines.append(f"  {mix['verdict'].upper()} — {mix['note']}")
        lines.append(f"  speech covers {mix['speech_coverage']:.0%} of the runtime")

    if pace["rushed"]:
        lines += ["", "RUSHED LINES (over 195 wpm)"]
        lines += [f"  {r['start']:>7.1f}s  {r['wpm']} wpm  {r['text']}" for r in pace["rushed"][:8]]
    if pace["slow"]:
        lines += ["", "SLOW LINES (under 95 wpm)"]
        lines += [f"  {r['start']:>7.1f}s  {r['wpm']} wpm  {r['text']}" for r in pace["slow"][:8]]

    # Before pacing and dead air: a line the narrator got WRONG matters more than one it took
    # half a second too long over, and this is the check that was missing when a bad take shipped.
    check = report.get("script_check")
    if check is not None:
        lines += ["", "AGAINST THE SCRIPT"]
        if check.get("error"):
            lines.append(f"  UNCHECKED — {check['error']}")
        elif check["faithful"]:
            lines.append(f"  faithful — every word accounted for ({check['similarity']:.0%} match)")
        else:
            for w in check["added"]:
                lines.append(f"  ADDED    {w!r} — not in the script")
            for w in check["dropped"]:
                lines.append(f"  DROPPED  {w!r} — in the script, never spoken")
    else:
        lines += ["", "AGAINST THE SCRIPT",
                  "  not checked — pass the script to catch an invented or skipped word"]

    if report["dead_air"]:
        lines += ["", "DEAD AIR (3s+ with no narration)"]
        lines += [f"  {g['start']:>7.1f}s → {g['end']:.1f}s  ({g['length']}s)"
                  for g in report["dead_air"][:10]]

    lines += ["", "TRANSCRIPT", ""]
    for seg in report["segments"]:
        if seg["text"].strip():
            lines.append(f"  {seg['start']:>7.1f}s  {seg['text']}")
    return "\n".join(lines)
