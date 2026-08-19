"""Cut and sew engine for metaphrasis.

Takes a sequence of segments drawn from any tracks in a library and renders one
continuous piece: segments snapped to clean cut points, levels matched, and
equal-power crossfades hiding the joins.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import features


def snap_to_splice(record: dict, seconds: float, window: float = 2.5, prefer: str = "quiet"):
    """Move a requested time to the nearest clean cut point.

    `prefer` picks what "clean" means, and the two cases are opposites:

    - "quiet" — for a hard cut. Land in a trough so the splice has nothing
      sounding across it.
    - "body"  — for a crossfade. Land where the music has substance, because a
      crossfade fades BOTH sides down through the overlap. Snapping both ends
      of a join to troughs fades silence into silence and digs an audible hole
      in the middle of the transition.

    Returns (time, note), falling back to the exact time rather than dragging
    the edit somewhere the caller did not ask for.
    """
    points = record.get("splice_points", [])
    if not points:
        return seconds, "no splice points, used exact time"

    near = [p for p in points if abs(p["sec"] - seconds) <= window]
    if not near:
        return seconds, f"nothing clean within {window}s, used exact time"

    if prefer == "body":
        # Closest to mid energy: enough material for the fade to work across,
        # without landing on a peak that would spike through the transition.
        best = min(near, key=lambda p: abs(p.get("energy", 0.5) - 0.5))
        why = f"{best['kind']} with body"
    else:
        best = max(near, key=lambda p: p["score"])
        why = best["kind"]

    delta = best["sec"] - seconds
    return best["sec"], f"snapped {delta:+.2f}s to {why}"


def build_plan(spec: dict, library: Path) -> dict:
    """Resolve a spec into concrete segment times, snapping where asked."""
    crossfade = float(spec.get("crossfade_sec", 2.0))
    resolved = []

    for i, seg in enumerate(spec["segments"]):
        audio = library / seg["file"]
        if not audio.exists():
            raise FileNotFoundError(f"segment {i}: {audio} not found")

        record = features.analyse(audio)
        start = float(seg.get("start", 0.0))
        duration = float(seg["duration"])
        notes = []

        if seg.get("snap", True):
            # Snapping exists to hide a HARD cut, by putting it where nothing is
            # sounding. A crossfade already hides its own seam, and snapping into
            # it makes things worse: on ambient material every candidate point is
            # a trough, so both sides of the join fade down from near-silence and
            # the transition drops into an audible hole. So snap only the edges
            # that are genuinely hard cuts — the outer ends of the whole piece,
            # and every edge when the crossfade is zero.
            last = i == len(spec["segments"]) - 1
            snap_start = crossfade <= 0 or i == 0
            snap_end = crossfade <= 0 or last

            if snap_start:
                start, note = snap_to_splice(record, start)
                notes.append(f"start: {note}")
            end = start + duration
            if snap_end:
                end, note = snap_to_splice(record, end)
                notes.append(f"end: {note}")
            else:
                notes.append("end: left free for the crossfade")
            duration = max(1.0, end - start)

        # A crossfade consumes audio from both sides; a segment shorter than the
        # fade would be swallowed whole.
        if duration <= crossfade and i not in (0, len(spec["segments"]) - 1):
            notes.append(f"WARNING duration {duration:.1f}s <= crossfade {crossfade}s")

        available = record["duration_sec"] - start
        if duration > available:
            duration = max(1.0, available)
            notes.append(f"clamped to {duration:.1f}s (end of track)")

        resolved.append({
            "file": seg["file"],
            "path": str(audio),
            "start": round(start, 3),
            "duration": round(duration, 3),
            "notes": notes,
            "key": record["key"],
            "lufs": record["loudness"].get("lufs"),
        })

    # Total length shrinks by one crossfade per join.
    total = sum(s["duration"] for s in resolved) - crossfade * (len(resolved) - 1)
    return {
        "segments": resolved,
        "crossfade_sec": crossfade,
        "target_lufs": float(spec.get("target_lufs", -20.0)),
        "estimated_duration": round(total, 2),
        "output": spec["output"],
    }


def render(plan: dict, dry_run: bool = False) -> str:
    """Render the plan to a single audio file via one ffmpeg invocation."""
    segments = plan["segments"]
    crossfade = plan["crossfade_sec"]
    target = plan["target_lufs"]

    cmd = ["ffmpeg", "-y", "-v", "error"]
    for seg in segments:
        # Seek before -i so ffmpeg does not decode the whole file first.
        cmd += ["-ss", str(seg["start"]), "-t", str(seg["duration"]), "-i", seg["path"]]

    filters = []
    for i, seg in enumerate(segments):
        # Level matching is a STATIC gain, deliberately not loudnorm. Single-pass
        # loudnorm is a dynamic processor: it rides gain over the stream and
        # pumps, which put a hole in this piece several seconds after a join and
        # looked exactly like a bad splice. We already measured each track's
        # integrated loudness, so the correction is simple arithmetic.
        measured = seg.get("lufs")
        gain = 0.0 if measured is None else round(target - measured, 2)
        # Keep a ceiling on boost so a very quiet source does not lift its noise
        # floor into audibility.
        gain = max(-24.0, min(12.0, gain))
        filters.append(
            f"[{i}:a]volume={gain}dB,"
            f"aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"afade=t=in:st=0:d=0.02[s{i}]"
        )

    if len(segments) == 1:
        chain, last = "", "s0"
    else:
        parts = []
        last = "s0"
        for i in range(1, len(segments)):
            out = f"x{i}"
            # qsin is the equal-power curve: constant perceived loudness through
            # the join, where a linear fade would dip in the middle.
            parts.append(f"[{last}][s{i}]acrossfade=d={crossfade}:c1=qsin:c2=qsin[{out}]")
            last = out
        chain = ";" + ";".join(parts)

    filter_complex = ";".join(filters) + chain
    cmd += ["-filter_complex", filter_complex, "-map", f"[{last}]"]

    output = Path(plan["output"])
    if output.suffix.lower() == ".wav":
        cmd += ["-c:a", "pcm_s16le"]
    else:
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
    cmd.append(str(output))

    if dry_run:
        return " ".join(cmd)

    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")
    return str(output)


def describe_plan(plan: dict) -> str:
    lines = [
        f"output: {plan['output']}",
        f"estimated length: {plan['estimated_duration']}s "
        f"({len(plan['segments'])} segments, {plan['crossfade_sec']}s crossfades, "
        f"normalised to {plan['target_lufs']} LUFS)",
        "",
    ]
    for i, seg in enumerate(plan["segments"], 1):
        key = seg["key"].get("key") or "?"
        mode = "m" if seg["key"].get("mode") == "minor" else ""
        lines.append(
            f"{i}. {seg['file']}  {seg['start']}s → {seg['start'] + seg['duration']:.1f}s "
            f"({seg['duration']}s)  key {key}{mode}  {seg['lufs']} LUFS"
        )
        for note in seg["notes"]:
            marker = "  !" if note.startswith("WARNING") else "   "
            lines.append(f"{marker} {note}")
    return "\n".join(lines)


def load_spec(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
