"""Text rendering for metaphrasis.

The encoding, in one sentence: energy becomes a block-character sparkline,
everything else becomes a short word or number, and a whole library fits in a
table you can read top to bottom.
"""

from __future__ import annotations

BLOCKS = "▁▂▃▄▅▆▇█"

# Semitone distance around the circle of fifths, indexed by pitch-class gap.
# Neighbours on the circle share six of seven notes and splice without a wince.
_FIFTHS_DISTANCE = {0: 0, 7: 1, 5: 1, 2: 2, 10: 2, 9: 3, 3: 3, 4: 4, 8: 4, 11: 5, 1: 5, 6: 6}
_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def sparkline(contour) -> str:
    """Energy contour as block characters."""
    if not contour:
        return ""
    return "".join(BLOCKS[min(7, max(0, int(v * 7.999)))] for v in contour)


def clock(seconds) -> str:
    if seconds is None:
        return "  -  "
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def key_label(key: dict) -> str:
    """Short key name, or '?' when the estimate is not trustworthy."""
    if not key or not key.get("key"):
        return "?"
    # A thin margin over the runner-up means drones or ambiguous tonality.
    if key.get("margin", 0) < 0.05:
        return "?"
    suffix = "m" if key.get("mode") == "minor" else ""
    return f"{key['key']}{suffix}"


def fingerprint(record: dict, name_width: int = 26) -> str:
    """One line per track. This is the scan format."""
    name = record["file"].removesuffix(".mp3").removesuffix(".wav")
    if len(name) > name_width:
        name = name[: name_width - 1] + "…"

    bpm = f"{record['tempo_bpm']:>5.0f}" if record.get("rhythmic") else "    -"
    lufs = record["loudness"].get("lufs")
    lufs_s = f"{lufs:>6.1f}" if lufs is not None else "     -"
    vo = record.get("voiceover", {}).get("grade", "?")
    words = " ".join(record["timbre"]["words"])

    return (
        f"{name:<{name_width}} {clock(record['duration_sec']):>5} {bpm} "
        f"{key_label(record['key']):>4} {lufs_s} {vo:<4} "
        f"{sparkline(record['contour'])} {words}"
    )


def scan_table(records: list, name_width: int = 26) -> str:
    header = (
        f"{'TRACK':<{name_width}} {'LEN':>5} {'BPM':>5} {'KEY':>4} {'LUFS':>6} {'VO':<4} "
        f"{'SHAPE (quiet ▁ → loud █)':<40} TIMBRE"
    )
    lines = [header, "-" * (len(header) + 6)]
    lines += [fingerprint(r, name_width) for r in records]
    return "\n".join(lines)


def strip_chart(record: dict, rows: int = 24) -> str:
    """Detailed read of a single track: time, energy, and splice candidates."""
    contour = record["contour"]
    duration = record["duration_sec"]
    step = duration / len(contour)

    # Bucket splice points so each row can show whether a cut lands near it.
    points = record.get("splice_points", [])
    per_bucket = {}
    for p in points:
        idx = min(len(contour) - 1, int(p["sec"] / step))
        per_bucket.setdefault(idx, []).append(p)

    out = [
        f"{record['file']}  ·  {clock(duration)}  ·  "
        f"{'%.0f BPM' % record['tempo_bpm'] if record['rhythmic'] else 'no usable beat grid'}  ·  "
        f"key {key_label(record['key'])}  ·  {record['loudness'].get('lufs')} LUFS  "
        f"(range {record['loudness'].get('range_lu')} LU)",
        f"timbre: {', '.join(record['timbre']['words'])}   "
        f"speech-band share: {record['speech_band_share']:.0%}   "
        f"voiceover: {record['voiceover']['grade']}",
        "",
        "  time │ energy               │ cut here?",
        "  ─────┼──────────────────────┼──────────",
    ]

    group = max(1, len(contour) // rows)
    for start in range(0, len(contour), group):
        chunk = contour[start:start + group]
        level = sum(chunk) / len(chunk)
        bar = BLOCKS[min(7, int(level * 7.999))] * 12
        t = start * step
        marks = []
        for i in range(start, min(len(contour), start + group)):
            marks.extend(per_bucket.get(i, []))
        if marks:
            best = max(marks, key=lambda m: m["score"])
            note = f"◆ {best['kind']} @ {best['sec']:.1f}s (score {best['score']})"
        else:
            note = ""
        out.append(f"  {clock(t):>5} │ {bar:<20} │ {note}")

    if record["voiceover"]["reasons"]:
        out.append("")
        out.append("voiceover notes:")
        out += [f"  - {r}" for r in record["voiceover"]["reasons"]]

    return "\n".join(out)


# --------------------------------------------------------------------------
# joinability
# --------------------------------------------------------------------------

def join_score(a: dict, b: dict) -> dict:
    """How cleanly track A can run into track B.

    Scores the four things that make a join wince: clashing keys, a tempo jump,
    a loudness step, and a timbre change the ear reads as 'different recording'.
    """
    penalties = []
    score = 100

    ka, kb = a["key"], b["key"]
    if key_label(ka) != "?" and key_label(kb) != "?":
        gap = (_PITCHES.index(kb["key"]) - _PITCHES.index(ka["key"])) % 12
        dist = _FIFTHS_DISTANCE[gap]
        if dist >= 4:
            score -= 35
            penalties.append(f"keys {key_label(ka)}→{key_label(kb)} are distant")
        elif dist >= 2:
            score -= 12
            penalties.append(f"keys {key_label(ka)}→{key_label(kb)} need a crossfade")
        if ka.get("mode") != kb.get("mode") and dist >= 2:
            score -= 8
            penalties.append("major/minor shift")
    else:
        penalties.append("one side is tonally ambiguous, key match unchecked")

    if a.get("rhythmic") and b.get("rhythmic"):
        ta, tb = a["tempo_bpm"], b["tempo_bpm"]
        if ta > 0 and tb > 0:
            ratio = max(ta, tb) / min(ta, tb)
            # Halves and doubles feel intentional; anything between them lurches.
            off = min(abs(ratio - 1), abs(ratio - 2), abs(ratio - 0.5))
            if off > 0.12:
                score -= 25
                penalties.append(f"tempo {ta:.0f}→{tb:.0f} BPM will lurch")
    elif a.get("rhythmic") != b.get("rhythmic"):
        score -= 15
        penalties.append("one side has a pulse and the other does not")

    la, lb = a["loudness"].get("lufs"), b["loudness"].get("lufs")
    if la is not None and lb is not None and abs(la - lb) > 3:
        score -= 15
        penalties.append(f"{abs(la - lb):.1f} LU level step, needs matching")

    wa, wb = set(a["timbre"]["words"]), set(b["timbre"]["words"])
    if not wa & wb:
        score -= 15
        penalties.append("no shared timbre, will read as two different pieces")

    score = max(0, score)
    verdict = "clean" if score >= 75 else ("workable" if score >= 50 else "rough")
    return {"score": score, "verdict": verdict, "penalties": penalties}


def matrix(records: list, name_width: int = 22) -> str:
    """Grid of join scores across a library."""
    names = [r["file"].removesuffix(".mp3") for r in records]
    short = [n[:6] for n in names]

    lines = ["Join scores: row → column (how cleanly A runs into B)", ""]
    lines.append(" " * (name_width + 2) + " ".join(f"{s:>6}" for s in short))
    for i, rec_a in enumerate(records):
        cells = []
        for j, rec_b in enumerate(records):
            if i == j:
                cells.append("     ·")
                continue
            cells.append(f"{join_score(rec_a, rec_b)['score']:>6}")
        label = names[i][:name_width]
        lines.append(f"{label:<{name_width}}  " + " ".join(cells))
    lines.append("")
    lines.append("75+ clean · 50-74 workable · under 50 rough")
    return "\n".join(lines)
