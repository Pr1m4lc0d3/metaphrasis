#!/usr/bin/env python3
"""metaphrasis: make an audio library legible, and cut new pieces from it.

Subcommands:
    scan    <dir>              one line per track, the whole library at a glance
    read    <file>             detailed strip chart for one track
    matrix  <dir>              which tracks join cleanly with which
    pick    <dir> --for <use>  shortlist tracks for a job
    plan    <spec.json>        resolve a splice spec without rendering
    splice  <spec.json>        cut and sew a new piece
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib import features, narration, render, splice  # noqa: E402

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

# What each job needs from a track. Kept here rather than in features so the
# judgement stays editable without touching the measurement code.
USE_CASES = {
    "voiceover": "sits under narration without fighting it",
    "opener": "grabs attention in the first seconds",
    "closer": "resolves and leaves room for an end card",
    "montage": "has a usable pulse to cut picture against",
    "long": "long enough for extended footage",
}


def collect(directory: Path) -> list:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"no audio files in {directory}")
    records = []
    for i, path in enumerate(files, 1):
        print(f"\r  analysing {i}/{len(files)}  {path.name[:40]:<42}", end="", file=sys.stderr)
        records.append(features.analyse(path))
    print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
    return records


def cmd_scan(args):
    records = collect(Path(args.directory))
    if args.json:
        print(json.dumps(records, indent=1))
        return
    print(render.scan_table(records))
    print()
    grades = {}
    for r in records:
        grades[r["voiceover"]["grade"]] = grades.get(r["voiceover"]["grade"], 0) + 1
    summary = "  ".join(f"{k}: {v}" for k, v in sorted(grades.items()))
    print(f"{len(records)} tracks   voiceover suitability — {summary}")


def cmd_read(args):
    record = features.analyse(Path(args.file))
    if args.json:
        print(json.dumps(record, indent=1))
        return
    print(render.strip_chart(record))
    print()
    print("best cut points:")
    for p in record["splice_points"][:10]:
        print(f"  {p['sec']:>8.2f}s  {p['kind']:<9} score {p['score']}")


def cmd_matrix(args):
    records = collect(Path(args.directory))
    print(render.matrix(records))


def cmd_pick(args):
    records = collect(Path(args.directory))
    use = args.use

    if use == "voiceover":
        chosen = [r for r in records if r["voiceover"]["grade"] in ("good", "ok")]
        chosen.sort(key=lambda r: -r["voiceover"]["score"])
    elif use == "opener":
        chosen = [r for r in records if r["duration_sec"] <= 30]
        chosen.sort(key=lambda r: r["duration_sec"])
    elif use == "closer":
        # A closer should end quieter than it peaked.
        chosen = [r for r in records if r["contour"] and r["contour"][-1] < max(r["contour"]) * 0.7]
        chosen.sort(key=lambda r: r["contour"][-1])
    elif use == "montage":
        chosen = [r for r in records if r["rhythmic"]]
        chosen.sort(key=lambda r: -r["onset_rate"])
    elif use == "long":
        chosen = [r for r in records if r["duration_sec"] >= 150]
        chosen.sort(key=lambda r: -r["duration_sec"])
    else:
        raise SystemExit(f"unknown use case: {use}")

    print(f"for '{use}' — {USE_CASES[use]}\n")
    if not chosen:
        print("  nothing in this library fits.")
        return
    print(render.scan_table(chosen))


def cmd_plan(args):
    spec = splice.load_spec(Path(args.spec))
    library = Path(spec.get("library", Path(args.spec).parent))
    plan = splice.build_plan(spec, library)
    print(splice.describe_plan(plan))


def cmd_splice(args):
    spec = splice.load_spec(Path(args.spec))
    library = Path(spec.get("library", Path(args.spec).parent))
    plan = splice.build_plan(spec, library)
    print(splice.describe_plan(plan))
    print()
    if args.dry_run:
        print("ffmpeg command:")
        print(splice.render(plan, dry_run=True))
        return
    out = splice.render(plan)
    print(f"wrote {out}")


def cmd_narration(args):
    script = None
    if args.script:
        sp = Path(args.script)
        if sp.exists():
            script = sp.read_text(encoding="utf-8")
        elif len(args.script) < 260 and ("\\" in args.script or "/" in args.script)                 and args.script.rstrip().endswith((".txt", ".md", ".json")):
            # Looks like a path and is not there. Treating it as the script text instead compares
            # the narration against a filename, which scores near zero and once read as a pass.
            raise SystemExit(f"no script file at {sp}")
        else:
            script = args.script
    report = narration.analyse(Path(args.file), script=script)
    if args.json:
        print(json.dumps(report, indent=1))
        return
    print(narration.render_report(report))


def main():
    # Block characters need a UTF-8 stdout; Windows consoles default otherwise.
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="metaphrasis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="one line per track")
    p.add_argument("directory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("read", help="detailed strip chart for one track")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("matrix", help="which tracks join cleanly")
    p.add_argument("directory")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("pick", help="shortlist tracks for a job")
    p.add_argument("directory")
    p.add_argument("--for", dest="use", required=True, choices=sorted(USE_CASES))
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("narration", help="transcribe a narrated video and check the mix")
    p.add_argument("file")
    # Without this the report can only say how the narration SOUNDS, never whether it is right.
    p.add_argument("--script", help="the text the narrator was given: a file path, or the text "
                                    "itself. Checks for invented and skipped words.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_narration)

    p = sub.add_parser("plan", help="resolve a splice spec without rendering")
    p.add_argument("spec")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("splice", help="cut and sew a new piece")
    p.add_argument("spec")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_splice)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
