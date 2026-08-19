---
name: metaphrasis
description: Read audio you cannot hear — music libraries and narrated video alike. Use when choosing background music, checking whether a track sits under a voiceover, triaging a folder of music, finding where a track can be cut, splicing sections from several tracks into one piece that changes tone by topic, or QA-ing a narrated marketing video (transcript, pacing, dead air, and whether the music is burying the voice). Also use when asked what a track sounds like, its tempo, key, loudness, or structure, or what a video says.
---

# metaphrasis

A compact encoding of audio into things a text-and-image model can actually read:
energy becomes a block-character sparkline, tempo and key and loudness become
numbers, timbre becomes words, and structure becomes a strip chart.

**This is measurement, not hearing.** It reliably catches loud, busy, restless,
badly-matched and wrongly-structured material. It cannot tell you a track is
beautiful, and it cannot hear an artifact or an AI shimmer. Say so rather than
letting a grade stand in for a judgement.

## Requirements

ffmpeg and ffprobe on PATH; Python with librosa, numpy, scipy, soundfile.

## Commands

```bash
B=~/.claude/skills/metaphrasis/scripts/metaphrasis.py

python "$B" scan   <dir>                 # one line per track — start here
python "$B" read   <file>                # strip chart + best cut points
python "$B" pick   <dir> --for voiceover # shortlist: voiceover|opener|closer|montage|long
python "$B" matrix <dir>                 # which tracks join cleanly with which
python "$B" plan   <spec.json>           # resolve a splice without rendering
python "$B" splice <spec.json>           # cut and sew
python "$B" narration <video>            # transcript + pacing + is the bed burying the voice
```

Analysis is cached in a `.metaphrasis/` folder beside the audio, keyed on file size
and mtime, so repeat commands are instant. First run on a large library is slow
— roughly a few seconds per track.

## Reading a scan line

```
the-verdict-03    2:59     -    C  -12.9 poor ▁▃▃▃▃▄▅▅▅▇▇▆▇▅▆▇▇▇▅▅▃▅▇▇▇▇▇▇▇███▃▂▃▆▇▆▆▂ dark smooth narrow
                   │       │    │     │    │   │                                        │
                 length  BPM   key  LUFS  VO  energy over the whole track            timbre
```

- **BPM shows `-` when there is no real pulse.** Do not treat that as missing
  data — it means the beat grid would be fiction. See "the metronome trap".
- **KEY shows `?`** when the estimate is tonally ambiguous, which drones often
  are. Do not harmonically match on a `?`.
- **LUFS** is perceptual loudness. A bed under narration usually wants -18 or
  quieter; these Suno tracks arrive around -13 and need pulling down.
- **VO** is triage only: good / ok / poor.
- **The sparkline is the most useful column.** A flat run of the same block is a
  static bed; a rising ramp is a build; `█` in the middle with `▁` at the ends
  is a track with a peak you can cut around.

## The metronome trap

librosa's beat tracker always returns a tempo. On a sustained pad it returns a
metronome it invented, and cutting picture to that makes a video twitch against
nothing.

Testing whether beats land on onsets does NOT catch this — the tracker places
beats on envelope peaks by construction, and that test scored the most static
pad in the library as the most rhythmic. What separates them is the SHAPE of the
onset envelope: a groove has many similar peaks (low kurtosis), a pad is flat
with occasional swells (high kurtosis). The threshold of 12 was calibrated on 42
tracks of known intent, with a clean gap between 11.0 and 13.1. Recalibrate on
very different material.

`music-to-video/scripts/analyze-beatgrid.py` is the deeper analyzer — drum
classification, risers, metrical position — and is the right tool for actually
beat-syncing a video. This skill makes a library legible and finds cut points.
Do not duplicate that analyzer here.

## Splicing

A spec is JSON:

```json
{
  "library": "F:/path/to/music",
  "output":  "F:/path/to/music/cuts/piece.mp3",
  "crossfade_sec": 3.0,
  "target_lufs": -20,
  "segments": [
    { "file": "open-question-01.mp3",    "start": 2,  "duration": 20 },
    { "file": "adversarial-pass-01.mp3", "start": 20, "duration": 22 },
    { "file": "the-verdict-02.mp3",      "start": 24, "duration": 26 }
  ]
}
```

Run `plan` first and read the notes before rendering.

Two rules the engine follows, both learned the hard way:

- **Snapping is for hard cuts only.** Cut points on ambient material are all
  troughs. Snapping both sides of a crossfade to troughs fades near-silence into
  near-silence. With a crossfade, trust the crossfade.
- **Level matching is a static gain, never `loudnorm`.** Single-pass loudnorm is
  a dynamic processor that rides gain and pumps. Each track's integrated
  loudness is already measured, so the correction is arithmetic.

Check `join_score` (via `matrix`) before designing a sequence. It scores key
distance on the circle of fifths, tempo compatibility, level step, and shared
timbre.

## Diagnosing a suspicious moment

Before blaming the splice engine for a dip or a bump, **map the timestamp back
to the source and measure it there.** `acrossfade` output is
`lenA + lenB - crossfade`, and the overlap sits in A's tail — so each segment's
audio starts earlier in the finished piece than a naive sum suggests. Getting
that mapping wrong turns ordinary source material into an imaginary bug, which
has already happened once.

## Narrated video

`narration <video>` transcribes the voiceover and measures it against the music
under it. Reports words per minute, rushed and dragging lines, dead air, and a
mix verdict: **clear** (8 dB+ separation), **tight** (4-8 dB), **buried** (under
4 dB).

Transcription runs offline through `tools/WhisperCli` — a small C# host over
Whisper.net 1.9.0, the same version Deliberon ships, using
`ggml-base.en.bin` from Deliberon's Data folder. Build once:

```bash
cd ~/.claude/skills/metaphrasis/tools/WhisperCli && dotnet build -c Release
```

Override with `WHISPER_CLI` and `WHISPER_MODEL` if either moves.

⛔ **The Python route does not work on this machine.** `transformers` imports
sklearn which imports pandas, and pandas there is built against an older numpy
ABI, so the import fails. Deliberon depends on that environment — do not "fix"
it to make a transcript happen.

A vocabulary prompt is passed by default (`VOCABULARY` in `narration.py`) because
Whisper otherwise returns "Deliveron" and "Box CPM2". Extend it for new product
names. It improves the words but makes segments longer and coarser, which is why
the mix measurement ignores segments entirely.

**The mix number must never depend on the transcript.** It is measured from the
speech-band energy ratio in the audio itself. An earlier version sampled the bed
from gaps between Whisper segments; adding the vocabulary prompt removed those
gaps and swung the verdict from "clear" to "buried" on a video that was fine.
Measuring the transcriber is not measuring the mix.

## Composes with

- **`paraphrasis`** — the mirror. This crosses a medium literally; that crosses an audience faithfully. Metaphrase is word-for-word, paraphrase is sense-for-sense, and both are built around what they refuse to do.
- **`music-to-video` / `hyperframes`** — pick the bed and the cut points here, build the video there, then come back for the narration pass. The same speech-band measurement that chose the bed checks afterwards whether the voice came through.
- **`fortress-truth`** — a claims audit of a *spoken* asset. Transcribe the narration, then run every claim against the rail. Video copy is rarely linted because it is not text until something transcribes it.
- **`peitho`** — for the script, once the transcript exists.

Full graph in `../SKILLS-MAP.md`.

## What this cannot do

- **Hear.** No aesthetic judgement, no artifact detection.
- **Lyrics.** Vocals need speech-to-text, which is a separate tool. Whisper is
  available on this machine (Whisper.net 1.9.0 ships with Deliberon, with
  `ggml-base.en.bin` under its Data folder; `torch` and `transformers` are also
  installed with `whisper-base.en` cached). Whisper is trained on speech and
  degrades on sung vocals over music, so check its output against real lyrics.
- **Replace listening.** Use it to cut 42 candidates down to 8, then listen.
