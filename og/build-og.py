"""Build the GitHub social-preview card.

Recomposes the source artwork (3:2, lockup in the left third) onto GitHub's
1280x640 social-preview ratio. The lockup stays LEFT, where the artwork puts it,
and the empty right side carries copy explaining the skill. That space is the
reason the artwork is left-weighted; it is not padding.

The paper texture is carried across the full frame and the lockup is pasted
through a feathered mask with bleed, so the two textures meet invisibly.

Canonical implementation. The copy in each skill repo must stay identical; if you
change one, change the other.

Run from this folder:
    python build-og.py --config peitho
    python build-og.py --config janus
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
W, H = 1280, 640                       # GitHub social-preview size

FONTS = Path("C:/Windows/Fonts")
F_DISPLAY = FONTS / "GARA.TTF"         # Garamond, for kicker and headline
F_TEXT = FONTS / "corbell.ttf"         # Corbel Light, for the bullets
F_MONOISH = FONTS / "calibril.ttf"     # Calibri Light: lining figures, so a URL sits on the baseline

GOLD, CREAM, MUTED, RULE = "#C9A96A", "#EFE6D4", "#9C9081", "#8A7449"

# Per-skill config. box is the lockup's trimmed bounding box in the source
# artwork, as (x0, y0, x1, y1). Find it with:
#   magick source-artwork.png -fuzz 15% -trim -format '%wx%h+%X+%Y' info:
CONFIGS = {
    "peitho": dict(
        out="peitho-og.png",
        box=(107, 105, 729, 920),
        kicker="PULL IS PARTICIPATION",
        headline="A reader keeps reading because you gave them something to do.",
        bullets=[
            "Visible gaps, not withheld endings",
            "Openings that earn the next paragraph",
            "Every claim sourced. No invented detail.",
        ],
        meta="MIT  ·  10 openings taken apart  ·  every claim evidence-tiered",
    ),
    "janus": dict(
        out="janus-og.png",
        box=(134, 175, 577, 843),
        kicker="CONTRADICTION IS THE POSITION",
        headline="Every other framework dissolves a contradiction. This one refuses.",
        # Straight from skills/janus/SKILL.md steps 1, 2 and 4. Do NOT write "what a
        # rival must give up" here: that is Idea Forge Pro's moat gate, a
        # different thing, and Janus is explicitly not a pipeline stage.
        bullets=[
            "Name the pair that must both be true",
            "Ban the three exits: compromise, sequence, segment",
            "Harvest a mechanism, not a slogan. Neither pole weakens.",
        ],
        meta="MIT  ·  5 steps  ·  3 worked examples",
    ),
    "metaphrasis": dict(
        out="metaphrasis-og.png",
        box=(2, 122, 906, 945),
        # This artwork's binary field runs to x=906, past the default patch.
        tex=(980, 620, 1500, 1010),
        kicker="MEASURED, NOT HEARD",
        headline="Audio, rendered as something a model can read.",
        # Straight from skills/metaphrasis/SKILL.md. Do NOT write "understands"
        # or "listens" here: the skill's whole discipline is that it measures,
        # and the first artwork failed review for claiming AI understanding.
        bullets=[
            "Tempo, key, loudness, structure",
            "Where a track can be cut, and joined",
            "Whether the bed is burying the voice",
        ],
        meta="MIT  ·  librosa + ffmpeg  ·  runs offline",
    ),
    "paraphrasis": dict(
        out="paraphrasis-og.png",
        box=(48, 83, 865, 897),
        kicker="THE MEANING HOLDS STILL",
        headline="Everything may change except what is being claimed.",
        # Do NOT write "persuade" here: persuasion is peitho's job and is the
        # pressure that causes drift. This skill guards the claim.
        bullets=[
            "The same claim, carried to a new reader",
            "A ledger of what survived the rewrite",
            "Nine named drifts, caught before they ship",
        ],
        meta="MIT  ·  9 named drifts  ·  fidelity ledger",
    ),
}


def tracked(draw, xy, text, font, fill, tracking=0.0):
    """Draw text with letterspacing. Returns the advance width."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tracking
    return x - xy[0]


def wrap(text, font, max_w):
    lines, words, line = [], text.split(), ""
    for word in words:
        trial = f"{line} {word}".strip()
        if font.getlength(trial) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def build(cfg, src_path):
    im = Image.open(src_path).convert("RGB")

    # 1. Canvas: real paper texture from the artwork's own empty right side,
    #    mirrored to fill 2:1 without repeating a visible edge.
    #
    #    The default patch assumes the artwork is empty from x=896. Artwork that
    #    extends further right needs its own patch, or the mirror smears real
    #    content across the card: metaphrasis carries a binary field to x=906
    #    and printed stray digits down both edges before this was configurable.
    #    Any patch is resized to 640, so it need not already be square.
    tex = im.crop(cfg.get("tex", (896, 192, 1536, 832))).resize((640, H), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H))
    canvas.paste(tex, (0, 0))
    canvas.paste(tex.transpose(Image.FLIP_LEFT_RIGHT), (640, 0))

    # 2. The lockup, cropped with bleed so the feather has texture to fade into.
    BLEED, CONTENT_H = 40, 520
    x0, y0, x1, y1 = cfg["box"]
    block = im.crop((x0 - BLEED, y0 - BLEED, x1 + BLEED, y1 + BLEED))
    scale = CONTENT_H / (y1 - y0)
    bw, bh = round(block.width * scale), round(block.height * scale)
    block = block.resize((bw, bh), Image.LANCZOS)

    # 3. Feathered alpha so the two textures meet invisibly.
    f = round(BLEED * scale)
    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).rectangle([f, f, bw - f, bh - f], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(f * 0.55))

    lock_x, lock_y = 84 - f, (H - bh) // 2
    canvas.paste(block, (lock_x, lock_y), mask)

    # 4. Right column, clear of the lockup's visible edge.
    col_x = 84 + (bw - 2 * f) + 96
    col_w = W - col_x - 84
    if col_w < 260:
        raise SystemExit(f"Right column is only {col_w}px. Narrow the lockup.")

    d = ImageDraw.Draw(canvas)
    y = 148

    # Fit the kicker to the column. It is letterspaced, so it overruns long
    # before the headline does, and an overrun is invisible here and obvious on
    # someone else's timeline.
    k_size, k_track = 25, 5.5
    while k_size > 15:
        k_font = ImageFont.truetype(str(F_DISPLAY), k_size)
        if sum(k_font.getlength(c) + k_track for c in cfg["kicker"]) - k_track <= col_w:
            break
        k_size -= 1
        k_track = max(2.0, k_track - 0.35)
    tracked(d, (col_x, y), cfg["kicker"], k_font, GOLD, k_track)
    y += 44

    head_font = ImageFont.truetype(str(F_DISPLAY), 45)
    for line in wrap(cfg["headline"], head_font, col_w):
        d.text((col_x, y), line, font=head_font, fill=CREAM)
        y += 50
    y += 20

    d.line([(col_x, y), (col_x + 96, y)], fill=RULE, width=1)
    y += 32

    bullet_font = ImageFont.truetype(str(F_TEXT), 25)
    for b in cfg["bullets"]:
        d.text((col_x, y), "\u00b7", font=bullet_font, fill=GOLD)
        d.text((col_x + 22, y), b, font=bullet_font, fill=MUTED)
        y += 40

    # Bottom line carries facts, never the repo URL. This card is only ever seen
    # attached to that URL, so printing it tells the viewer where they already
    # are and spends the last legible line on nothing.
    #
    # Anchored to the bottom, but pushed down if the column ran long: the fixed
    # H-140 assumed a two-line headline, and a three-line one printed the meta
    # straight through the last bullet.
    meta_y = max(H - 140, y + 14)
    if meta_y > H - 40:
        raise SystemExit(f"Column overruns the card by {meta_y - (H - 40)}px. Shorten the headline.")
    tracked(d, (col_x, meta_y), cfg["meta"], ImageFont.truetype(str(F_MONOISH), 21), MUTED, 1.4)

    out = HERE / cfg["out"]
    canvas.save(out, optimize=True)
    print(f"wrote {out} {canvas.size} right column {col_w}px")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--source", default=str(HERE / "source-artwork.png"))
    a = ap.parse_args()
    build(CONFIGS[a.config], a.source)
