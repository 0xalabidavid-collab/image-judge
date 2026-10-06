"""Generate a small labelled dataset of simple shape edits, for smoke-testing the harness.

    python scripts/make_synthetic_dataset.py --out datasets/synthetic --n 40

Each task has one correct result and one result with a single specific defect
(wrong colour, off-by-one count, misspelled text, wrong object removed,
unrequested change, edit not applied, one source image ignored). The A/B
position of the correct result is random.

These tasks are far easier and more uniform than real generation outputs.
Use them to check the plumbing (e.g. a swap-mapping bug shows up as ~50%
accuracy) - NOT as evidence of real-world accuracy.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 640, 480
COLORS = {"red": (210, 40, 40), "blue": (40, 80, 210), "green": (40, 150, 70),
          "yellow": (235, 200, 40), "purple": (130, 60, 170), "orange": (240, 130, 30)}


def canvas(bg=(250, 250, 248)):
    img = Image.new("RGB", (W, H), bg)
    return img, ImageDraw.Draw(img)


def shape(d, kind, cx, cy, r, color):
    if kind == "circle":
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif kind == "square":
        d.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif kind == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif kind == "star":
        import math
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r * 0.45
            a = math.pi / 2 + i * math.pi / 5
            pts.append((cx + rad * math.cos(a), cy - rad * math.sin(a)))
        d.polygon(pts, fill=color)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def t_recolor(rng):
    kind = rng.choice(["circle", "square", "triangle"])
    src, dst, wrong = rng.sample(list(COLORS), 3)
    def draw(color, k=kind):
        img, d = canvas(); shape(d, k, W // 2, H // 2, 110, COLORS[color]); return img
    if rng.random() < 0.5:
        bad = draw(wrong)  # wrong colour
    else:
        other = rng.choice([k for k in ("circle", "square", "triangle") if k != kind])
        bad = draw(dst, other)  # right colour, shape changed
    return f"Change the {kind} to {dst}. Keep everything else the same.", [draw(src)], draw(dst), bad


def t_count(rng):
    color = rng.choice(list(COLORS))
    n = rng.randint(2, 4)
    target = n + 1
    def draw(k):
        img, d = canvas()
        for i in range(k):
            shape(d, "star", 80 + i * (W - 160) // max(1, 5), H // 2, 45, COLORS[color])
        return img
    wrong = rng.choice([target + 1, n])
    return f"Add one more {color} star so there are exactly {target} stars in a row.", [draw(n)], draw(target), draw(wrong)


def t_text(rng):
    word = rng.choice(["SALE", "OPEN", "FRESH", "HELLO", "SUMMER", "CAFE"])
    i = rng.randrange(len(word) - 1)
    typo = word[:i] + word[i + 1] + word[i] + word[i + 2:] if word[i] != word[i + 1] else word[:-1]
    def draw(text):
        img, d = canvas(); shape(d, "circle", W // 2, H // 2 + 60, 100, COLORS["blue"])
        if text:
            d.text((W // 2, 70), text, fill=(20, 20, 20), font=font(72), anchor="mm")
        return img
    return f'Add the word "{word}" in large black letters at the top of the image.', [draw(None)], draw(word), draw(typo)


def t_remove(rng):
    a, b = rng.sample(["circle", "square", "triangle"], 2)
    ca, cb = rng.sample(list(COLORS), 2)
    def draw(show_a=True, show_b=True):
        img, d = canvas()
        if show_a: shape(d, a, W // 3, H // 2, 80, COLORS[ca])
        if show_b: shape(d, b, 2 * W // 3, H // 2, 80, COLORS[cb])
        return img
    return f"Remove the {cb} {b}. Keep the {ca} {a} exactly as it is.", [draw()], draw(True, False), draw(False, True)


def t_preserve(rng):
    kind = rng.choice(["circle", "square", "star"])
    c, c2 = rng.sample([k for k in COLORS if k != "yellow"], 2)
    def draw(bg, color):
        img, d = canvas(bg); shape(d, kind, W // 2, H // 2, 110, COLORS[color]); return img
    yellow_bg = (250, 235, 150)
    return (f"Make the background pale yellow. Do not change the {kind}.",
            [draw((250, 250, 248), c)], draw(yellow_bg, c), draw(yellow_bg, c2))


def t_unchanged(rng):
    kind = rng.choice(["circle", "square", "triangle"])
    color = rng.choice(list(COLORS))
    def draw(x):
        img, d = canvas(); shape(d, kind, x, H // 2, 70, COLORS[color]); return img
    orig = draw(W // 4)
    return f"Move the {kind} to the right side of the image.", [orig], draw(3 * W // 4), orig.copy()


def t_compose(rng):
    k1, k2 = rng.sample(["circle", "square", "triangle", "star"], 2)
    c1, c2 = rng.sample(list(COLORS), 2)
    def single(k, c):
        img, d = canvas(); shape(d, k, W // 2, H // 2, 100, COLORS[c]); return img
    good, d = canvas(); shape(d, k1, W // 3, H // 2, 90, COLORS[c1]); shape(d, k2, 2 * W // 3, H // 2, 90, COLORS[c2])
    bad, d = canvas(); shape(d, k1, W // 3, H // 2, 90, COLORS[c1]); shape(d, k1, 2 * W // 3, H // 2, 90, COLORS[c1])
    return (f"Combine the two images: put the object from image 1 on the left and the object from image 2 on the right.",
            [single(k1, c1), single(k2, c2)], good, bad)


TEMPLATES = [t_recolor, t_count, t_text, t_remove, t_preserve, t_unchanged, t_compose]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/synthetic")
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        tmpl = TEMPLATES[i % len(TEMPLATES)]
        prompt, originals, good, bad = tmpl(rng)
        tid = f"syn{i:03d}_{tmpl.__name__[2:]}"
        label = rng.choice("AB")
        a, b = (good, bad) if label == "A" else (bad, good)
        orig_paths = []
        for j, o in enumerate(originals, 1):
            p = f"images/{tid}_orig{j}.png"; o.save(out / p); orig_paths.append(p)
        a.save(out / f"images/{tid}_a.png"); b.save(out / f"images/{tid}_b.png")
        rows.append({"id": tid, "prompt": prompt, "originals": ";".join(orig_paths),
                     "result_a": f"images/{tid}_a.png", "result_b": f"images/{tid}_b.png", "correct_label": label})
    with (out / "tasks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} tasks to {out / 'tasks.csv'}")


if __name__ == "__main__":
    main()
