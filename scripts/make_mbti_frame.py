"""Turn an AI-generated frame on a flat chroma-key background into a site-ready MBTI frame.

    python3 scripts/make_mbti_frame.py SOURCE.png static/images/mbti/frames/<name>.png \
        --label ENFP --color "#4fa982" [--photo static/images/you.jpg --preview preview.png]

The source should be a square frame drawn on a flat magenta background (the frame itself must
contain no magenta), with the empty window also filled with that colour. The script keys the
background out, removes colour spill, keeps only the frame, adds a label badge on the top
edge and writes a transparent PNG (512x512 by default). It prints the CSS `window` inset to paste into
`mbti_frames` in templates/components.html.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("source")
ap.add_argument("out")
ap.add_argument("--label", default="", help="badge text, e.g. ENFP (empty = no badge)")
ap.add_argument("--color", default="#4fa982", help="badge fill colour")
ap.add_argument("--photo", help="photo for a preview render")
ap.add_argument("--preview", help="where to write the preview (needs --photo)")
ap.add_argument("--size", type=int, default=512, help="output edge length in px")
ap.add_argument("--check", help="write a checkerboard/dark-background edge check image")
args = ap.parse_args()
SRC, OUT, PHOTO, PREVIEW, CHECK = args.source, args.out, args.photo, args.preview, args.check
BADGE_RGB = tuple(int(args.color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
FONT = "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"

img = np.asarray(Image.open(SRC).convert("RGB")).astype(np.float32)
H, W, _ = img.shape
R, G, B = img[..., 0], img[..., 1], img[..., 2]

# Key colour = median of the four corner patches
corners = np.concatenate([img[:40, :40].reshape(-1, 3), img[:40, -40:].reshape(-1, 3),
                          img[-160:-120, :40].reshape(-1, 3)])
K = np.median(corners, axis=0)
print("key colour", K)

# Magenta-ness: how far min(R,B) sits above G. Frame greens/creams/tans are all <= 0.
m = np.minimum(R, B) - G
k_m = min(K[0], K[2]) - K[1]
lo, hi = 0.18 * k_m, 0.75 * k_m
alpha_m = np.clip((hi - m) / (hi - lo), 0, 1)
# Distance to the key colour: keeps lavender/pink-purple art opaque, which the
# magenta-ness test alone would partly key out.
dist = np.linalg.norm(img - K, axis=-1)
alpha_d = np.clip((dist - 60) / (140 - 60), 0, 1)
alpha = np.maximum(alpha_m, alpha_d)

# Un-premultiply against the key colour, then clamp spill on edge pixels only
# (clamping everywhere would turn purple art grey)
a = np.maximum(alpha, 1e-3)[..., None]
fg = (img - (1 - a) * K) / a
fg = np.clip(fg, 0, 255)
Rf, Gf, Bf = fg[..., 0], fg[..., 1], fg[..., 2]
spill = np.minimum(Rf, Bf) - Gf
over = np.clip(spill, 0, None) * (alpha < 0.98)
fg[..., 0] -= over
fg[..., 2] -= over

# Keep only the frame (largest opaque component) -> drops watermark and texture specks
lab, n = ndimage.label(alpha > 0.5)
sizes = ndimage.sum(np.ones_like(alpha), lab, range(1, n + 1))
keep = ndimage.binary_dilation(lab == (1 + int(np.argmax(sizes))), iterations=3)
alpha = alpha * keep

# Window (transparent region connected to the centre)
hole = alpha < 0.5
hl, _ = ndimage.label(hole)
win = hl == hl[H // 2, W // 2]
ys, xs = np.where(win)
win_box = (xs.min(), ys.min(), xs.max(), ys.max())
# Outer frame bbox
ys2, xs2 = np.where(alpha > 0.05)
outer = (xs2.min(), ys2.min(), xs2.max(), ys2.max())
print("window", win_box, "outer", outer)

rgba = np.dstack([fg, alpha * 255]).astype(np.uint8)
frame = Image.fromarray(rgba, "RGBA")

# ---- label badge on the top edge, centred ----
def draw_badge(text, fill_rgb):
    S = 2  # supersample for clean edges
    pad = 20
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    font = ImageFont.truetype(FONT, 50 * S)
    spacing = 4 * S
    widths = [d0.textlength(c, font=font) for c in text]
    total = sum(widths) + spacing * (len(text) - 1)
    ph = 82
    pw = max(250, int(total / S) + 70)
    off = pad * S
    badge = Image.new("RGBA", (pw * S + 2 * off, ph * S + 2 * off), (0, 0, 0, 0))
    sh = Image.new("RGBA", badge.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([off, off + 5 * S, off + pw * S, off + ph * S + 5 * S],
                                         radius=ph * S // 2, fill=(40, 60, 50, 110))
    badge.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6 * S)))
    d = ImageDraw.Draw(badge)
    d.rounded_rectangle([off, off, off + pw * S, off + ph * S], radius=ph * S // 2,
                        fill=(255, 248, 222, 255))  # cream rim
    rim = 7 * S
    d.rounded_rectangle([off + rim, off + rim, off + pw * S - rim, off + ph * S - rim],
                        radius=(ph * S - 2 * rim) // 2, fill=fill_rgb + (255,))
    tb = d.textbbox((0, 0), text, font=font)
    ty = off + (ph * S - (tb[3] - tb[1])) / 2 - tb[1]
    x = off + (pw * S - total) / 2
    for c, w_ in zip(text, widths):
        d.text((x, ty), c, font=font, fill=(255, 255, 255, 255))
        x += w_ + spacing
    return badge.resize((badge.width // S, badge.height // S), Image.LANCZOS)


if args.label:
    band_y = (outer[1] + win_box[1]) / 2  # middle of the top frame band
    cx = (outer[0] + outer[2]) / 2
    b = draw_badge(args.label, BADGE_RGB)
    frame.alpha_composite(b, (int(cx - b.width / 2), int(band_y - b.height / 2)))

# ---- crop to square around frame, downsample to 512 ----
pad = 6
side = max(outer[2] - outer[0], outer[3] - outer[1]) + 2 * pad
left = int((outer[0] + outer[2]) / 2 - side / 2)
top = int((outer[1] + outer[3]) / 2 - side / 2)
box = (left, top, left + side, top + side)
N = args.size
final = frame.crop(box).resize((N, N), Image.LANCZOS)
# final spill clamp on faint edge pixels (resampling can reintroduce a slight cast)
fa = np.asarray(final).astype(np.float32)
sp = np.clip(np.minimum(fa[..., 0], fa[..., 2]) - fa[..., 1], 0, None)
edge = fa[..., 3] < 128
fa[..., 0] -= sp * edge
fa[..., 2] -= sp * edge
final = Image.fromarray(np.clip(fa, 0, 255).astype(np.uint8), "RGBA")
final.save(OUT, optimize=True)
scale = N / side
wb = [(win_box[0] - left) * scale, (win_box[1] - top) * scale,
      (win_box[2] - left) * scale, (win_box[3] - top) * scale]
pct = [v * 100 / N for v in wb]  # left, top, right, bottom edges in %
bleed = 1.0  # tuck the photo ~1% under the inner rim
inset = [pct[1] - bleed, 100 - pct[2] - bleed, 100 - pct[3] - bleed, pct[0] - bleed]
print('window for mbti_frames: "' + " ".join(f"{v:.0f}%" for v in inset) + '"')

# ---- verification: magenta-ish pixels among visible ones ----
arr = np.asarray(final).astype(np.float32)
vis = (arr[..., 3] > 20) & (arr[..., 3] < 200)  # edge pixels; purple art itself is fine
mm = np.minimum(arr[..., 0], arr[..., 2]) - arr[..., 1]
print("edge px with magenta cast (>25):", int((vis & (mm > 25)).sum()))

# ---- preview: photo under frame, cover-fit into window (+ small bleed under rim) ----
def compose(bg_rgba, size):
    canvas = Image.new("RGBA", (size, size), bg_rgba)
    f = final.resize((size, size), Image.LANCZOS)
    k = size / N
    bleed = 6 * k
    x0, y0, x1, y1 = [wb[0] * k - bleed, wb[1] * k - bleed, wb[2] * k + bleed, wb[3] * k + bleed]
    ph_ = Image.open(PHOTO).convert("RGBA")
    tw_, th_ = int(x1 - x0), int(y1 - y0)
    sc = max(tw_ / ph_.width, th_ / ph_.height)
    ph_ = ph_.resize((int(ph_.width * sc) + 1, int(ph_.height * sc) + 1), Image.LANCZOS)
    cl = (ph_.width - tw_) // 2
    ct = int((ph_.height - th_) * 0.25)  # bias toward top to keep the face
    ph_ = ph_.crop((cl, ct, cl + tw_, ct + th_))
    mask = Image.new("L", (tw_, th_), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw_ - 1, th_ - 1], radius=int(40 * k), fill=255)
    canvas.paste(ph_, (int(x0), int(y0)), mask)
    canvas.alpha_composite(f)
    return canvas

if PHOTO and PREVIEW:
    compose((255, 255, 255, 255), 420).convert("RGB").save(PREVIEW)

# checker + dark backgrounds side by side for edge inspection
if CHECK:
    chk = Image.new("RGBA", (N, N))
    cd = ImageDraw.Draw(chk)
    for i in range(0, N, 16):
        for j in range(0, N, 16):
            cd.rectangle([i, j, i + 15, j + 15], fill=(200, 200, 200, 255) if (i + j) // 16 % 2 else (255, 255, 255, 255))
    chk.alpha_composite(final)
    dark = Image.new("RGBA", (N, N), (30, 34, 44, 255))
    dark.alpha_composite(final)
    both = Image.new("RGB", (2 * N, N))
    both.paste(chk.convert("RGB"), (0, 0))
    both.paste(dark.convert("RGB"), (N, 0))
    both.save(CHECK)
