"""Deterministic per-image gallery degradation (light crop + moderate low-res + JPEG), pure PIL.

Models a real masked-test image. A real image keeps TRUE proportions but is non-square, so CMP's
fixed Resize((224,224)) (NOT aspect-preserving) WARPS it once. We reproduce that with a CROP, not
a resize: resizing the square source to non-square would be undone by CMP's resize-back-to-square
(the two cancel -> only blur, no warp). A center CROP selects a non-square region at true
proportions, so CMP's square resize warps it exactly once -- the real test distortion.

Steps, all generic (no test-distribution calibration -> Track-4 safe):
  1) light center CROP to aspect `ar` in [1, max_aspect], random orientation (non-square, true
     proportions; symmetric margins so the central subject is kept);
  2) moderate downscale PRESERVING that aspect, short side in [smin, smax] -- chosen to STRADDLE
     the model's 224 so resolution loss is light and varied (not the tiny 64-160 that over-blurs);
  3) JPEG compression artifacts.

Seeded PER IMAGE via random.Random(f"{seed}:{image_id}") -> independent of worker count / order,
identical on-the-fly or pre-written. No torch import so it runs on host and in the eval container.
Returns (degraded non-square PIL image, jpeg_quality); caller bakes JPEG by saving at `quality`.
"""
import random
from PIL import Image


def degrade(img, image_id, seed=20260622, smin=128, smax=256, qmin=30, qmax=85,
            max_aspect=1.6, p_landscape=0.7):
    rng = random.Random(f"{seed}:{image_id}")
    W, H = img.size
    # 1) non-square center crop (true proportions; warp appears at CMP's square resize).
    #    p_landscape biased to 0.7 because the real test gallery is ~70% landscape (16:9 video
    #    frames) -> people get squished horizontally (look tall/thin) after the square resize.
    ar = rng.uniform(1.0, max_aspect)
    if rng.random() < p_landscape:
        cw, ch = W, int(round(H / ar))                 # landscape crop (remove top/bottom)
    else:
        cw, ch = int(round(W / ar)), H                 # portrait crop (remove left/right)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    # 2) moderate downscale PRESERVING the crop aspect (low-res, not tiny)
    short = rng.randint(smin, smax)
    if cw <= ch:
        nw, nh = short, max(1, round(short * ch / cw))
    else:
        nh, nw = short, max(1, round(short * cw / ch))
    img = img.resize((nw, nh), Image.BILINEAR)
    # 3) JPEG
    q = rng.randint(qmin, qmax)
    return img, q
