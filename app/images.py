"""Image normalisation: decode, fix orientation, resize for the vision API, encode."""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass

from typing import Optional

from PIL import Image, ImageChops, ImageFilter, ImageOps, UnidentifiedImageError

# Claude vision gains nothing above ~1568px on the long edge / ~1.15 megapixels;
# larger images are downscaled server-side anyway, so we do it first to save
# upload size and latency.
MAX_LONG_EDGE = 1568
MAX_PIXELS = 1_150_000
PNG_MAX_BYTES = 1_500_000
THUMB = 192


class ImageError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedImage:
    media_type: str
    data_b64: str
    sha256: str  # hash of the *original* bytes; stable cache key
    width: int
    height: int
    thumb: Image.Image  # 192x192 RGB copy for cheap similarity checks (colour matters: recolour edits)

    def content_block(self) -> dict:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": self.media_type, "data": self.data_b64},
        }


def _target_size(w: int, h: int) -> tuple[int, int]:
    scale = min(1.0, MAX_LONG_EDGE / max(w, h), (MAX_PIXELS / (w * h)) ** 0.5)
    return max(1, round(w * scale)), max(1, round(h * scale))


def prepare_image(raw: bytes) -> PreparedImage:
    if not raw:
        raise ImageError("empty image")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError(f"not a readable image: {exc}") from exc

    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if has_alpha else "RGB")

    size = _target_size(*img.size)
    if size != img.size:
        img = img.resize(size, Image.Resampling.LANCZOS)

    # PNG keeps transparency and crisp text; fall back to high-quality JPEG when large.
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    media_type = "image/png"
    if not has_alpha and buf.tell() > PNG_MAX_BYTES:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        media_type = "image/jpeg"

    thumb = img.convert("RGB").resize((THUMB, THUMB), Image.Resampling.BILINEAR)
    return PreparedImage(
        media_type=media_type,
        data_b64=base64.standard_b64encode(buf.getvalue()).decode("ascii"),
        sha256=hashlib.sha256(raw).hexdigest(),
        width=img.width,
        height=img.height,
        thumb=thumb,
    )


def near_identical(a: PreparedImage, b: PreparedImage) -> bool:
    """True when two images differ by no more than re-encoding noise.

    Uses both the mean and the max pixel difference so that a small but real
    local edit (one changed word, a recoloured button) is never reported as
    identical, while a JPEG round-trip of the same picture is.
    """
    hist = ImageChops.difference(a.thumb, b.thumb).histogram()  # 256 bins per channel
    channels = [hist[c * 256:(c + 1) * 256] for c in range(3)]
    mean = sum(i * n for ch in channels for i, n in enumerate(ch)) / (3 * THUMB * THUMB)
    peak = max(max(i for i, n in enumerate(ch) if n) for ch in channels)
    return mean < 0.5 and peak < 24


def to_pil(p: PreparedImage) -> Image.Image:
    return Image.open(io.BytesIO(base64.standard_b64decode(p.data_b64))).convert("RGB")


def difference_map(result: PreparedImage, original: PreparedImage,
                   max_edge: int = 640) -> tuple[Optional[PreparedImage], Optional[float]]:
    """Heat map of where `result` differs from `original`, and the fraction of the area that changed.

    Like the "Difference" block in a labelling tool: black = unchanged, brighter = more changed.
    Returns (None, None) when the two images have different aspect ratios (the map would be noise).
    """
    if abs(result.width / result.height - original.width / original.height) > 0.03:
        return None, None
    base = to_pil(original)
    scale = min(1.0, max_edge / max(base.size))
    size = (max(1, round(base.width * scale)), max(1, round(base.height * scale)))
    a = base.resize(size, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(1))
    b = to_pil(result).resize(size, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(1))
    r, g, bl = ImageChops.difference(a, b).split()
    diff = ImageChops.lighter(ImageChops.lighter(r, g), bl)  # strongest channel: catches pure recolours
    hist = diff.histogram()
    changed = sum(hist[30:]) / (size[0] * size[1])            # ignore re-encoding noise
    heat = diff.point(lambda v: min(255, v * 3))
    buf = io.BytesIO()
    heat.save(buf, format="PNG")
    return prepare_image(buf.getvalue()), changed
