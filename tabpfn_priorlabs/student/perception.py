"""T1.2 - RGB perception: find coloured cubes in an RGB image.

Uses only the supplied RGB pixels and requested colour names, with NumPy and SciPy.
Colour thresholds describe appearance, not cube positions. No scene configuration,
depth, simulator state, expected cube locations or previous detections are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Detection:
    color: str
    mask: np.ndarray            # (H, W) bool, pixels belonging to this cube
    centroid_px: tuple          # (u, v) of the mask centroid
    area_px: int


# All six assignment colours are supported. Hue is expressed in turns (0..1).
# Orange is a required class independent of the optional brown extension.
ASSIGNMENT_COLORS = ("red", "green", "blue", "yellow", "orange", "purple")

_HUE_RANGES = {
    "red": (0.96, 0.035),
    "orange": (0.035, 0.115),
    "yellow": (0.115, 0.20),
    "green": (0.20, 0.48),
    "blue": (0.48, 0.72),
    "purple": (0.72, 0.96),
}

#thresholds reject unwanted pixels and small regions.
#Reject grey Pixel

_MIN_SATURATION = 0.35  # Reject the grey table, robot and low-saturation tan bin.
_MIN_VALUE = 0.10       # Keep shaded cube faces; suppress hue in near-black pixels.
_MIN_AREA_PX = 20       # Reject isolated sensor noise without losing small cubes.
# Brown shares orange's hue. Classify whole blobs by median brightness only
# when the caller explicitly requests the optional brown class.
_BROWN_MAX_MEDIAN_VALUE = 0.60
# A light 3x3 cross-shaped median: centre plus four direct neighbours.
# The singleton channel dimension keeps red, green and blue separate.
_DENOISE_FOOTPRINT = ndimage.generate_binary_structure(2, 1)[..., None]
_ENABLE_DENOISING = True
_CLEANING_STRUCTURE = ndimage.generate_binary_structure(2, 1)

#IMAGE FILTERING AND MORPHOLOGICAL OPERATIONS
def _clean_color_mask(mask: np.ndarray) -> np.ndarray:

    padded = np.pad(mask, 1, mode="edge")
    seeds = ndimage.binary_opening(padded, structure=_CLEANING_STRUCTURE)
    restored = ndimage.binary_propagation(
        seeds, structure=_CLEANING_STRUCTURE, mask=padded,
    )
    closed = ndimage.binary_closing(restored, structure=_CLEANING_STRUCTURE)
    enclosed = ndimage.binary_fill_holes(restored)
    cleaned = restored | (closed & enclosed)
    return cleaned[1:-1, 1:-1]

# Cube detection uses RGB only. Intrinsics/extrinsics are used by localization,
# not imported or needed here.
def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert uint8 RGB to hue, saturation and value arrays in [0, 1]."""
    image = np.asarray(rgb, dtype=float) / 255.0 #NORMALIZE THE RGB VALUES TO [0,1] RANGE
    red, green, blue = image[..., 0], image[..., 1], image[..., 2] #separate the RGB channels from the image array
    value = image.max(axis=-1) #RETURNS MAX VALUE OF EACH PIXEL (H,W) ACROSS THE RGB CHANNELS
    chroma = value - image.min(axis=-1)
    saturation = np.divide(chroma, value, out=np.zeros_like(value), where=value > 0)
    hue = np.zeros_like(value)# INITIALIZE HUE ARRAY WITH ZEROS FOR EACH PIXEL TO FIND WHICH COLOR CHANNEL IS DOMINANT
    coloured = chroma > 0 #BOOELAN MASK 
    for channel, difference, offset in (
        (red, green - blue, 0.0),
        (green, blue - red, 2.0),
        (blue, red - green, 4.0),
    ):
        selected = coloured & (value == channel)
        hue[selected] = difference[selected] / chroma[selected] + offset
    return (hue / 6.0) % 1.0, saturation, value  #NROMALIZE HUE TO [0,1] RANGE AND RETURN HUE, SATURATION, VALUE ARRAYS


def detect_cubes(rgb: np.ndarray, colors: list[str]) -> list[Detection]:
    """
    Segment requested colours in an (H, W, 3) uint8 RGB image.

    Return one detection per 8-connected blob of at least _MIN_AREA_PX pixels.
    Centroids use image coordinates (u, v) = (column, row). Duplicate colour
    requests are processed once; unsupported colour names are ignored.
    Every call scans the full image anew, so blobs may appear anywhere or move
    between observations. Positions are measured from the current image only.
    A small spatial median suppresses isolated RGB noise before HSV conversion,
    without resizing the image or modifying the caller's RGB array.
    Morphological cleaning removes tiny islands and closes small enclosed holes.
    Input must be uint8 RGB (0..255); other data types raise TypeError.
    Optional brown is separated from orange by whole-blob brightness when
    requested. Without brown in colors, the original six-colour behaviour stays.
    In assignment runs, pass the task's requested subset of red, green, blue,
    yellow, orange and purple; dark orange remains orange without a brown request.
    RGB alone cannot separate touching cubes of the same colour.
    """
    # 1. Validate the RGB layout and the camera's uint8 pixel representation.
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if rgb.dtype != np.uint8:
        raise TypeError("rgb must have dtype uint8 with RGB values in 0..255")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0 or not colors:
        return []

    # 2. Optional light denoising, with no mixing across RGB channels.
    filtered_rgb = rgb
    if _ENABLE_DENOISING:
        filtered_rgb = ndimage.median_filter(
            rgb, footprint=_DENOISE_FOOTPRINT, mode="nearest",
        )
    # 3. Convert the full observation to HSV; no position-specific cropping.
    hue, saturation, value = _rgb_to_hsv(filtered_rgb)
    foreground = (saturation >= _MIN_SATURATION) & (value >= _MIN_VALUE)#FOR EACH PIXEL, CHECK IF SATURATION AND VALUE ARE ABOVE THE MINIMUM THRESHOLDS TO DETERMINE IF IT BELONGS TO THE FOREGROUND
    connectivity = ndimage.generate_binary_structure(2, 2)
    detections = []
    requested = dict.fromkeys(colors)
    processed = set()
    for color in requested:
        hue_color = "orange" if color == "brown" else color
        if hue_color not in _HUE_RANGES or hue_color in processed:
            continue
        processed.add(hue_color)
        # 4. Colour-specific hue thresholds plus saturation and brightness.
        low, high = _HUE_RANGES[hue_color]
        if low > high:
            in_range = (hue >= low) | (hue < high)
        else:
            in_range = (hue >= low) & (hue < high)
        # 5. Clean the binary colour mask before labelling objects.
        mask = _clean_color_mask(foreground & in_range)
        # 6. Find distinct 8-connected regions and reject small components.
        labels, count = ndimage.label(mask, structure=connectivity)
        areas = np.bincount(labels.ravel(), minlength=count + 1)
        valid = np.flatnonzero(areas[1:] >= _MIN_AREA_PX) + 1
        if not valid.size:
            continue
        # 7. Extract full-resolution masks, (u, v) centroids and pixel areas.
        centres = ndimage.center_of_mass(np.ones_like(hue), labels, valid)
        for label, (v, u) in zip(valid, centres):
            component = labels == label
            detected_color = hue_color
            if hue_color == "orange" and "brown" in requested:
                # Use the complete region so shaded orange faces do not become
                # separate brown detections. This remains lighting-dependent.
                if np.median(value[component]) <= _BROWN_MAX_MEDIAN_VALUE:
                    detected_color = "brown"
            if detected_color not in requested:
                continue
            detections.append(Detection(
                color=detected_color,
                mask=component,
                centroid_px=(float(u), float(v)),
                area_px=int(areas[label]),
            ))
    return detections
