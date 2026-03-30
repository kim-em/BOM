#!/usr/bin/env python3
"""
Fetch latest radar frames, run nowcasting model, output predicted PNGs.

Usage:
    python3 predict.py --output-dir /tmp/radar-predictions

Outputs:
    /tmp/radar-predictions/pred_1.png through pred_6.png
    (composited with BOM overlays, 512x512, ready to display)
"""

import argparse
import os
import sys
import re
from io import BytesIO

import numpy as np
import torch
from PIL import Image
import urllib.request

# --- Config ---

BASE_URL = "https://reg.bom.gov.au"
RADAR_ID = "IDR403"
MODEL_PATH = os.path.expanduser("~/projects/BOM/checkpoints/convgru/best.pt")
TARGET_SIZE = 128
OUTPUT_SIZE = 512

# Same color mapping as preprocess.py
COLOR_TO_REFL = {
    (0, 0, 0):       0.00,
    (192, 192, 192):  0.00,
    (255, 255, 255):  0.00,
    (245, 245, 255):  1/15,
    (180, 180, 255):  2/15,
    (120, 120, 255):  3/15,
    (20,  20,  255):  4/15,
    (0,   216, 195):  5/15,
    (0,   150, 144):  6/15,
    (0,   102, 102):  7/15,
    (255, 255, 0):    8/15,
    (255, 200, 0):    9/15,
    (255, 150, 0):   10/15,
    (255, 100, 0):   11/15,
    (255, 0,   0):   12/15,
    (200, 0,   0):   13/15,
    (120, 0,   0):   14/15,
    (40,  0,   0):    1.0,
}

# Reverse mapping: reflectivity value → RGB color (for output)
REFL_COLORS = [
    (0.00,  (0, 0, 0)),         # no data
    (1/15,  (245, 245, 255)),
    (2/15,  (180, 180, 255)),
    (3/15,  (120, 120, 255)),
    (4/15,  (20, 20, 255)),
    (5/15,  (0, 216, 195)),
    (6/15,  (0, 150, 144)),
    (7/15,  (0, 102, 102)),
    (8/15,  (255, 255, 0)),
    (9/15,  (255, 200, 0)),
    (10/15, (255, 150, 0)),
    (11/15, (255, 100, 0)),
    (12/15, (255, 0, 0)),
    (13/15, (200, 0, 0)),
    (14/15, (120, 0, 0)),
]


def fetch_frame_paths():
    """Get list of radar frame paths from BOM loop page."""
    url = f"{BASE_URL}/products/{RADAR_ID}.loop.shtml"
    data = urllib.request.urlopen(url, timeout=15).read().decode()
    pattern = rf"/radar/{RADAR_ID}\.T\.\d+\.png"
    return re.findall(pattern, data)


def fetch_image(path):
    """Fetch a PNG from BOM and return as PIL Image."""
    url = f"{BASE_URL}{path}"
    data = urllib.request.urlopen(url, timeout=15).read()
    return Image.open(BytesIO(data))


def fetch_overlays():
    """Fetch background/topography/locations/range overlay images."""
    names = ["background", "topography", "locations", "range"]
    overlays = []
    for name in names:
        url = f"{BASE_URL}/products/radar_transparencies/{RADAR_ID}.{name}.png"
        data = urllib.request.urlopen(url, timeout=15).read()
        overlays.append(Image.open(BytesIO(data)).convert("RGBA"))
    return overlays


def rgb_to_reflectivity(rgb_img):
    """Convert RGB PIL Image to 128x128 reflectivity array."""
    small = rgb_img.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    arr = np.array(small, dtype=np.uint8)

    # Build packed LUT
    packed_lut = {}
    for (r, g, b), refl in COLOR_TO_REFL.items():
        packed_lut[r * 65536 + g * 256 + b] = refl

    packed = arr[:, :, 0].astype(np.int32) * 65536 + \
             arr[:, :, 1].astype(np.int32) * 256 + \
             arr[:, :, 2].astype(np.int32)

    result = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    for pval in np.unique(packed):
        if pval in packed_lut:
            result[packed == pval] = packed_lut[pval]
        else:
            # Nearest color match
            r, g, b = (pval >> 16) & 0xFF, (pval >> 8) & 0xFF, pval & 0xFF
            best_refl, best_dist = 0.0, float('inf')
            for (cr, cg, cb), refl in COLOR_TO_REFL.items():
                dist = (r-cr)**2 + (g-cg)**2 + (b-cb)**2
                if dist < best_dist:
                    best_dist = dist
                    best_refl = refl
            packed_lut[pval] = best_refl
            result[packed == pval] = best_refl

    return result


def reflectivity_to_rgba(refl_frame, size=OUTPUT_SIZE):
    """Convert 128x128 reflectivity to 512x512 RGBA image."""
    h, w = refl_frame.shape

    # Apply radar circle mask: clamp predictions inside circle,
    # keep outside as transparent
    center = h / 2.0
    radius = center - 1
    yy, xx = np.ogrid[:h, :w]
    inside_circle = ((xx - center)**2 + (yy - center)**2) <= radius**2

    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Only draw rain colors for values above the first rain threshold (1/15)
    first_threshold = 1/15
    for threshold, (r, g, b) in REFL_COLORS:
        if threshold < first_threshold:
            continue  # skip the black "no data" entry
        mask = (refl_frame >= threshold) & inside_circle
        rgba[mask] = [r, g, b, 255]

    # Everything below first rain threshold (or outside circle) stays transparent

    img = Image.fromarray(rgba, 'RGBA')
    return img.resize((size, size), Image.NEAREST)


def composite(radar_rgba, overlays):
    """Composite radar data with BOM overlays."""
    size = radar_rgba.size[0]
    result = Image.new('RGBA', (size, size), (0, 0, 0, 255))

    # Background layers
    for overlay in overlays[:2]:
        ov = overlay.resize((size, size), Image.NEAREST)
        result = Image.alpha_composite(result, ov)

    # Radar data
    result = Image.alpha_composite(result, radar_rgba)

    # Foreground overlays (locations, range)
    for overlay in overlays[2:]:
        ov = overlay.resize((size, size), Image.NEAREST)
        result = Image.alpha_composite(result, ov)

    return result.convert('RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/radar-predictions')
    parser.add_argument('--model', default=MODEL_PATH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Fetch latest frames
    print("Fetching radar frames...", file=sys.stderr)
    frame_paths = fetch_frame_paths()
    if len(frame_paths) < 6:
        print(f"Only {len(frame_paths)} frames available, need 6", file=sys.stderr)
        sys.exit(1)

    # Use the latest 6
    latest_paths = frame_paths[-6:]
    frames_rgb = [fetch_image(p).convert('RGB') for p in latest_paths]
    print(f"Fetched {len(frames_rgb)} frames", file=sys.stderr)

    # Fetch overlays
    print("Fetching overlays...", file=sys.stderr)
    overlays = fetch_overlays()

    # Convert to reflectivity
    refl_frames = np.array([rgb_to_reflectivity(f) for f in frames_rgb])
    input_tensor = torch.from_numpy(refl_frames).unsqueeze(0)  # (1, 6, 128, 128)

    # Load model and run inference
    print("Running model...", file=sys.stderr)
    from models.convgru import ConvGRUNet
    model = ConvGRUNet()
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    model.eval()

    with torch.no_grad():
        predictions = model(input_tensor).numpy()[0]  # (6, 128, 128)

    # Convert predictions to composited PNGs
    print("Generating output PNGs...", file=sys.stderr)
    for i in range(6):
        radar_rgba = reflectivity_to_rgba(predictions[i])
        composited = composite(radar_rgba, overlays)
        out_path = os.path.join(args.output_dir, f"pred_{i+1}.png")
        composited.save(out_path)

    # Also output the frame timestamps for the app
    print(",".join(latest_paths))  # stdout: comma-separated input frame paths
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main()
