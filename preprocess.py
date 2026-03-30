#!/usr/bin/env python3
"""
Preprocess BOM radar PNGs into training sequences.

IMPORTANT: BOM PNGs have INCONSISTENT palette orderings across images.
The same index number maps to different colors in different files.
We must convert to RGB first, then map colors to reflectivity.
"""

import os
import sys
from glob import glob
from datetime import datetime

import numpy as np
from PIL import Image

# --- Configuration ---

PNG_DIR = os.path.expanduser("~/projects/BOM/pngs")
OUT_DIR = os.path.expanduser("~/projects/BOM/data")
SEQ_LEN = 12
TARGET_SIZE = 128
MAX_GAP_MINUTES = 7
MIN_WEATHER_FRAC = 0.005
ARCHIVE_START = datetime(2018, 12, 3)

# --- RGB color → reflectivity mapping ---
# BOM radar colors in reflectivity order (low → high).
# Map each RGB color to a normalised reflectivity value.
# We use exact RGB matching since these are paletted PNGs (no anti-aliasing).

COLOR_TO_REFL = {
    # No data / background / text
    (0, 0, 0):       0.00,   # no data / out of range / text
    (192, 192, 192):  0.00,   # grey text background
    (255, 255, 255):  0.00,   # white (sometimes used for clear)

    # Reflectivity scale: low → high
    (245, 245, 255):  1/15,   # very faint echo
    (180, 180, 255):  2/15,   # lavender
    (120, 120, 255):  3/15,   # light blue
    (20,  20,  255):  4/15,   # blue
    (0,   216, 195):  5/15,   # cyan (the "green" ring)
    (0,   150, 144):  6/15,   # teal
    (0,   102, 102):  7/15,   # dark teal
    (255, 255, 0):    8/15,   # yellow
    (255, 200, 0):    9/15,   # gold
    (255, 150, 0):   10/15,   # orange
    (255, 100, 0):   11/15,   # dark orange
    (255, 0,   0):   12/15,   # red
    (200, 0,   0):   13/15,   # dark red
    (120, 0,   0):   14/15,   # very dark red
    (40,  0,   0):    1.0,    # extreme (if it exists)
}


def build_rgb_lut():
    """Build a lookup table: for each possible (R,G,B), map to nearest known reflectivity."""
    # For exact matches (paletted PNGs won't have intermediate colors)
    # But we need a fast lookup. Convert to a dict keyed by packed RGB.
    lut = {}
    for (r, g, b), refl in COLOR_TO_REFL.items():
        lut[(r, g, b)] = refl
    return lut


def rgb_to_reflectivity(rgb_arr, lut):
    """
    Convert (H, W, 3) uint8 RGB array to (H, W) float32 reflectivity.
    Uses exact color matching with fallback to nearest for unknown colors.
    """
    h, w, _ = rgb_arr.shape
    result = np.zeros((h, w), dtype=np.float32)

    # Vectorised: pack RGB into single int for fast lookup
    packed = rgb_arr[:, :, 0].astype(np.int32) * 65536 + \
             rgb_arr[:, :, 1].astype(np.int32) * 256 + \
             rgb_arr[:, :, 2].astype(np.int32)

    # Build packed LUT
    packed_lut = {}
    for (r, g, b), refl in lut.items():
        packed_lut[r * 65536 + g * 256 + b] = refl

    # Map each unique packed value
    for pval in np.unique(packed):
        if pval in packed_lut:
            result[packed == pval] = packed_lut[pval]
        else:
            # Unknown color - find nearest by Euclidean distance
            r, g, b = (pval >> 16) & 0xFF, (pval >> 8) & 0xFF, pval & 0xFF
            best_refl = 0.0
            best_dist = float('inf')
            for (cr, cg, cb), refl in COLOR_TO_REFL.items():
                dist = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
                if dist < best_dist:
                    best_dist = dist
                    best_refl = refl
            packed_lut[pval] = best_refl  # cache for next time
            result[packed == pval] = best_refl
            if best_dist > 0:
                print(f"    Unknown color ({r},{g},{b}) → nearest refl {best_refl:.3f} (dist={best_dist})")

    return result


def parse_timestamp(filename):
    base = os.path.basename(filename)
    ts_str = base.split('.T.')[1].split('.png')[0]
    return datetime.strptime(ts_str, "%Y%m%d%H%M")


def get_week_number(dt):
    return (dt - ARCHIVE_START).days // 7


def get_split(week_num):
    mod = week_num % 10
    if mod == 0:
        return 'val'
    elif mod == 1:
        return 'test'
    else:
        return 'train'


def has_weather(frames, threshold=MIN_WEATHER_FRAC):
    for frame in frames:
        if np.sum(frame > 0.08) / frame.size > threshold:
            return True
    return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lut = build_rgb_lut()

    png_files = sorted(glob(os.path.join(PNG_DIR, "IDR403.T.*.png")))
    print(f"Found {len(png_files)} PNG files")

    timestamps = []
    for f in png_files:
        try:
            timestamps.append((parse_timestamp(f), f))
        except ValueError:
            continue

    timestamps.sort(key=lambda x: x[0])
    print(f"Parsed {len(timestamps)} timestamps")
    print(f"Range: {timestamps[0][0]} to {timestamps[-1][0]}")

    # Load all frames as reflectivity values via RGB conversion
    print("Loading frames (RGB → reflectivity)...")
    all_refl = np.empty((len(timestamps), TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    bad_indices = set()

    for idx, (dt, fpath) in enumerate(timestamps):
        try:
            img = Image.open(fpath)
            # Convert to RGB first (this applies the palette correctly regardless of ordering)
            rgb = img.convert('RGB')
            # Downscale
            rgb_small = rgb.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
            rgb_arr = np.array(rgb_small, dtype=np.uint8)
            # Map to reflectivity
            all_refl[idx] = rgb_to_reflectivity(rgb_arr, lut)
        except Exception as e:
            print(f"  SKIP corrupt: {os.path.basename(fpath)} ({e})")
            all_refl[idx] = 0
            bad_indices.add(idx)

        if (idx + 1) % 10000 == 0:
            print(f"  {idx + 1}/{len(timestamps)} loaded")

    if bad_indices:
        print(f"  {len(bad_indices)} corrupt files skipped")
    print(f"  All frames loaded: {all_refl.nbytes / 1e9:.1f} GB")

    # Verify: check value distribution
    vals = all_refl[all_refl > 0]
    if len(vals) > 0:
        unique_vals = np.unique(np.round(vals, 4))
        print(f"  Non-zero reflectivity values: {len(vals):,} pixels")
        print(f"  Unique levels: {len(unique_vals)}")
        print(f"  Range: [{vals.min():.4f}, {vals.max():.4f}]")

    # Build sequences
    CHUNK_SIZE = 5000
    split_files = {s: [] for s in ['train', 'val', 'test']}
    split_counts = {s: 0 for s in ['train', 'val', 'test']}
    split_buffers = {s: [] for s in ['train', 'val', 'test']}
    total_seqs = 0
    weather_seqs = 0
    clear_seqs = 0
    gap_breaks = 0

    i = 0
    while i <= len(timestamps) - SEQ_LEN:
        window = timestamps[i:i + SEQ_LEN]
        has_gap = False
        for j in range(1, SEQ_LEN):
            gap = (window[j][0] - window[j-1][0]).total_seconds() / 60
            if gap > MAX_GAP_MINUTES:
                has_gap = True
                i = i + j
                gap_breaks += 1
                break

        if has_gap:
            continue

        start_week = get_week_number(window[0][0])
        end_week = get_week_number(window[-1][0])
        if get_split(start_week) != get_split(end_week):
            i += 1
            continue

        split = get_split(start_week)

        if bad_indices & set(range(i, i + SEQ_LEN)):
            i += 1
            continue

        frames = all_refl[i:i + SEQ_LEN].copy()

        if has_weather(frames):
            weather_seqs += 1
        else:
            clear_seqs += 1

        split_buffers[split].append(frames)
        split_counts[split] += 1
        total_seqs += 1

        for s in split_buffers:
            if len(split_buffers[s]) >= CHUNK_SIZE:
                chunk_idx = len(split_files[s])
                chunk_path = os.path.join(OUT_DIR, f"{s}_chunk_{chunk_idx:03d}.npz")
                np.savez_compressed(chunk_path, data=np.array(split_buffers[s], dtype=np.float32))
                split_files[s].append(chunk_path)
                split_buffers[s] = []
                print(f"  Saved {chunk_path}")

        if total_seqs % 5000 == 0:
            print(f"  {total_seqs} sequences ({weather_seqs} weather, {clear_seqs} clear) "
                  f"[train:{split_counts['train']} val:{split_counts['val']} test:{split_counts['test']}]")

        i += 1

    # Flush remaining
    for s in split_buffers:
        if split_buffers[s]:
            chunk_idx = len(split_files[s])
            chunk_path = os.path.join(OUT_DIR, f"{s}_chunk_{chunk_idx:03d}.npz")
            np.savez_compressed(chunk_path, data=np.array(split_buffers[s], dtype=np.float32))
            split_files[s].append(chunk_path)

    print(f"\nTotal sequences: {total_seqs}")
    print(f"  Weather: {weather_seqs} ({100*weather_seqs/max(total_seqs,1):.1f}%)")
    print(f"  Clear: {clear_seqs} ({100*clear_seqs/max(total_seqs,1):.1f}%)")
    print(f"  Gap breaks: {gap_breaks}")

    # Concatenate chunks
    for split_name in ['train', 'val', 'test']:
        chunks = split_files[split_name]
        if not chunks:
            print(f"  {split_name}: 0 sequences")
            continue

        all_data = np.concatenate([np.load(c)['data'] for c in chunks], axis=0)
        out_path = os.path.join(OUT_DIR, f"{split_name}.npz")
        np.savez_compressed(out_path, data=all_data)
        print(f"  {split_name}: {len(all_data)} sequences → {out_path} "
              f"({all_data.nbytes / 1e6:.0f}MB uncompressed)")

        for c in chunks:
            os.remove(c)


if __name__ == "__main__":
    main()
