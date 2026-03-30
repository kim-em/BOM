#!/usr/bin/env python3
"""
Compare BOM radar PNGs with reconstructions from NCI PVOL data.

Extracts the lowest-elevation reflectivity sweep from ODIM HDF5 volumes,
renders to 512x512 PNGs matching BOM's format, and compares pixel-by-pixel.
"""

import os
import sys
import zipfile
import tempfile
from glob import glob
from datetime import datetime

import numpy as np
import h5py
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# BOM's exact 19-entry palette (from the vol→PNG agent's analysis)
# Index 0 = transparent (no data), 1 = grey (text bg), 2 = black (text),
# 3 = white (no echo), 4-18 = reflectivity colors
BOM_PALETTE_RGB = [
    (0, 0, 0),        # 0: transparent/no data
    (192, 192, 192),   # 1: grey (text background)
    (0, 0, 0),        # 2: black (text)
    (255, 255, 255),   # 3: white (no echo / below threshold)
    (200, 200, 255),   # 4: light lavender (very light rain)
    (180, 180, 255),   # 5: lavender
    (120, 120, 255),   # 6: light blue
    (20, 20, 255),     # 7: blue
    (0, 216, 195),     # 8: cyan/teal
    (0, 150, 144),     # 9: dark teal
    (0, 210, 28),      # 10: green
    (0, 150, 0),       # 11: dark green
    (255, 255, 0),     # 12: yellow
    (255, 200, 0),     # 13: gold/amber
    (255, 150, 0),     # 14: orange
    (255, 100, 0),     # 15: dark orange
    (255, 0, 0),       # 16: red
    (200, 0, 0),       # 17: dark red
    (128, 0, 128),     # 18: purple (extreme)
]

# dBZ thresholds for each color index (approximate, needs calibration)
# These map reflectivity values to palette indices 3-18
DBZ_THRESHOLDS = [
    (-999, 0.0,  3),   # no echo -> white
    (0.0,  5.0,  4),   # very light
    (5.0,  10.0, 5),
    (10.0, 15.0, 6),
    (15.0, 20.0, 7),
    (20.0, 25.0, 8),
    (25.0, 30.0, 9),
    (30.0, 35.0, 10),
    (35.0, 40.0, 11),
    (40.0, 45.0, 12),
    (45.0, 50.0, 13),
    (50.0, 55.0, 14),
    (55.0, 60.0, 15),
    (60.0, 65.0, 16),
    (65.0, 70.0, 17),
    (70.0, 999,  18),  # extreme
]


def extract_bom_palette(png_path):
    """Extract the actual palette from a BOM PNG file."""
    img = Image.open(png_path)
    if img.mode == 'P':
        palette = img.getpalette()
        n_colors = len(set(img.getdata()))
        print(f"  Palette mode, {n_colors} unique indices used")
        for i in range(min(19, len(palette) // 3)):
            r, g, b = palette[i*3], palette[i*3+1], palette[i*3+2]
            print(f"  Index {i:2d}: ({r:3d}, {g:3d}, {b:3d})")
        return palette
    else:
        print(f"  Image mode: {img.mode} (not paletted)")
        return None


def load_pvol_sweep(h5_path, use_clean=True):
    """Load the lowest elevation reflectivity sweep from an ODIM HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Find all datasets (sweeps)
        datasets = sorted([k for k in f.keys() if k.startswith('dataset')])
        if not datasets:
            raise ValueError(f"No datasets found in {h5_path}")

        # Find the lowest elevation sweep
        best_dataset = None
        best_elev = 999
        for ds_name in datasets:
            ds = f[ds_name]
            if 'where' in ds.attrs:
                elev = ds['where'].attrs.get('elangle', 999)
            elif 'where' in ds:
                elev = ds['where'].attrs.get('elangle', 999)
            else:
                continue
            if elev < best_elev:
                best_elev = elev
                best_dataset = ds_name

        if best_dataset is None:
            # Fall back to last dataset (often lowest elevation)
            best_dataset = datasets[-1]
            print(f"  Could not determine elevation, using {best_dataset}")

        ds = f[best_dataset]
        print(f"  Using {best_dataset}, elevation: {best_elev:.1f}°")

        # Find reflectivity data
        field_name = None
        for data_key in sorted([k for k in ds.keys() if k.startswith('data')]):
            what = ds[data_key]['what']
            quantity = what.attrs.get('quantity', b'').decode() if isinstance(
                what.attrs.get('quantity', b''), bytes) else what.attrs.get('quantity', '')
            if use_clean and quantity == 'DBZH_CLEAN':
                field_name = data_key
                break
            elif quantity == 'DBZH':
                field_name = data_key

        if field_name is None:
            raise ValueError("No DBZH field found")

        data_group = ds[field_name]
        raw = data_group['data'][:]
        what = data_group['what']
        gain = what.attrs['gain']
        offset = what.attrs['offset']
        nodata = what.attrs.get('nodata', 0)
        undetect = what.attrs.get('undetect', 0)

        quantity = what.attrs.get('quantity', b'')
        if isinstance(quantity, bytes):
            quantity = quantity.decode()
        print(f"  Field: {quantity}, gain={gain}, offset={offset}")

        # Convert to dBZ
        dbz = raw.astype(np.float64) * gain + offset
        # Mask nodata and undetect
        mask = (raw == nodata) | (raw == undetect)
        dbz[mask] = np.nan

        # Get scan geometry
        where = ds['where']
        nbins = where.attrs['nbins']
        nrays = where.attrs['nrays']
        rscale = where.attrs['rscale']  # meters per bin
        rstart = where.attrs.get('rstart', 0) * 1000  # km to m? check units
        max_range = nbins * rscale

        print(f"  {nrays} rays x {nbins} bins, rscale={rscale}m, max_range={max_range/1000:.0f}km")

        return dbz, nrays, nbins, rscale, max_range


def render_to_bom_png(dbz, nrays, nbins, rscale, max_range, target_range_km=128, size=512):
    """Render polar reflectivity data to a 512x512 paletted PNG matching BOM format."""
    # Create output grid
    img_data = np.zeros((size, size), dtype=np.uint8)  # 0 = no data

    # Target range in meters
    target_range = target_range_km * 1000

    # Bins to use (may be subset if radar range > target)
    max_bin = min(nbins, int(target_range / rscale))

    # Map polar to cartesian
    center = size / 2
    scale = (size / 2) / target_range  # pixels per meter

    azimuths = np.linspace(0, 2 * np.pi, nrays, endpoint=False)

    for ray_idx in range(nrays):
        az = azimuths[ray_idx]
        sin_az = np.sin(az)
        cos_az = np.cos(az)

        for bin_idx in range(max_bin):
            range_m = (bin_idx + 0.5) * rscale
            x = center + range_m * sin_az * scale
            y = center - range_m * cos_az * scale

            ix, iy = int(x), int(y)
            if 0 <= ix < size and 0 <= iy < size:
                val = dbz[ray_idx, bin_idx]
                if np.isnan(val):
                    img_data[iy, ix] = 0  # no data
                else:
                    # Map dBZ to palette index
                    idx = 3  # default: white (below threshold)
                    for lo, hi, pidx in DBZ_THRESHOLDS:
                        if lo <= val < hi:
                            idx = pidx
                            break
                    img_data[iy, ix] = idx

    return img_data


def create_paletted_png(img_data, output_path, palette_rgb=None):
    """Create a paletted PNG with BOM's color table."""
    if palette_rgb is None:
        palette_rgb = BOM_PALETTE_RGB

    img = Image.fromarray(img_data, mode='P')
    flat_palette = []
    for r, g, b in palette_rgb:
        flat_palette.extend([r, g, b])
    # Pad to 256 entries
    flat_palette.extend([0, 0, 0] * (256 - len(palette_rgb)))
    img.putpalette(flat_palette)

    # Set index 0 as transparent
    img.save(output_path, transparency=0)
    return img


def compare_images(bom_path, reconstructed_data, bom_palette):
    """Compare BOM PNG with reconstructed data pixel by pixel."""
    bom_img = Image.open(bom_path)

    if bom_img.mode == 'P':
        bom_data = np.array(bom_img)
    else:
        bom_data = np.array(bom_img.convert('P'))

    # Only compare non-text pixels (exclude indices 1, 2 which are text overlays)
    # and only within the radar circle
    h, w = bom_data.shape
    cy, cx = h // 2, w // 2

    # Create mask: inside radar circle, not text overlay
    yy, xx = np.ogrid[:h, :w]
    circle_mask = ((xx - cx)**2 + (yy - cy)**2) <= (min(cx, cy))**2
    text_mask = (bom_data == 1) | (bom_data == 2)
    compare_mask = circle_mask & ~text_mask

    bom_masked = bom_data[compare_mask]
    recon_masked = reconstructed_data[compare_mask]

    exact_match = np.sum(bom_masked == recon_masked)
    total = len(bom_masked)
    off_by_one = np.sum(np.abs(bom_masked.astype(int) - recon_masked.astype(int)) <= 1)

    print(f"\n  Pixel comparison (within radar circle, excluding text):")
    print(f"  Total pixels compared: {total:,}")
    print(f"  Exact match: {exact_match:,} ({100*exact_match/total:.1f}%)")
    print(f"  Off by ≤1 index: {off_by_one:,} ({100*off_by_one/total:.1f}%)")

    # Per-index breakdown
    print(f"\n  Per-index accuracy:")
    for idx in range(19):
        bom_count = np.sum(bom_masked == idx)
        if bom_count > 0:
            recon_match = np.sum((bom_masked == idx) & (recon_masked == idx))
            print(f"    Index {idx:2d}: {recon_match:6d}/{bom_count:6d} ({100*recon_match/bom_count:5.1f}%)")

    return exact_match / total if total > 0 else 0


def main():
    project_dir = os.path.expanduser("~/projects/BOM")
    png_dir = os.path.join(project_dir, "pngs")
    vol_dir = os.path.join(project_dir, "nci_vol")
    out_dir = os.path.join(project_dir, "comparison_output")
    os.makedirs(out_dir, exist_ok=True)

    # Find available PVOL zip files
    vol_zips = sorted(glob(os.path.join(vol_dir, "40_*.pvol.zip")))
    if not vol_zips:
        print("No PVOL zip files found in", vol_dir)
        sys.exit(1)

    for vol_zip in vol_zips:
        basename = os.path.basename(vol_zip)
        date_str = basename.split('_')[1].split('.')[0]  # e.g. "20191103"
        print(f"\n{'='*60}")
        print(f"Processing {basename} (date: {date_str})")
        print(f"{'='*60}")

        # Find matching BOM PNGs for this date
        pattern = os.path.join(png_dir, f"IDR403.T.{date_str}*.png")
        bom_pngs = sorted(glob(pattern))
        if not bom_pngs:
            print(f"  No BOM PNGs found for date {date_str}")
            continue
        print(f"  Found {len(bom_pngs)} BOM PNGs for this date")

        # Extract BOM palette from first PNG
        print(f"\nBOM palette from {os.path.basename(bom_pngs[0])}:")
        bom_palette = extract_bom_palette(bom_pngs[0])

        # Extract PVOL zip
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\nExtracting {basename}...")
            with zipfile.ZipFile(vol_zip, 'r') as zf:
                zf.extractall(tmpdir)

            h5_files = sorted(glob(os.path.join(tmpdir, "*.h5")))
            print(f"  {len(h5_files)} volume files extracted")

            # Match timestamps
            comparisons = []
            for bom_png in bom_pngs[:10]:  # Compare up to 10 frames
                png_name = os.path.basename(bom_png)
                # Extract timestamp: IDR403.T.202603260504.png -> 202603260504
                ts = png_name.split('.T.')[1].split('.png')[0]
                # Convert to H5 naming: 40_20191103_050400.pvol.h5
                h5_ts = f"40_{ts[:8]}_{ts[8:12]}00"
                # Try exact match and nearby
                matches = [f for f in h5_files if h5_ts in os.path.basename(f)]
                if not matches:
                    # Try ±1 minute
                    dt = datetime.strptime(ts, "%Y%m%d%H%M")
                    for delta in range(-2, 3):
                        alt_dt = dt.replace(minute=dt.minute + delta) if 0 <= dt.minute + delta < 60 else dt
                        alt_ts = f"40_{alt_dt.strftime('%Y%m%d_%H%M')}00"
                        matches = [f for f in h5_files if alt_ts in os.path.basename(f)]
                        if matches:
                            break

                if matches:
                    comparisons.append((bom_png, matches[0]))

            print(f"\n  Matched {len(comparisons)} timestamp pairs")

            if not comparisons:
                # Try looser matching - just pick some from same date
                print("  Trying looser matching...")
                for i, bom_png in enumerate(bom_pngs[:5]):
                    if i < len(h5_files):
                        png_name = os.path.basename(bom_png)
                        ts = png_name.split('.T.')[1].split('.png')[0]
                        # Find h5 file closest in time
                        png_minutes = int(ts[8:10]) * 60 + int(ts[10:12])
                        best_h5 = None
                        best_diff = 999999
                        for h5f in h5_files:
                            h5_name = os.path.basename(h5f)
                            h5_time = h5_name.split('_')[2].split('.')[0]
                            h5_minutes = int(h5_time[0:2]) * 60 + int(h5_time[2:4])
                            diff = abs(h5_minutes - png_minutes)
                            if diff < best_diff:
                                best_diff = diff
                                best_h5 = h5f
                        if best_h5 and best_diff < 10:
                            comparisons.append((bom_png, best_h5))
                            print(f"    Matched {os.path.basename(bom_png)} <-> {os.path.basename(best_h5)} (Δ{best_diff}min)")

            # Run comparisons
            results = []
            for bom_png, h5_file in comparisons[:5]:  # First 5
                print(f"\n--- Comparing ---")
                print(f"  BOM: {os.path.basename(bom_png)}")
                print(f"  VOL: {os.path.basename(h5_file)}")

                try:
                    dbz, nrays, nbins, rscale, max_range = load_pvol_sweep(h5_file, use_clean=True)
                except Exception as e:
                    print(f"  Error loading volume: {e}")
                    try:
                        dbz, nrays, nbins, rscale, max_range = load_pvol_sweep(h5_file, use_clean=False)
                    except Exception as e2:
                        print(f"  Error loading DBZH too: {e2}")
                        continue

                recon_data = render_to_bom_png(dbz, nrays, nbins, rscale, max_range)

                # Save reconstructed PNG
                recon_name = f"recon_{os.path.basename(bom_png)}"
                recon_path = os.path.join(out_dir, recon_name)
                if bom_palette:
                    # Use BOM's actual palette
                    palette_rgb = []
                    for i in range(min(19, len(bom_palette) // 3)):
                        palette_rgb.append((bom_palette[i*3], bom_palette[i*3+1], bom_palette[i*3+2]))
                    create_paletted_png(recon_data, recon_path, palette_rgb)
                else:
                    create_paletted_png(recon_data, recon_path)

                # Compare
                accuracy = compare_images(bom_png, recon_data, bom_palette)
                results.append((os.path.basename(bom_png), accuracy))

                # Side-by-side comparison image
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                bom_img = Image.open(bom_png).convert('RGB')
                recon_img = Image.open(recon_path).convert('RGB')

                axes[0].imshow(bom_img)
                axes[0].set_title(f"BOM Official\n{os.path.basename(bom_png)}")
                axes[0].axis('off')

                axes[1].imshow(recon_img)
                axes[1].set_title(f"Reconstructed from PVOL\n{os.path.basename(h5_file)}")
                axes[1].axis('off')

                # Difference map
                bom_arr = np.array(Image.open(bom_png))
                diff = np.abs(bom_arr.astype(int) - recon_data.astype(int))
                axes[2].imshow(diff, cmap='hot', vmin=0, vmax=5)
                axes[2].set_title(f"Difference (palette index)\nExact match: {accuracy:.1%}")
                axes[2].axis('off')

                comp_name = f"compare_{os.path.basename(bom_png).replace('.png', '')}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, comp_name), dpi=150)
                plt.close()
                print(f"  Saved {comp_name}")

            # Summary
            if results:
                print(f"\n{'='*60}")
                print("SUMMARY")
                print(f"{'='*60}")
                for name, acc in results:
                    print(f"  {name}: {acc:.1%} exact match")
                avg = np.mean([a for _, a in results])
                print(f"\n  Average: {avg:.1%}")


if __name__ == "__main__":
    main()
