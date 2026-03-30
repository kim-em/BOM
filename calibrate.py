#!/usr/bin/env python3
"""
Calibrate dBZ-to-palette-index mapping by comparing same-timestamp
BOM PNGs with NCI PVOL data pixel-by-pixel.
"""

import os
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


def load_pvol_dbz(h5_path):
    """Load lowest-elevation DBZH_CLEAN data and return dBZ grid in cartesian."""
    with h5py.File(h5_path, 'r') as f:
        datasets = sorted([k for k in f.keys() if k.startswith('dataset')])

        # Find lowest elevation
        best_ds, best_elev = None, 999
        for ds_name in datasets:
            elev = f[ds_name]['where'].attrs.get('elangle', 999)
            if elev < best_elev:
                best_elev = elev
                best_ds = ds_name

        ds = f[best_ds]
        where = ds['where']
        nbins = int(where.attrs['nbins'])
        nrays = int(where.attrs['nrays'])
        rscale = float(where.attrs['rscale'])

        # Find DBZH_CLEAN, fall back to DBZH
        field_key = None
        for dk in sorted([k for k in ds.keys() if k.startswith('data')]):
            q = ds[dk]['what'].attrs.get('quantity', b'')
            if isinstance(q, bytes): q = q.decode()
            if q == 'DBZH_CLEAN':
                field_key = dk
                break
            elif q == 'DBZH' and field_key is None:
                field_key = dk

        what = ds[field_key]['what']
        raw = ds[field_key]['data'][:]
        gain = float(what.attrs['gain'])
        offset = float(what.attrs['offset'])
        nodata = what.attrs.get('nodata', 0)
        undetect = what.attrs.get('undetect', 0)

        dbz = raw.astype(np.float64) * gain + offset
        mask = (raw == nodata) | (raw == undetect)
        dbz[mask] = np.nan

    # Polar to cartesian (512x512, 128km range)
    size = 512
    target_range = 128000  # meters
    max_bin = min(nbins, int(target_range / rscale))
    center = size / 2
    scale = (size / 2) / target_range

    dbz_grid = np.full((size, size), np.nan)
    azimuths = np.linspace(0, 2 * np.pi, nrays, endpoint=False)

    for ray_idx in range(nrays):
        az = azimuths[ray_idx]
        sin_az, cos_az = np.sin(az), np.cos(az)
        for bin_idx in range(max_bin):
            range_m = (bin_idx + 0.5) * rscale
            x = int(center + range_m * sin_az * scale)
            y = int(center - range_m * cos_az * scale)
            if 0 <= x < size and 0 <= y < size:
                dbz_grid[y, x] = dbz[ray_idx, bin_idx]

    return dbz_grid


def load_bom_indices(png_path):
    """Load BOM PNG as palette index array."""
    img = Image.open(png_path)
    if img.mode != 'P':
        raise ValueError(f"Expected paletted PNG, got {img.mode}")
    return np.array(img)


def calibrate(bom_pngs, h5_files, max_pairs=20):
    """
    For each same-timestamp pair, collect (dBZ_value, bom_palette_index) samples.
    Then determine the dBZ threshold for each palette index.
    """
    all_dbz = []
    all_idx = []

    pairs_found = 0
    for bom_png in bom_pngs:
        png_name = os.path.basename(bom_png)
        ts = png_name.split('.T.')[1].split('.png')[0]  # e.g. 201911030548
        png_minutes = int(ts[8:10]) * 60 + int(ts[10:12])

        # Find closest h5 file
        best_h5, best_diff = None, 999
        for h5f in h5_files:
            h5_name = os.path.basename(h5f)
            h5_time = h5_name.split('_')[2].split('.')[0]  # e.g. 054831
            h5_minutes = int(h5_time[0:2]) * 60 + int(h5_time[2:4])
            diff = abs(h5_minutes - png_minutes)
            if diff < best_diff:
                best_diff = diff
                best_h5 = h5f

        if best_h5 is None or best_diff > 3:
            continue

        pairs_found += 1
        if pairs_found > max_pairs:
            break

        print(f"  Pair {pairs_found}: {os.path.basename(bom_png)} <-> {os.path.basename(best_h5)} (Δ{best_diff}min)")

        bom_data = load_bom_indices(bom_png)
        dbz_grid = load_pvol_dbz(best_h5)

        # Only sample pixels inside the radar circle, excluding text (idx 1,2)
        h, w = bom_data.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        circle = ((xx - cx)**2 + (yy - cy)**2) <= (min(cx, cy) - 5)**2
        valid = circle & (bom_data > 2) & ~np.isnan(dbz_grid)

        all_dbz.append(dbz_grid[valid])
        all_idx.append(bom_data[valid])

    if not all_dbz:
        print("No valid pairs found!")
        return

    dbz_all = np.concatenate(all_dbz)
    idx_all = np.concatenate(all_idx)
    print(f"\n  Total samples: {len(dbz_all):,}")

    # For each palette index, compute dBZ statistics
    print(f"\n  {'Index':>5} {'Count':>8} {'Min dBZ':>8} {'Mean dBZ':>9} {'Max dBZ':>8} {'Std':>6}")
    print(f"  {'-'*50}")

    thresholds = {}
    for idx in range(3, 19):
        mask = idx_all == idx
        count = np.sum(mask)
        if count > 0:
            vals = dbz_all[mask]
            lo, hi = np.percentile(vals, [2, 98])
            print(f"  {idx:5d} {count:8d} {np.min(vals):8.1f} {np.mean(vals):9.1f} {np.max(vals):8.1f} {np.std(vals):6.1f}  [{lo:.1f}, {hi:.1f}]")
            thresholds[idx] = (lo, np.mean(vals), hi)
        else:
            print(f"  {idx:5d} {0:8d}")

    # Also check index 0 (no data / outside range)
    mask0 = idx_all == 0
    if np.sum(mask0) > 0:
        vals0 = dbz_all[mask0]
        print(f"\n  Index 0 (transparent): {np.sum(mask0):,} samples, dBZ range [{np.min(vals0):.1f}, {np.max(vals0):.1f}]")

    # Determine clean threshold boundaries
    print("\n\n  === CALIBRATED THRESHOLDS ===")
    print("  (midpoint between adjacent index means)")
    sorted_indices = sorted(thresholds.keys())
    for i in range(len(sorted_indices) - 1):
        idx_lo = sorted_indices[i]
        idx_hi = sorted_indices[i + 1]
        boundary = (thresholds[idx_lo][1] + thresholds[idx_hi][1]) / 2
        print(f"  Index {idx_lo} -> {idx_hi} boundary: {boundary:.1f} dBZ")

    # Generate visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot of dBZ vs palette index
    subsample = np.random.choice(len(dbz_all), min(50000, len(dbz_all)), replace=False)
    axes[0].scatter(dbz_all[subsample], idx_all[subsample], alpha=0.05, s=1)
    axes[0].set_xlabel('dBZ from PVOL')
    axes[0].set_ylabel('BOM palette index')
    axes[0].set_title('dBZ vs BOM Color Index')
    axes[0].grid(True, alpha=0.3)

    # Box plot per index
    data_per_idx = []
    labels = []
    for idx in range(3, 19):
        mask = idx_all == idx
        if np.sum(mask) > 10:
            data_per_idx.append(dbz_all[mask])
            labels.append(str(idx))
    axes[1].boxplot(data_per_idx, labels=labels)
    axes[1].set_xlabel('BOM palette index')
    axes[1].set_ylabel('dBZ from PVOL')
    axes[1].set_title('dBZ Distribution per Color Index')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/projects/BOM/calibration.png'), dpi=150)
    plt.close()
    print("\n  Saved calibration.png")

    return thresholds


def main():
    project_dir = os.path.expanduser("~/projects/BOM")
    png_dir = os.path.join(project_dir, "pngs")
    vol_dir = os.path.join(project_dir, "nci_vol")

    bom_pngs = sorted(glob(os.path.join(png_dir, "IDR403.T.20191103*.png")))
    print(f"Found {len(bom_pngs)} BOM PNGs")

    # Extract volumes
    vol_zip = os.path.join(vol_dir, "40_20191103.pvol.zip")
    tmpdir = tempfile.mkdtemp()
    print(f"Extracting volumes to {tmpdir}...")
    with zipfile.ZipFile(vol_zip, 'r') as zf:
        zf.extractall(tmpdir)
    h5_files = sorted(glob(os.path.join(tmpdir, "*.h5")))
    print(f"Extracted {len(h5_files)} volumes")

    print("\nCalibrating dBZ -> palette index mapping...")
    thresholds = calibrate(bom_pngs, h5_files, max_pairs=20)


if __name__ == "__main__":
    main()
