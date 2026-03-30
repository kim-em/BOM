#!/usr/bin/env python3
"""
Best-effort reconstruction of BOM IDR403 PNG from NCI PVOL data.
Uses empirical calibration from same-timestamp comparison.
"""

import os
import sys
import zipfile
import tempfile
from glob import glob

import numpy as np
import h5py
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_pvol_dbz(h5_path):
    """Load lowest-elevation reflectivity and return polar data + metadata."""
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
        nodata_mask = (raw == nodata)
        undetect_mask = (raw == undetect)
        dbz[nodata_mask | undetect_mask] = np.nan

    return dbz, nodata_mask, undetect_mask, nrays, nbins, rscale


def polar_to_cartesian(dbz, nodata_mask, undetect_mask, nrays, nbins, rscale,
                       target_range_km=128, size=512):
    """Convert polar reflectivity to 512x512 cartesian grid."""
    target_range = target_range_km * 1000
    max_bin = min(nbins, int(target_range / rscale))
    center = size / 2.0
    scale = (size / 2.0) / target_range

    dbz_grid = np.full((size, size), np.nan)
    is_nodata = np.ones((size, size), dtype=bool)  # outside radar = nodata
    is_undetect = np.zeros((size, size), dtype=bool)

    azimuths = np.linspace(0, 2 * np.pi, nrays, endpoint=False)

    for ray_idx in range(nrays):
        az = azimuths[ray_idx]
        sin_az, cos_az = np.sin(az), np.cos(az)
        for bin_idx in range(max_bin):
            range_m = (bin_idx + 0.5) * rscale
            x = int(center + range_m * sin_az * scale)
            y = int(center - range_m * cos_az * scale)
            if 0 <= x < size and 0 <= y < size:
                is_nodata[y, x] = nodata_mask[ray_idx, bin_idx]
                is_undetect[y, x] = undetect_mask[ray_idx, bin_idx]
                dbz_grid[y, x] = dbz[ray_idx, bin_idx]

    return dbz_grid, is_nodata, is_undetect


def empirical_calibrate(bom_png_path, dbz_grid, is_nodata, is_undetect):
    """
    Empirically determine dBZ thresholds by comparing with BOM PNG.
    Returns sorted list of (palette_index, min_dbz, max_dbz).
    """
    bom = np.array(Image.open(bom_png_path))
    h, w = bom.shape
    cy, cx = h // 2, w // 2

    # Only compare inside radar circle, excluding text (indices 1, 2)
    yy, xx = np.ogrid[:h, :w]
    circle = ((xx - cx)**2 + (yy - cy)**2) <= ((min(cx, cy) - 10)**2)
    valid = circle & ~is_nodata & ~is_undetect & (bom > 2) & ~np.isnan(dbz_grid)

    bom_vals = bom[valid]
    dbz_vals = dbz_grid[valid]

    print(f"  Calibration samples: {len(dbz_vals):,}")

    # For each palette index, find the median dBZ
    index_medians = {}
    for idx in sorted(set(bom_vals)):
        mask = bom_vals == idx
        count = np.sum(mask)
        if count > 50:
            vals = dbz_vals[mask]
            median = np.median(vals)
            p10, p90 = np.percentile(vals, [10, 90])
            index_medians[idx] = (median, p10, p90, count)
            print(f"    Index {idx:2d}: median={median:6.1f} dBZ, "
                  f"p10={p10:6.1f}, p90={p90:6.1f}, n={count}")

    # Sort by median dBZ to get the reflectivity ordering
    sorted_indices = sorted(index_medians.keys(), key=lambda k: index_medians[k][0])
    print(f"  Reflectivity order: {sorted_indices}")

    # Build threshold table: boundaries at midpoints between adjacent medians
    thresholds = []
    for i, idx in enumerate(sorted_indices):
        if i == 0:
            lo = -100
        else:
            prev_idx = sorted_indices[i - 1]
            lo = (index_medians[prev_idx][0] + index_medians[idx][0]) / 2
        if i == len(sorted_indices) - 1:
            hi = 100
        else:
            next_idx = sorted_indices[i + 1]
            hi = (index_medians[idx][0] + index_medians[next_idx][0]) / 2
        thresholds.append((lo, hi, idx))

    return thresholds, index_medians


def render_with_thresholds(dbz_grid, is_nodata, is_undetect, thresholds, bom_palette):
    """Render dBZ grid to paletted PNG using calibrated thresholds."""
    h, w = dbz_grid.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Default: index 0 (transparent/no data)
    # Undetect = clear sky within range
    # Find the "clear sky" index (usually white or very light)
    # Index 17 is (255,255,255) = white, likely clear
    result[is_undetect & ~is_nodata] = 17

    # Apply thresholds for actual echoes
    has_data = ~np.isnan(dbz_grid) & ~is_nodata & ~is_undetect
    for lo, hi, idx in thresholds:
        mask = has_data & (dbz_grid >= lo) & (dbz_grid < hi)
        result[mask] = idx

    # Create paletted image
    img = Image.fromarray(result, mode='P')
    flat_pal = list(bom_palette) + [0] * (768 - len(bom_palette))
    img.putpalette(flat_pal)
    return img, result


def main():
    project_dir = os.path.expanduser("~/projects/BOM")

    # Pick a stormy timestamp
    target_ts = "201911030600"  # Peak storm
    bom_png = os.path.join(project_dir, f"pngs/IDR403.T.{target_ts}.png")

    if not os.path.exists(bom_png):
        print(f"BOM PNG not found: {bom_png}")
        sys.exit(1)

    # Get palette from BOM PNG
    bom_img = Image.open(bom_png)
    bom_palette = bom_img.getpalette()
    print(f"BOM PNG: {bom_png}")
    print(f"  Palette entries: {len(bom_palette) // 3}")

    # Extract matching PVOL
    vol_zip = os.path.join(project_dir, "nci_vol/40_20191103.pvol.zip")
    tmpdir = tempfile.mkdtemp()

    with zipfile.ZipFile(vol_zip, 'r') as zf:
        names = zf.namelist()
        # Find closest timestamp: 0600 -> look for 060031 or similar
        target_h5 = [n for n in names if '0600' in n]
        if not target_h5:
            # Try nearby
            for offset in ['0554', '0606', '0548', '0612']:
                target_h5 = [n for n in names if offset in n]
                if target_h5:
                    break

        if not target_h5:
            print("No matching PVOL found")
            sys.exit(1)

        h5_name = target_h5[0]
        print(f"PVOL: {h5_name}")
        zf.extract(h5_name, tmpdir)
        h5_path = os.path.join(tmpdir, h5_name)

    # Load PVOL data
    print("\nLoading PVOL...")
    dbz, nodata_mask, undetect_mask, nrays, nbins, rscale = load_pvol_dbz(h5_path)
    print(f"  {nrays} rays x {nbins} bins, rscale={rscale}m")

    # Convert to cartesian
    print("Projecting to cartesian...")
    dbz_grid, is_nodata, is_undetect = polar_to_cartesian(
        dbz, nodata_mask, undetect_mask, nrays, nbins, rscale)

    # Calibrate thresholds using a DIFFERENT timestamp (to avoid overfitting)
    print("\nCalibrating from nearby timestamps...")
    cal_pngs = sorted(glob(os.path.join(project_dir, "pngs/IDR403.T.20191103054*.png")))
    if not cal_pngs:
        cal_pngs = sorted(glob(os.path.join(project_dir, "pngs/IDR403.T.201911030[0-5]*.png")))[:5]

    all_thresholds = []
    for cal_png in cal_pngs[:3]:
        cal_ts = os.path.basename(cal_png).split('.T.')[1].split('.png')[0]
        cal_minutes = int(cal_ts[8:10]) * 60 + int(cal_ts[10:12])

        # Find matching h5
        with zipfile.ZipFile(vol_zip, 'r') as zf:
            names = zf.namelist()
            best_h5, best_diff = None, 999
            for n in names:
                h5_time = n.split('_')[2].split('.')[0]
                h5_min = int(h5_time[0:2]) * 60 + int(h5_time[2:4])
                diff = abs(h5_min - cal_minutes)
                if diff < best_diff:
                    best_diff = diff
                    best_h5 = n

            if best_h5 and best_diff <= 3:
                print(f"\n  Calibrating with {os.path.basename(cal_png)} <-> {best_h5}")
                zf.extract(best_h5, tmpdir)
                cal_h5_path = os.path.join(tmpdir, best_h5)
                cal_dbz, cal_nd, cal_ud, cal_nr, cal_nb, cal_rs = load_pvol_dbz(cal_h5_path)
                cal_grid, cal_nd_g, cal_ud_g = polar_to_cartesian(
                    cal_dbz, cal_nd, cal_ud, cal_nr, cal_nb, cal_rs)
                thresholds, _ = empirical_calibrate(cal_png, cal_grid, cal_nd_g, cal_ud_g)
                all_thresholds.append(thresholds)

    # Use the first calibration (or average if multiple)
    if all_thresholds:
        thresholds = all_thresholds[0]
    else:
        # Fallback
        print("WARNING: Using fallback thresholds")
        thresholds = [
            (-100, 10, 8), (10, 15, 5), (15, 20, 6), (20, 25, 4),
            (25, 30, 3), (30, 33, 7), (33, 37, 9), (37, 40, 10),
            (40, 45, 11), (45, 50, 12), (50, 55, 13), (55, 60, 14),
            (60, 65, 15), (65, 100, 16),
        ]

    # Render reconstruction
    print("\nRendering reconstruction...")
    recon_img, recon_data = render_with_thresholds(
        dbz_grid, is_nodata, is_undetect, thresholds, bom_palette)

    recon_path = os.path.join(project_dir, "best_reconstruction.png")
    recon_img.save(recon_path, transparency=0)

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    bom_rgb = Image.open(bom_png).convert('RGB')
    recon_rgb = recon_img.convert('RGB')

    axes[0].imshow(bom_rgb)
    axes[0].set_title(f"BOM Official\nIDR403.T.{target_ts}.png", fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(recon_rgb)
    axes[1].set_title(f"Reconstructed from NCI PVOL\n{h5_name}", fontsize=13)
    axes[1].axis('off')

    plt.suptitle("BOM Radar PNG vs PVOL Reconstruction", fontsize=15, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(project_dir, "side_by_side.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # Also compute match stats on the TARGET image
    print(f"\nMatch stats for target ({target_ts}):")
    bom_data = np.array(bom_img)
    h, w = bom_data.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    circle = ((xx - cx)**2 + (yy - cy)**2) <= ((min(cx, cy) - 5)**2)
    text_mask = (bom_data == 1) | (bom_data == 2)
    compare = circle & ~text_mask

    exact = np.sum(bom_data[compare] == recon_data[compare])
    total = np.sum(compare)
    print(f"  Exact match: {exact:,}/{total:,} ({100*exact/total:.1f}%)")

    # Color distance (both converted to RGB, euclidean)
    bom_arr = np.array(bom_rgb)
    recon_arr = np.array(recon_rgb)
    color_diff = np.sqrt(np.sum((bom_arr.astype(float) - recon_arr.astype(float))**2, axis=2))
    mean_diff = np.mean(color_diff[compare])
    print(f"  Mean RGB distance: {mean_diff:.1f}")


if __name__ == "__main__":
    main()
