#!/usr/bin/env python3
"""
Use pyart to properly grid PVOL data and compare with BOM PNG.
"""

import os
import sys
import zipfile
import tempfile

import numpy as np
import pyart
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# BOM radar color scale (from palette extraction)
# Ordered by reflectivity (empirically: light to heavy rain)
BOM_COLORS_RGB = [
    (245/255, 245/255, 255/255),  # very faint (almost white)
    (180/255, 180/255, 255/255),  # lavender
    (120/255, 120/255, 255/255),  # light blue
    ( 20/255,  20/255, 255/255),  # blue
    (  0/255, 216/255, 195/255),  # cyan
    (  0/255, 150/255, 144/255),  # teal
    (  0/255, 102/255, 102/255),  # dark teal
    (255/255, 255/255,   0/255),  # yellow
    (255/255, 200/255,   0/255),  # gold
    (255/255, 150/255,   0/255),  # orange
    (255/255, 100/255,   0/255),  # dark orange
    (255/255,   0/255,   0/255),  # red
    (200/255,   0/255,   0/255),  # dark red
    (120/255,   0/255,   0/255),  # very dark red
]

# dBZ boundaries matching BOM's scale (standard reflectivity color table)
# BOM's color scale: wide bins for faint echoes, narrow for heavy rain
BOM_DBZ_LEVELS = [-32, -4, 4, 12, 20, 28, 33, 38, 43, 48, 53, 58, 63, 68, 80]


def main():
    project_dir = os.path.expanduser("~/projects/BOM")
    target_ts = "201911030600"

    bom_png = os.path.join(project_dir, f"pngs/IDR403.T.{target_ts}.png")
    vol_zip = os.path.join(project_dir, "nci_vol/40_20191103.pvol.zip")

    # Find matching h5 file
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(vol_zip, 'r') as zf:
        names = zf.namelist()
        target_h5 = [n for n in names if '060030' in n or '0600' in n]
        if not target_h5:
            for offset in ['0554', '0606']:
                target_h5 = [n for n in names if offset in n]
                if target_h5: break
        h5_name = target_h5[0]
        print(f"Extracting {h5_name}")
        zf.extract(h5_name, tmpdir)
        h5_path = os.path.join(tmpdir, h5_name)

    # Load with pyart
    print("Loading with pyart...")
    radar = pyart.aux_io.read_odim_h5(h5_path, file_field_names=True)
    print(f"  Fields: {list(radar.fields.keys())}")
    print(f"  Sweeps: {radar.nsweeps}")
    print(f"  Rays: {radar.nrays}")
    print(f"  Gates: {radar.ngates}")
    print(f"  Lat: {radar.latitude['data'][0]:.4f}")
    print(f"  Lon: {radar.longitude['data'][0]:.4f}")

    # Use DBZH_CLEAN if available
    field = 'DBZH_CLEAN' if 'DBZH_CLEAN' in radar.fields else 'DBZH'
    print(f"  Using field: {field}")

    # Create BOM-like colormap
    bom_cmap = ListedColormap(BOM_COLORS_RGB)
    bom_norm = BoundaryNorm(BOM_DBZ_LEVELS, bom_cmap.N)

    # Figure: side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: BOM official PNG
    bom_img = Image.open(bom_png).convert('RGB')
    axes[0].imshow(bom_img)
    axes[0].set_title(f"BOM Official\nIDR403.T.{target_ts}.png", fontsize=13)
    axes[0].axis('off')

    # Right: pyart PPI
    display = pyart.graph.RadarDisplay(radar)
    ax = axes[1]
    display.plot_ppi(field, 0, ax=ax,
                     vmin=-32, vmax=56,
                     cmap=bom_cmap, norm=bom_norm,
                     title=f"pyart from NCI PVOL\n{h5_name}",
                     colorbar_label='dBZ',
                     axislabels=('', ''))
    display.set_limits((-128, 128), (-128, 128), ax=ax)

    plt.suptitle("BOM Radar vs pyart Reconstruction (same timestamp)",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(project_dir, "side_by_side_pyart.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # Also do a cleaner version - just the radar data, matching BOM dimensions
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))

    axes2[0].imshow(bom_img)
    axes2[0].set_title("BOM Official", fontsize=14)
    axes2[0].axis('off')

    # Render PPI without axes/labels to match BOM's clean look
    display2 = pyart.graph.RadarDisplay(radar)
    display2.plot_ppi(field, 0, ax=axes2[1],
                      vmin=-32, vmax=56,
                      cmap=bom_cmap, norm=bom_norm,
                      title="Reconstructed from PVOL",
                      colorbar_flag=False,
                      axislabels=('', ''))
    display2.set_limits((-128, 128), (-128, 128), ax=axes2[1])
    axes2[1].set_aspect('equal')

    output2 = os.path.join(project_dir, "side_by_side_clean.png")
    plt.tight_layout()
    plt.savefig(output2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output2}")


if __name__ == "__main__":
    main()
