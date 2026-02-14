#!/usr/bin/env python3
"""
extract_path_gaussian.py

Extract trajectory frames near an MFEP using Gaussian sampling
around CV points and KD-tree matching.

Author: Your Name
"""

import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
from scipy.spatial import cKDTree
import os
import glob


# ==========================================================
# Core Processing
# ==========================================================
def process_folder(
    folder_path,
    traj_cv,
    cv_columns,
    n_samples,
    gaussian_std,
    query_radius,
    output_name,
):

    print(f"\nProcessing: {folder_path}")

    colvar_file = os.path.join(folder_path, "COLVAR_bias")
    xtc_file = os.path.join(folder_path, "md_bias_nopbc.xtc")
    topology_file = os.path.join(folder_path, "md_bias.gro")
    output_xtc = os.path.join(folder_path, output_name)

    try:
        # ---- Load CV data ----
        colvar = np.loadtxt(colvar_file, usecols=cv_columns)

        # ---- Build KD-tree ----
        tree = cKDTree(colvar)

        selected_indices = set()

        # ---- Gaussian sampling around MFEP ----
        dim = len(cv_columns)

        for point in traj_cv:

            gaussian_samples = np.random.normal(
                loc=point,
                scale=gaussian_std,
                size=(n_samples, dim),
            )

            for sample in gaussian_samples:
                nearby_indices = tree.query_ball_point(sample, r=query_radius)
                selected_indices.update(nearby_indices)

        selected_indices = sorted(selected_indices)

        print(f"  Selected {len(selected_indices)} frames")

        # ---- Load trajectory ----
        u = mda.Universe(topology_file, xtc_file)

        # ---- Write output trajectory ----
        with XTCWriter(output_xtc, n_atoms=u.atoms.n_atoms) as w:
            for idx in selected_indices:
                u.trajectory[idx]
                w.write(u.atoms)

        print(f"  ✅ Saved {output_name}")

    except Exception as e:
        print(f"  ⚠️ Error: {e}")


# ==========================================================
# Main
# ==========================================================
def main():

    parser = argparse.ArgumentParser(
        description="Extract Gaussian-sampled neighborhood frames along MFEP"
    )

    parser.add_argument("base_dir", help="Base directory containing US folders")
    parser.add_argument("traj_file", help="MFEP CV trajectory file")

    parser.add_argument(
        "--cvcols",
        nargs="+",
        type=int,
        default=[1, 2],
        help="CV column indices (0-based, default: 1 2)",
    )

    parser.add_argument(
        "--pattern",
        default="US_*",
        help="Folder pattern (default: US_*)",
    )

    parser.add_argument(
        "--nsamples",
        type=int,
        default=10,
        help="Gaussian samples per MFEP point (default: 10)",
    )

    parser.add_argument(
        "--std",
        type=float,
        default=0.01,
        help="Gaussian standard deviation (default: 0.01)",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=0.01,
        help="KD-tree query radius (default: 0.01)",
    )

    parser.add_argument(
        "--out",
        default="path_gaussian.xtc",
        help="Output trajectory name",
    )

    args = parser.parse_args()

    # ---- Load MFEP CV path ----
    traj_cv = np.loadtxt(args.traj_file)

    # ---- Detect umbrella folders automatically ----
    folders = sorted(glob.glob(os.path.join(args.base_dir, args.pattern)))

    if not folders:
        raise RuntimeError("No folders found matching pattern.")

    print(f"Found {len(folders)} folders")

    # ---- Process each folder ----
    for folder_path in folders:
        process_folder(
            folder_path,
            traj_cv,
            args.cvcols,
            args.nsamples,
            args.std,
            args.radius,
            args.out,
        )


if __name__ == "__main__":
    main()
