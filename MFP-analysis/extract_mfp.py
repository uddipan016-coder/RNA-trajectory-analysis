#!/usr/bin/env python3
"""
extract_path_xtc.py

Match frames from a reference trajectory CV path and extract
corresponding frames from umbrella sampling trajectories.

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
# Core Function
# ==========================================================
def process_folder(folder_path, traj_cv, cv_columns):

    print(f"\nProcessing: {folder_path}")

    colvar_file = os.path.join(folder_path, "COLVAR_bias")
    xtc_file = os.path.join(folder_path, "md_bias_nopbc.xtc")
    topology_file = os.path.join(folder_path, "md_bias.gro")
    output_xtc = os.path.join(folder_path, "path.xtc")

    try:
        # ---- Load COLVAR CV data ----
        colvar = np.loadtxt(colvar_file, usecols=cv_columns)

        # ---- KD-tree matching ----
        tree = cKDTree(colvar)
        _, nearest_indices = tree.query(traj_cv)

        _, unique_indices = np.unique(nearest_indices, return_index=True)
        ordered_indices = [nearest_indices[i] for i in sorted(unique_indices)]

        print(f"  Matched {len(ordered_indices)} unique frames")

        # ---- Load trajectory ----
        u = mda.Universe(topology_file, xtc_file)

        # ---- Write frames ----
        with XTCWriter(output_xtc, n_atoms=u.atoms.n_atoms) as w:
            for idx in ordered_indices:
                u.trajectory[idx]
                w.write(u.atoms)

        print(f"  Saved path.xtc")

    except Exception as e:
        print(f"  ⚠️ Error: {e}")


# ==========================================================
# Main
# ==========================================================
def main():

    parser = argparse.ArgumentParser(
        description="Extract matched frames along CV path from umbrella sampling folders"
    )

    parser.add_argument("base_dir", help="Base directory containing US_* folders")
    parser.add_argument("traj_file", help="Reference trajectory CV file (.traj)")
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

    args = parser.parse_args()

    # ---- Load reference CV path ----
    traj_cv = np.loadtxt(args.traj_file)

    # ---- Find umbrella folders automatically ----
    folders = sorted(glob.glob(os.path.join(args.base_dir, args.pattern)))

    if not folders:
        raise RuntimeError("No folders found matching pattern.")

    print(f"Found {len(folders)} folders")

    # ---- Process each folder ----
    for folder_path in folders:
        process_folder(folder_path, traj_cv, args.cvcols)


if __name__ == "__main__":
    main()
