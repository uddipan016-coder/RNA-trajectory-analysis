!/usr/bin/env python3
"""
Flexible PLUMED-aware TICA analysis.

Features:
- Reads CV names from COLVAR header (# ! FIELDS)
- User selects CVs by name OR index
- Generic fallback labels if header absent
"""

import argparse
import numpy as np
import pyemma
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ==========================================================
# COLVAR Parsing
# ==========================================================
def parse_colvar(filepath):
    """
    Load COLVAR file and extract CV names from header.
    """

    cv_names = None

    with open(filepath) as f:
        for line in f:
            if line.startswith("#!") and "FIELDS" in line:
                fields = line.strip().split()[2:]
                cv_names = fields[1:]  # remove time column
                break

    data = np.loadtxt(filepath, comments="#")

    total_cvs = data.shape[1] - 1

    if cv_names is None:
        cv_names = [f"CV{i+1}" for i in range(total_cvs)]

    return data, cv_names


# ==========================================================
# CV Selection
# ==========================================================
def select_cvs(data, cv_names, selected):
    """
    Select CVs using names or indices.
    """

    total_cvs = len(cv_names)

    # Convert indices if numeric
    indices = []

    for item in selected:

        # Try integer index
        if item.isdigit():
            idx = int(item) - 1
            if idx < 0 or idx >= total_cvs:
                raise ValueError(
                    f"CV index {item} out of range (1-{total_cvs})."
                )
            indices.append(idx)

        else:
            if item not in cv_names:
                raise ValueError(
                    f"CV name '{item}' not found in COLVAR."
                )
            indices.append(cv_names.index(item))

    # Remove duplicates while preserving order
    indices = list(dict.fromkeys(indices))

    selected_data = data[:, 1:][:, indices]
    selected_names = [cv_names[i] for i in indices]

    return selected_data, selected_names


# ==========================================================
# TICA
# ==========================================================
def run_tica(data, lag):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    tica = pyemma.coordinates.tica(data_scaled, lag=lag)
    return tica


def compute_contributions(eigenvectors):
    tic1 = eigenvectors[:, 0]
    tic2 = eigenvectors[:, 1]

    tic1_contrib = (tic1**2 / np.sum(tic1**2)) * 100
    tic2_contrib = (tic2**2 / np.sum(tic2**2)) * 100

    return tic1_contrib, tic2_contrib


# ==========================================================
# Plot
# ==========================================================
def plot_contributions(tic1, tic2, labels, outfile):

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.bar(x - width/2, tic1, width,
           label="TIC 1", edgecolor="black", linewidth=1.2)
    ax.bar(x + width/2, tic2, width,
           label="TIC 2", edgecolor="black", linewidth=1.2)

    ax.set_ylabel("Contribution (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ==========================================================
# Main
# ==========================================================
def main():

    parser = argparse.ArgumentParser(
        description="PLUMED-aware flexible TICA analysis"
    )

    parser.add_argument("colvar", help="COLVAR file")
    parser.add_argument("--cvs", nargs="+", required=True,
                        help="CVs to use (names or indices)")
    parser.add_argument("--lag", type=int, default=10)
    parser.add_argument("--out", default="tica_contributions.png")

    args = parser.parse_args()

    # ---- Load COLVAR ----
    data, cv_names = parse_colvar(args.colvar)

    print("Detected CVs:", ", ".join(cv_names))

    # ---- Select CVs ----
    selected_data, selected_names = select_cvs(
        data,
        cv_names,
        args.cvs
    )

    print("Using CVs:", ", ".join(selected_names))

    # ---- Run TICA ----
    tica = run_tica(selected_data, args.lag)

    tic1, tic2 = compute_contributions(tica.eigenvectors)

    # ---- Plot ----
    plot_contributions(tic1, tic2, selected_names, args.out)

    print("✅ Analysis finished.")
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()
