import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_difference(tsv_path: str) -> pd.Series:
    df = pd.read_csv(tsv_path, sep="\t")
    roi_df = df[df["wordpos"] == df["ROI"]].copy()
    roi_df["meaning_type"] = roi_df["comparison"].map(
        {"expected": "homonym", "unexpected": "polyseme"}
    )
    # Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html
    # documents aggregating columns via pivot_table eg. mean surprisal by pair/meaning
    pair_df = roi_df.pivot_table(
        index="pairid",
        columns="meaning_type",
        values="adjusted_surp",
        aggfunc="mean",
    ).reset_index()
    pair_df["difference"] = pair_df["homonym"] - pair_df["polyseme"]
    return pair_df["difference"]


datasets = [
    (
        "Small",
        "results/small_homonym_minimal_pairs_byword_adjusted_GAM.tsv",
        "graphs/small_ROI_difference_distribution_GAM.png",
    ),
    (
        "Large",
        "results/large_homonym_minimal_pairs_byword_adjusted_GAM.tsv",
        "graphs/large_ROI_difference_distribution_GAM.png",
    ),
]

differences = []
for label, tsv_path, _ in datasets:
    diff = load_difference(tsv_path)
    print(f"{label} set summary:")
    print(diff.describe())
    differences.append(diff)

all_values = pd.concat(differences, ignore_index=True)
global_min = all_values.min()
global_max = all_values.max()
if global_min == global_max:
    global_min -= 0.5
    global_max += 0.5
padding = 0.05 * (global_max - global_min)
xlim = (global_min - padding, global_max + padding)
# Reference: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
# shows how to make evenly spaced bin edges with np.linspace(start, stop, num_points).
bin_edges = np.linspace(xlim[0], xlim[1], 31)

max_count = 0
for diff in differences:
    # Reference: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    counts, _ = np.histogram(diff.dropna(), bins=bin_edges)
    if len(counts):
        max_count = max(max_count, counts.max())
ylim = (0, max_count * 1.1 if max_count else 1)

# plots histogram
# Reference: https://stackoverflow.com/questions/1663807/how-do-i-iterate-over-two-lists-in-parallel
for (label, _, output_path), diff in zip(datasets, differences):
    plt.figure(figsize=(10, 6))
    sns.histplot(diff, bins=bin_edges, kde=True, color="steelblue")
    sns.rugplot(diff, color="black", alpha=0.2)

    plt.axvline(0, color="red", linestyle="--", label="Zero difference")
    plt.axvline(diff.mean(), color="green", linestyle="-", label=f"Mean = {diff.mean():.2f}")

    plt.title(f"({label}) Distribution of Homograph - Non-Homograph Adjusted Surprisal Differences")
    plt.xlabel("Difference in Adjusted Surprisal (Homograph − Non-Homograph)")
    plt.ylabel("Count")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
