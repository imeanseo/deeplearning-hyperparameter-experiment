from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    exp_main = pd.DataFrame(
        [
            ("Exp1", "BoW Baseline", 67.38),
            ("Exp2", "BoW + Sweep", 68.26),
            ("Exp3", "TF-IDF + Sweep", 68.30),
            ("Exp4", "GloVe300 + Sweep", 62.31),
            ("Exp5", "MiniLM + Sweep", 67.82),
            ("Exp6", "MPNet + Sweep", 66.28),
        ],
        columns=["exp", "method", "test_acc"],
    )

    exp3_improve = pd.DataFrame(
        [
            ("v6", "Preprocess", 68.36),
            ("v7", "Handcrafted", 68.61),
            ("v8", "Final Sweep", 69.13),
            ("v9", "Scaled Features", 68.76),
            ("v10", "Epoch 100", 69.13),
        ],
        columns=["ver", "method", "test_acc"],
    )

    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    bars = plt.bar(exp_main["exp"], exp_main["test_acc"])
    plt.axhline(67.38, linestyle="--", linewidth=1, color="red", label="Baseline 67.38")
    plt.title("Main Experiments (Exp1~6) Test Accuracy")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(60, 70.5)
    plt.legend()
    for bar, v in zip(bars, exp_main["test_acc"]):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.subplot(1, 2, 2)
    plt.plot(exp3_improve["ver"], exp3_improve["test_acc"], marker="o")
    plt.title("Exp3 Improvements (v6~v10) Test Accuracy")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(68.2, 69.4)
    for x, y in zip(exp3_improve["ver"], exp3_improve["test_acc"]):
        plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_file = OUT_DIR / "experiment_comparison_all.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()

