import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train_df.csv"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str)
    df["word_count"] = df["text"].str.split().str.len()
    df["char_count"] = df["text"].str.len()
    return df


def save_label_distribution(df: pd.DataFrame) -> None:
    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    counts = df["label"].value_counts().sort_index()
    labels = [label_names[i] for i in counts.index]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts.values)
    plt.title("Label Distribution (Train)")
    plt.ylabel("Count")
    for bar, v in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_label_distribution.png", dpi=180)
    plt.close()


def save_length_distribution(df: pd.DataFrame) -> None:
    label_names = {0: "negative", 1: "neutral", 2: "positive"}

    plt.figure(figsize=(10, 6))
    for label in [0, 1, 2]:
        subset = df[df["label"] == label]["word_count"]
        plt.hist(subset, bins=40, alpha=0.4, label=label_names[label], density=True)
    plt.title("Word Count Distribution by Label")
    plt.xlabel("Word Count")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_wordcount_by_label.png", dpi=180)
    plt.close()


def save_special_signal_plot(df: pd.DataFrame) -> None:
    rows = []
    for label, name in [(0, "negative"), (1, "neutral"), (2, "positive")]:
        sub = df[df["label"] == label]
        rows.append(
            {
                "label": name,
                "exclamation_avg": sub["text"].str.count("!").mean(),
                "question_avg": sub["text"].str.count(r"\?").mean(),
                "caps_word_avg": sub["text"].apply(
                    lambda t: sum(1 for w in str(t).split() if w.isupper() and len(w) > 1)
                ).mean(),
            }
        )
    sig = pd.DataFrame(rows).set_index("label")

    plt.figure(figsize=(10, 5))
    width = 0.25
    x = range(len(sig.index))
    plt.bar([i - width for i in x], sig["exclamation_avg"], width=width, label="Exclamation !")
    plt.bar(x, sig["question_avg"], width=width, label="Question ?")
    plt.bar([i + width for i in x], sig["caps_word_avg"], width=width, label="ALL CAPS words")
    plt.xticks(list(x), sig.index.tolist())
    plt.title("Special Signals by Label (Train)")
    plt.ylabel("Average per text")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_special_signals.png", dpi=180)
    plt.close()


def save_slang_top10(df: pd.DataFrame) -> None:
    slang_patterns = {
        "lol": r"\blol\b",
        "gonna": r"\bgonna\b",
        "wanna": r"\bwanna\b",
        "omg": r"\bomg\b",
        "idk": r"\bidk\b",
        "ur": r"\bur\b",
        "wtf": r"\bwtf\b",
        "tho": r"\btho\b",
        "kinda": r"\bkinda\b",
        "cuz": r"\bcuz\b",
    }
    counts = {}
    lower_series = df["text"].str.lower()
    for key, pat in slang_patterns.items():
        counts[key] = lower_series.str.contains(pat, regex=True).sum()
    out = pd.Series(counts).sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(out.index, out.values)
    plt.title("Top Slang Frequency (Train)")
    plt.ylabel("Count")
    for bar, v in zip(bars, out.values):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_slang_top10.png", dpi=180)
    plt.close()


def main() -> None:
    df = load_data()
    save_label_distribution(df)
    save_length_distribution(df)
    save_special_signal_plot(df)
    save_slang_top10(df)
    print(f"Saved EDA figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()

