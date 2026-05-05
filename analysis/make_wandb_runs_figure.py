import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main() -> None:
    """
    W&B 전체 run 성능 산점도 이미지 생성.
    사용 전: `wandb login` 필요.
    """
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb가 설치되어 있지 않습니다. `pip install wandb` 후 실행하세요.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="imeanseo_/nlp-hw1")
    parser.add_argument("--max-runs", type=int, default=200)
    args = parser.parse_args()

    project = args.project
    max_runs = args.max_runs
    api = wandb.Api(timeout=30)
    print(f"Fetching runs from {project} (max {max_runs}) ...")
    runs = api.runs(project, per_page=50)

    rows = []
    for i, r in enumerate(runs, start=1):
        if i > max_runs:
            break
        summ = dict(r.summary)
        best_dev = summ.get("best_dev_accuracy")
        test_acc = summ.get("test_accuracy")
        if best_dev is None or test_acc is None:
            continue
        rows.append(
            {
                "run_name": r.name,
                "run_id": r.id,
                "best_dev": float(best_dev) * 100.0,
                "test_acc": float(test_acc) * 100.0,
            }
        )

    print(f"Fetched {min(i if 'i' in locals() else 0, max_runs)} runs.")

    if not rows:
        raise SystemExit("best_dev_accuracy/test_accuracy가 있는 run을 찾지 못했습니다.")

    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)

    # 명백한 이상치(초기 실패 run) 때문에 축이 망가지지 않도록 2개 버전 생성
    q01_dev, q99_dev = df["best_dev"].quantile(0.01), df["best_dev"].quantile(0.99)
    q01_test, q99_test = df["test_acc"].quantile(0.01), df["test_acc"].quantile(0.99)
    core = df[
        (df["best_dev"].between(q01_dev, q99_dev))
        & (df["test_acc"].between(q01_test, q99_test))
    ].copy()

    out_dir = Path(__file__).resolve().parents[1] / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 전체 산점도 (라벨 없음, 심플)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(8.5, 6.2))
    plt.scatter(df["best_dev"], df["test_acc"], alpha=0.55, s=35, edgecolor="none")
    plt.xlabel("Best Dev Accuracy (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("W&B Runs: Best Dev vs Test (All Runs)")
    plt.tight_layout()
    out_all = out_dir / "wandb_runs_scatter_all.png"
    plt.savefig(out_all, dpi=220)
    plt.close()

    # 2) 보고서용: 이상치 제거 + 추세선 + 상위 3개만 라벨
    plt.figure(figsize=(8.8, 6.4))
    plt.scatter(core["best_dev"], core["test_acc"], alpha=0.65, s=40, edgecolor="none")

    # 회귀선(1차)
    x = core["best_dev"].to_numpy()
    y = core["test_acc"].to_numpy()
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, linestyle="--", linewidth=2, label=f"Trend (slope={m:.2f})")

    top3 = core.sort_values("test_acc", ascending=False).head(3)
    plt.scatter(top3["best_dev"], top3["test_acc"], s=70, marker="*", zorder=3, label="Top-3 runs")
    for _, row in top3.iterrows():
        plt.annotate(
            f"{row['run_name'][:14]} ({row['test_acc']:.2f})",
            (row["best_dev"], row["test_acc"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )

    plt.xlabel("Best Dev Accuracy (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("W&B Runs: Core Region with Trendline")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    out_pretty = out_dir / "wandb_runs_scatter_pretty.png"
    plt.savefig(out_pretty, dpi=220)
    plt.close()

    print(f"Saved: {out_all}")
    print(f"Saved: {out_pretty}")
    print(f"Total plotted runs (with metrics): {len(df)}")


if __name__ == "__main__":
    main()

