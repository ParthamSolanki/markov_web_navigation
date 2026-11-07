import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from markov_model import *

random.seed(42)


ensure_msnbc_dataset()
data = load_msnbc_sequences()
random.shuffle(data)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]
print(f"Total sequences: {len(data)}, Train: {len(train_data)}, Test: {len(test_data)}")

results = []

print(
    f"Running search over orders k={config.K_VALUES} and pruning thresholds tau={config.TAU_VALUES}..."
)

for k in config.K_VALUES:
    for tau in config.TAU_VALUES:
        # Build Markov model on training sequences
        model = build_markov_model(train_data, k=k, tau=tau)

        # Evaluate model and get metrics for benchmarking & comparison with other models.
        avg_ll, acc = predict_and_evaluate(
            model=model,
            test_sequences=test_data,
            k=k,
            alpha=config.ALPHA,
            num_states=config.NUM_STATES,
        )

        results.append(
            {
                "Order (k)": k,
                "Threshold (tau)": tau,
                "Active States": len(model),
                "Max Theoretical States": config.NUM_STATES**k,
                "Accuracy": acc,
                "Avg Log-Likelihood": avg_ll,
            }
        )

df_results = pd.DataFrame(results)

df_results["Accuracy"] = df_results["Accuracy"].round(4)
df_results["Avg Log-Likelihood"] = df_results["Avg Log-Likelihood"].round(4)

csv_file = config.TABLES / "markov_results.csv"
df_results.to_csv(csv_file, index=False)
print(f"Saved results table to '{csv_file}'")

sns.set_theme(style="whitegrid", font="sans-serif")

# State-Space Expansion Plot (Active States vs. Order (k))
plt.figure(figsize=(8, 5))
ax = sns.lineplot(
    data=df_results,
    x="Order (k)",
    y="Active States",
    hue="Threshold (tau)",
    marker="o",
    palette="crest",
    linewidth=2,
)
plt.yscale("log")
plt.title("State-Space Context Growth vs. Markov Order (k)")
plt.xlabel("Markov Order (k)")
plt.ylabel("Observed Contexts $|S^k_{obs}|$ (Log Scale)")
plt.xticks(config.K_VALUES)
plt.legend(title=r"Pruning $\tau$", frameon=True)

plt.savefig(config.FIGURES / "state_space_growth.svg", bbox_inches="tight")
plt.close()


# Accuracy & Log-Likelihood
fig, ax_acc = plt.subplots(figsize=(8, 5))
df_unpruned = df_results[df_results["Threshold (tau)"] == 0]

# Primary axis: Accuracy
line1 = ax_acc.plot(
    df_unpruned["Order (k)"],
    df_unpruned["Accuracy"],
    color="blue",
    marker="o",
    label="Top-1 Accuracy",
)
ax_acc.set_xlabel("Markov Order (k)")
ax_acc.set_ylabel("Top-1 Accuracy", color="blue")
ax_acc.tick_params(axis="y", labelcolor="blue")
ax_acc.set_xticks(config.K_VALUES)

# Secondary axis: Avg Log-Likelihood
ax_ll = ax_acc.twinx()
line2 = ax_ll.plot(
    df_unpruned["Order (k)"],
    df_unpruned["Avg Log-Likelihood"],
    color="red",
    marker="s",
    linestyle="--",
    label="Avg Log-Likelihood",
)
ax_ll.set_ylabel("Average Log-Likelihood", color="red")
ax_ll.tick_params(axis="y", labelcolor="red")
ax_ll.grid(False)  # Turned off twin axis grid

# Combined Legend
h1, l1 = ax_acc.get_legend_handles_labels()
h2, l2 = ax_ll.get_legend_handles_labels()
ax_acc.legend(h1 + h2, l1 + l2, loc="lower right")

plt.title("Predictive Performance Metrics vs. Order (k) [$\\tau=0$]")
fig.tight_layout()

plt.savefig(config.FIGURES / "predictive_performance.svg", bbox_inches="tight")
plt.close()


# Transition Matrix Heatmap for 1st order model
first_order_model = build_markov_model(train_data, k=1, tau=0)
trans_matrix = np.zeros((config.NUM_STATES, config.NUM_STATES))

for i in range(1, config.NUM_STATES + 1):
    counts = first_order_model.get((i,), Counter())
    denom = sum(counts.values()) + (config.ALPHA * config.NUM_STATES)
    for j in range(1, config.NUM_STATES + 1):
        trans_matrix[i - 1, j - 1] = (counts.get(j, 0) + config.ALPHA) / denom

plt.figure(figsize=(9, 8))
sns.heatmap(
    trans_matrix,
    annot=False,
    cmap="YlGnBu",
    xticklabels=config.CATEGORIES,
    yticklabels=config.CATEGORIES,
    cbar_kws={"label": r"Transition Probability $P(S_{t+1} \mid S_t)$"},
)
plt.title("First-Order State Transition Probability Matrix (Laplace Smoothed)", pad=12)
plt.xlabel("Next Category ($S_{t+1}$)")
plt.ylabel("Current Category ($S_t$)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(config.FIGURES / "transition_matrix_heatmap.svg", bbox_inches="tight")
plt.close()

print(f"All figures successfully saved to '{config.FIGURES}'.")
