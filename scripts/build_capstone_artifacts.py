"""
Builds every artifact the capstone paper needs, from the real starter dataset,
reusing the exact baseline (w04) and model (w05) logic already in this repo.

Outputs:
- work/outputs/baseline_action_score.csv   (w04, unchanged logic)
- work/outputs/model_vs_baseline.csv       (w05 comparison table)
- work/outputs/feature_importance.csv      (w05 top features)
- work/outputs/action_playbook.csv         (w07 - final ranked, anonymized queue)
- work/outputs/summary_stats.json          (all headline numbers used in the paper)
- work/outputs/charts/*.png                (paper figures)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "raw" / "content_refresh_anonymized.csv"
OUT = REPO / "work" / "outputs"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

# Report-style chart aesthetics: minimal, consistent, print-safe
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
INK = "#1a1a1a"
ACCENT = "#2b5fa8"
ACCENT2 = "#c65911"
GREY = "#8a8a8a"

data = pd.read_csv(DATA)
# is_declining_label is derived, not a raw column (docs/data-dictionary.md):
# 1 when trend_direction == "down" (16,262 rows = 54.2%), else 0.
data["is_declining_label"] = (data["trend_direction"] == "down").astype(int)
print("Loaded", data.shape)
print("is_declining_label positive:", int(data["is_declining_label"].sum()))

# ---------------------------------------------------------------------------
# Section A: Week-1 discovery numbers (CTR by position tier) — for the paper
# ---------------------------------------------------------------------------
visible = data[(data["impressions_90d"] > 0) & (data["avg_position"] > 0)].copy()

position_check = (
    visible.groupby("position_tier")
    .agg(mean_ctr=("ctr", "mean"), n=("ctr", "size"))
    .sort_values("mean_ctr", ascending=False)
)
print("\nCTR by position tier:\n", position_check)

# ---------------------------------------------------------------------------
# Section B: Week-4 baseline (verbatim logic from w04_baseline_score.ipynb)
# ---------------------------------------------------------------------------
baseline = visible.copy()
expected_ctr_full = baseline.groupby("position_tier")["ctr"].mean().rename("expected_ctr")
baseline = baseline.join(expected_ctr_full, on="position_tier")
baseline["ctr_gap"] = baseline["expected_ctr"] - baseline["ctr"]
baseline["opportunity_score"] = baseline["ctr_gap"].clip(lower=0) * np.log1p(baseline["impressions_90d"])
baseline["reason_code"] = np.where(baseline["opportunity_score"] > 0, "LOW_CTR_VS_POSITION", "NONE")
baseline["action"] = np.where(baseline["opportunity_score"] > 0, "REVIEW_CTR", "NO_ACTION")
baseline = baseline.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
baseline["rank"] = np.arange(1, len(baseline) + 1)

baseline_cols = [
    "rank", "content_type", "impressions_90d", "ctr", "avg_position",
    "position_tier", "expected_ctr", "ctr_gap", "opportunity_score",
    "reason_code", "action",
]
baseline_queue = baseline[baseline_cols].copy()
baseline_queue.to_csv(OUT / "baseline_action_score.csv", index=False)
print("\nWrote", OUT / "baseline_action_score.csv", len(baseline_queue), "rows")

# ---------------------------------------------------------------------------
# Section C: Week-5 model (verbatim logic from w05_model.ipynb)
# ---------------------------------------------------------------------------
target = "is_declining_label"
group = "client_id"

numeric_features = [
    "impressions_90d", "clicks_90d", "pageviews_90d", "sessions_90d", "users_90d",
    "engaged_sessions_90d", "ai_sessions_90d", "scroll_events_90d", "content_age_days",
    "days_since_last_update", "ctr", "avg_position", "engagement_rate", "scroll_rate",
    "ai_traffic_pct", "search_volume", "competition", "cpc",
]
categorical_features = [
    "content_type", "main_intent", "competition_level", "age_tier",
    "freshness_tier", "position_tier", "impression_tier",
]
numeric_features = [c for c in numeric_features if c in data.columns]
categorical_features = [c for c in categorical_features if c in data.columns]
features = numeric_features + categorical_features

X = data[features].copy()
y = data[target].astype(int)
groups = data[group]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_clients = data.iloc[train_idx][group].nunique()
test_clients = data.iloc[test_idx][group].nunique()
print("\nTrain rows:", len(X_train), "Train clients:", train_clients)
print("Test rows:", len(X_test), "Test clients:", test_clients)

numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])
model = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=10,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
pipeline.fit(X_train, y_train)

test_probability = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, test_probability)
print("\nROC-AUC:", round(roc_auc, 4))

evaluation = data.iloc[test_idx].copy()
evaluation["model_score"] = test_probability
evaluation = evaluation.sort_values("model_score", ascending=False)
model_precision_at_50 = evaluation.head(50)[target].mean()
print("Model Precision@50:", round(model_precision_at_50, 4))

train_reference = data.iloc[train_idx].copy()
expected_ctr_train = train_reference.groupby("position_tier")["ctr"].mean()
evaluation["expected_ctr"] = evaluation["position_tier"].map(expected_ctr_train)
evaluation["ctr_gap"] = (evaluation["expected_ctr"] - evaluation["ctr"]).clip(lower=0)
evaluation["baseline_score"] = evaluation["ctr_gap"] * np.log1p(evaluation["impressions_90d"])
baseline_eval_sorted = evaluation.sort_values("baseline_score", ascending=False)
baseline_precision_at_50 = baseline_eval_sorted.head(50)[target].mean()
print("Baseline Precision@50:", round(baseline_precision_at_50, 4))

base_rate = y_test.mean()
print("Base rate (test):", round(base_rate, 4))

comparison = pd.DataFrame({
    "method": ["Week-4 baseline (rule)", "Random Forest (model)"],
    "precision_at_50": [baseline_precision_at_50, model_precision_at_50],
    "roc_auc": [np.nan, roc_auc],
})
comparison.to_csv(OUT / "model_vs_baseline.csv", index=False)
print("\nWrote", OUT / "model_vs_baseline.csv")

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_
importance_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)
# clean sklearn's num__/cat__ prefixes for readability in the paper
importance_df["feature_clean"] = (
    importance_df["feature"].str.replace(r"^(num__|cat__)", "", regex=True)
)
importance_df.to_csv(OUT / "feature_importance.csv", index=False)
print("\nWrote", OUT / "feature_importance.csv")
print(importance_df[["feature_clean", "importance"]])

# ---------------------------------------------------------------------------
# Section D: Week-7 action playbook — final ranked, anonymized recommendation
# queue. Combines the transparent baseline score (primary, since it beat the
# model on this validation) with the model's probability as a secondary
# corroborating signal + reason codes. No content_id/client_id in output.
# ---------------------------------------------------------------------------
full_probability = pipeline.predict_proba(X)[:, 1]
playbook = visible.copy()
playbook["model_probability"] = pd.Series(full_probability, index=data.index).loc[playbook.index]

expected_ctr_all = playbook.groupby("position_tier")["ctr"].mean().rename("expected_ctr")
playbook = playbook.join(expected_ctr_all, on="position_tier")
playbook["ctr_gap"] = (playbook["expected_ctr"] - playbook["ctr"]).clip(lower=0)
playbook["baseline_score"] = playbook["ctr_gap"] * np.log1p(playbook["impressions_90d"])

# Normalize both scores to 0-100 so they can be combined transparently
def to_100(s):
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo == 0:
        return s * 0
    return (s - lo) / (hi - lo) * 100

playbook["baseline_score_100"] = to_100(playbook["baseline_score"])
playbook["model_score_100"] = to_100(playbook["model_probability"])
# Baseline weighted higher: it is the validated, held-out winner (see Results)
playbook["final_score"] = 0.6 * playbook["baseline_score_100"] + 0.4 * playbook["model_score_100"]

def reason_codes(row):
    codes = []
    if row["ctr_gap"] > 0:
        codes.append("low_ctr_vs_position")
    if row["model_probability"] >= 0.6:
        codes.append("model_flags_decline_risk")
    if row["impressions_90d"] >= playbook["impressions_90d"].quantile(0.75):
        codes.append("high_exposure")
    if row["days_since_last_update"] >= 180:
        codes.append("stale_content")
    if not codes:
        codes.append("no_strong_signal")
    return "|".join(codes)

playbook["reason_codes"] = playbook.apply(reason_codes, axis=1)

def suggested_action(row):
    if row["ctr_gap"] > 0 and row["model_probability"] >= 0.6:
        return "review_ctr_and_content"
    if row["ctr_gap"] > 0:
        return "review_ctr"
    if row["model_probability"] >= 0.6:
        return "monitor_decline_risk"
    return "no_action"

playbook["suggested_action"] = playbook.apply(suggested_action, axis=1)

playbook = playbook.sort_values("final_score", ascending=False).reset_index(drop=True)
playbook["rank"] = np.arange(1, len(playbook) + 1)

playbook_cols = [
    "rank", "content_type", "main_intent", "position_tier", "impression_tier",
    "age_tier", "freshness_tier", "impressions_90d", "ctr", "avg_position",
    "baseline_score_100", "model_score_100", "final_score",
    "reason_codes", "suggested_action",
]
action_playbook = playbook[playbook_cols].copy()
action_playbook.to_csv(OUT / "action_playbook.csv", index=False)
print("\nWrote", OUT / "action_playbook.csv", len(action_playbook), "rows")
print(action_playbook.head(10))

action_mix = action_playbook["suggested_action"].value_counts()
print("\nAction mix:\n", action_mix)

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

# 1. Baseline vs Random Forest — Precision@50
fig, ax = plt.subplots(figsize=(6, 4.2))
methods = comparison["method"]
values = comparison["precision_at_50"]
colors = [ACCENT2, ACCENT]
bars = ax.bar(methods, values, color=colors, width=0.5)
ax.axhline(base_rate, color=GREY, linestyle="--", linewidth=1, label=f"base rate ({base_rate:.2f})")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Precision@50 (held-out client split)")
ax.set_ylim(0, 1.0)
ax.set_title("Baseline rule vs. Random Forest, same held-out split")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(CHARTS / "precision_comparison.png", dpi=180)
plt.close()

# 2. Top feature importances
fig, ax = plt.subplots(figsize=(7, 5))
top_imp = importance_df.head(10).iloc[::-1]
ax.barh(top_imp["feature_clean"], top_imp["importance"], color=ACCENT)
ax.set_xlabel("Random Forest feature importance")
ax.set_title("What the model leans on most")
plt.tight_layout()
plt.savefig(CHARTS / "feature_importance.png", dpi=180)
plt.close()

# 3. CTR by position tier
fig, ax = plt.subplots(figsize=(6.5, 4.2))
pc = position_check.reset_index()
order = ["top_3", "page_1", "striking", "page_3_5", "deep"]
pc["position_tier"] = pd.Categorical(pc["position_tier"], categories=[o for o in order if o in pc["position_tier"].values], ordered=True)
pc = pc.sort_values("position_tier")
ax.bar(pc["position_tier"].astype(str), pc["mean_ctr"], color=ACCENT)
ax.set_ylabel("Mean CTR (%)")
ax.set_title("CTR drops sharply below the top of page 1")
plt.tight_layout()
plt.savefig(CHARTS / "ctr_by_position.png", dpi=180)
plt.close()

# 4. Ranked recommendation / action mix
fig, ax = plt.subplots(figsize=(6.5, 4.2))
action_mix_sorted = action_mix.sort_values(ascending=True)
ax.barh(action_mix_sorted.index, action_mix_sorted.values, color=[ACCENT2, ACCENT, GREY, "#444444"][:len(action_mix_sorted)])
ax.set_xlabel("Pages")
ax.set_title("Ranked review queue: recommended action mix")
plt.tight_layout()
plt.savefig(CHARTS / "action_mix.png", dpi=180)
plt.close()

# 5. Opportunity score distribution (visualizes the baseline ranking concept)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.hist(baseline_queue["opportunity_score"], bins=40, color=ACCENT, edgecolor="white", linewidth=0.3)
ax.set_xlabel("Baseline opportunity score")
ax.set_ylabel("Number of pages")
ax.set_title("Most pages score near zero; a long tail needs review")
plt.tight_layout()
plt.savefig(CHARTS / "opportunity_distribution.png", dpi=180)
plt.close()

print("\nWrote 5 charts to", CHARTS)

# ---------------------------------------------------------------------------
# Summary stats for the paper (single source of truth)
# ---------------------------------------------------------------------------
summary = {
    "rows_total": int(len(data)),
    "rows_visible": int(len(visible)),
    "n_clients": int(data["client_id"].nunique()),
    "label_positive": int(data[target].sum()),
    "label_negative": int(len(data) - data[target].sum()),
    "label_rate": float(data[target].mean()),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "train_clients": int(train_clients),
    "test_clients": int(test_clients),
    "roc_auc": float(roc_auc),
    "model_precision_at_50": float(model_precision_at_50),
    "baseline_precision_at_50": float(baseline_precision_at_50),
    "base_rate_test": float(base_rate),
    "ctr_position_correlation": float(visible["avg_position"].corr(visible["ctr"])),
    "ctr_by_position_tier": position_check["mean_ctr"].round(4).to_dict(),
    "action_mix": action_mix.to_dict(),
    "playbook_rows": int(len(action_playbook)),
    "top_features": importance_df[["feature_clean", "importance"]].head(10).round(4).to_dict("records"),
    "reproducibility_note": (
        "An earlier session of w05_model.ipynb cached Precision@50=0.82 for the model, "
        "but its data-loading cells had since been cleared, so that number could not be "
        "reproduced. Rebuilding the notebook to run top-to-bottom on the same split/seed "
        "reproduces ROC-AUC ~0.6145 (materially unchanged) but Precision@50=0.74 (baseline "
        "advantage widens from 0.02 to 0.10). This 0.74/0.84 pair is what is reported "
        "throughout the capstone; both runs agree the transparent baseline meets or beats "
        "the Random Forest."
    ),
}
with open(OUT / "summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nWrote", OUT / "summary_stats.json")
print(json.dumps(summary, indent=2))
