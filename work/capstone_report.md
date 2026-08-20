# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** Sukrit
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/SUKRIT004/flyrank-ml-internship
- **Date:** 2026-08-20

## 0. Abstract

Can search-performance and content signals identify visible pages that deserve CTR or
engagement review? Using 28,795 visible, pseudonymized content items across 32 clients from
the FlyRank ML Internship starter release, a transparent baseline rule (CTR gap versus a
page's search-position tier, weighted by exposure) is compared against a Random Forest
classifier under a client-level held-out split. The baseline reaches Precision@50 of 0.84
against the model's 0.74 (ROC-AUC 0.6145, base rate 0.51) — the simpler, fully-readable rule
outperforms the more complex model on the metric that matters for this decision. The output is
a ranked, reason-coded review queue intended as decision support for a content/SEO reviewer,
not a causal claim about what improves rankings.

## 1. Problem framing

**Decision supported:** out of thousands of visible pages, which ones should a content/SEO
reviewer look at first for a possible CTR or engagement fix? **Unit of analysis:** one content
item (a page), as of its 90-day trailing snapshot. **Output:** a ranked list (`final_score`,
0–100) with plain-language reason codes and a suggested action per page. **Action a human
takes:** open the top-ranked pages, sanity-check the reason codes against the actual page
(query intent, recent changes, seasonality), and decide whether to rewrite, monitor, or leave
it. **Cost of a wrong call:** a false positive wastes a reviewer's time on a page that's
actually fine; a false negative leaves a genuinely under-performing page unreviewed for
another cycle. Neither is catastrophic, which is why this is framed as triage support rather
than an automated action. **Why ML/data helps here at all:** with tens of thousands of pages
and no way to manually audit all of them, a repeatable score that surfaces the highest-signal
candidates first is strictly better than reviewing in arbitrary order — even a simple,
transparent rule beats no prioritization.

## 2. Data safety

**Source used:** `data/raw/content_refresh_anonymized.csv` — 30,000 rows, one row per
pseudonymized content item, 32 pseudonymized clients, trailing-90-day metrics. No client
names, domains, URLs, or private query text exist in this file or anywhere it was joined.

**Deliberately excluded columns and why:**
- `trend_direction`, `trend_pct` — the target `is_declining_label` is *derived* from
  `trend_direction == "down"`. Including either as a feature would leak the label into the
  model (the model would learn to read the answer off the label's own source column).
- `content_id`, `client_id` — identifiers, not measurements. `client_id` is used only to
  *group* the train/test split, never passed to the model as a feature.

**Leakage risks considered:** the label-derivation leak above was the primary risk in this
lane, since `trend_direction`/`trend_pct` are semantically adjacent to the target and easy to
include by accident. Checked by asserting, in code, that the forbidden set
`{is_declining_label, trend_direction, trend_pct, content_id, client_id}` has no overlap with
the final feature list before training (see `work/notebooks/capstone.ipynb`, Section 3). A
second, lower risk is pseudonymous IDs leaking through the *split* rather than the features —
addressed by grouping the split on `client_id` so no client's pages appear in both train and
test.

**Confirmed nothing client-identifying appears in `work/`:** `grep`-checked
`work/outputs/action_playbook.csv` and `docs/index.html` for `content_id`/`client_id` values,
API keys, tokens, and secret-like patterns — clean. The playbook export contains only
content-type/signal columns (position tier, impressions, CTR, reason codes), never row-level
identifiers.

## 3. Baseline

The Week-4 baseline is a transparent rule with no fitted weights:

```
expected_ctr      = mean CTR for the page's position_tier
ctr_gap           = max(0, expected_ctr - ctr)
opportunity_score = ctr_gap * log1p(impressions_90d)
```

It's a fair comparison because it uses the same visible-page population (28,795 rows), the
same evaluation metric (Precision@50), and the same held-out split as the model — nothing
about the baseline is tuned on the test set. Its reason code (`LOW_CTR_VS_POSITION`) makes
every ranking auditable by a human, which the model's probability score does not offer on its
own. On the held-out split: **Precision@50 = 0.84**.

## 4. Model / analysis

**Method:** Random Forest classifier (`n_estimators=200, max_depth=12, min_samples_leaf=10,
class_weight="balanced", random_state=42`). Chosen because the lane's signals are a mix of
numeric traffic/engagement metrics and categorical content attributes with plausible
non-linear interactions (e.g. content age matters differently at different position tiers) —
a tree ensemble handles that mix without manual feature engineering, and gives inspectable
feature importances.

**Target:** `is_declining_label` — 1 when `trend_direction == "down"` (16,262 of 30,000 rows,
54.2%), 0 otherwise.

**Feature list (25 total):**
`impressions_90d, clicks_90d, pageviews_90d, sessions_90d, users_90d, engaged_sessions_90d,
ai_sessions_90d, scroll_events_90d, content_age_days, days_since_last_update, ctr,
avg_position, engagement_rate, scroll_rate, ai_traffic_pct, search_volume, competition, cpc,
content_type, main_intent, competition_level, age_tier, freshness_tier, position_tier,
impression_tier`

**Left out on purpose:** `trend_direction`, `trend_pct` (label-derived — see Section 2),
`content_id`, `client_id` (identifiers, not measurements).

## 5. Evaluation

**Split:** `GroupShuffleSplit` grouped by `client_id`, 20% of clients held out, seed fixed at
42. Grouped rather than random because a random row-level split would let the model see other
pages from the same client in training, letting it memorize client-specific quirks instead of
learning patterns that generalize to a client it's never seen — which is the realistic
deployment scenario (scoring a new client's pages). Not time-aware, since this is a single
snapshot rather than a longitudinal release.

**Split sizes:** train 23,837 rows / 25 clients; test 6,163 rows / 7 clients. Test-set base
rate: 0.511 (majority class alone would get ~51% precision by chance at any K).

| Method | Precision@50 | ROC-AUC |
|---|---:|---:|
| Week-4 baseline (rule) | **0.84** | — |
| Random Forest (model) | 0.74 | 0.6145 |

The baseline beats the model by 0.10 Precision@50 on the identical held-out rows.

**Error analysis:** the Random Forest's ROC-AUC of 0.6145 shows some overall ability to
separate declining from non-declining pages, but that broad discrimination didn't translate
into a better *top-50* ranking than the simple rule — the model's highest-confidence pages
weren't concentrated enough among true positives at the very top of the list, where the
baseline's exposure-weighted CTR gap does better. This suggests the model is picking up real
but diffuse signal (useful in aggregate, per the feature importances below) rather than a
sharp top-K discriminator, while the baseline's narrower, hand-designed signal is precisely
matched to the top-K use case it was built for.

**Reproducibility finding, not swept under the rug:** an earlier run of the model notebook had
cached Precision@50 = 0.82, but the cells that loaded and prepared its data had since been
cleared, making that number unreproducible. Rebuilt to run top-to-bottom on the same split and
seed, it reproduces ROC-AUC ≈ 0.6145 (materially unchanged) but Precision@50 = 0.74 — a wider
baseline advantage in the same direction. Both runs agree on the directional conclusion
(baseline ≥ model); 0.74 is the number reported everywhere in this repo and the paper because
it is the one that's actually reproducible end-to-end.

## 6. Interpretation

Top model feature importances: `impressions_90d` (0.168), `content_age_days` (0.119),
`avg_position` (0.117), `scroll_rate` (0.048), `ctr` (0.043), `clicks_90d` (0.036),
`search_volume` (0.033), `position_tier_top_3` (0.033), `pageviews_90d` (0.031),
`days_since_last_update` (0.031). In plain words: the model leans hardest on how much exposure
a page gets, how old it is, and where it ranks — traffic volume and page age carry more
predictive weight than any single engagement metric. No one feature dominates overwhelmingly
(the top feature is ~17% of total importance), which is a mild reassurance against leakage —
a leaked label-derived feature would typically show up as a single feature carrying most of
the importance.

**Negative/surprising result, stated plainly:** the more complex model did not outperform the
hand-written rule. That's a valid, useful finding on its own — it means the extra complexity
(harder to explain to a non-technical reviewer, harder to audit row by row) isn't currently
buying predictive power over the transparent baseline for this specific top-K decision. The
recommendation below reflects that: the baseline is primary, the model is kept only as a
secondary corroborating signal.

## 7. Recommendation

The ranked action playbook (`work/outputs/action_playbook.csv`, 28,795 rows) combines the
baseline's opportunity score (60% weight — the validated winner) with the model's decline-risk
probability (40% weight — corroborating signal) into one `final_score`, with reason codes on
every row (`low_ctr_vs_position`, `model_flags_decline_risk`, `high_exposure`,
`stale_content`). Suggested-action mix across all 28,795 pages: `review_ctr` (13,591),
`review_ctr_and_content` (10,180), `no_action` (3,684), `monitor_decline_risk` (1,340).

**How a FlyRank editor would use this tomorrow:** open the queue, start from rank 1, and work
down; for each page, read the reason codes, glance at the page itself, and decide whether to
rewrite the meta/content, leave it, or flag it for a longer review. High-exposure top-3
pages with a wide CTR gap and a model-flagged decline risk (the top of the current queue) are
the highest-value first stop, since fixing those affects the most search traffic.

**Confidence and limits, stated explicitly:** this is decision support, not a directive —
Precision@50 of 0.84 means roughly 8 in 10 of the top 50 flagged pages are genuinely declining
by this dataset's definition, not that fixing them will improve CTR or rankings (see the
paper's Limitations section for the full list: not causal, not a Google-algorithm model, a
thin 7-client test set, and `ctr_gap` only compares a page to its position-tier average, not
its specific query context).

## 8. Reproducibility

**Commands to re-run everything from a fresh clone:**

```bash
pip install -r requirements.txt
pip install nbformat nbclient jupyter --break-system-packages   # notebook execution only

# Regenerate every chart/number from raw data in one run
python3 scripts/build_capstone_artifacts.py

# Re-execute the notebooks top-to-bottom (writes fresh outputs in place)
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w05_model.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w06_validation_audit.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/w07_action_playbook.ipynb
jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
```

**Random seed:** 42, fixed everywhere a split or the model is instantiated
(`GroupShuffleSplit(random_state=42)`, `RandomForestClassifier(random_state=42)`).

**Environment (key versions this was verified against):** `scikit-learn==1.8.0`,
`pandas==3.0.2`, `numpy==2.4.4`, against the `requirements.txt` floor of
`scikit-learn>=1.4`, `pandas>=2.2`, `numpy>=1.26`. Note: the Section 5 reproducibility finding
is a direct illustration of why this matters — tree-ensemble top-K metrics can shift a few
points between library versions or even between reruns with cleared/rerun cells, even with a
fixed seed, so the version floor above is the one this report's numbers were checked against.

**Held-out evaluation, checkable not just claimed:** the client-grouped split is built in
`work/notebooks/w05_model.ipynb` (Section 3, cell 6) and `work/notebooks/capstone.ipynb`
(Section 3); the resulting metrics are written to `work/outputs/model_vs_baseline.csv` and
`work/outputs/summary_stats.json` by `scripts/build_capstone_artifacts.py`. Both the
split-building code and the metrics file it produces are committed to the repo, so "evaluated
once, on a held-out client group" is verifiable by re-running the commands above, not taken on
faith.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).

---

> **Claims checklist before submitting:**
> - Base rate reported next to Precision@50 everywhere it appears (test base rate: 0.511) —
>   both baseline (0.84) and model (0.74) clear it by a wide margin, so neither score is just
>   restating the base rate.
> - Language used throughout: observed, measured, associated, directional, decision-support.
>   No causal claims, no "predicted Google's algorithm."
> - No client-identifying details in `work/outputs/`, `docs/`, or this report — checked by
>   grep for `content_id`/`client_id`/secret-like patterns.
> - Numbers in this report match a fresh re-run: confirmed by re-executing
>   `w05_model.ipynb`, `w06_validation_audit.ipynb`, `w07_action_playbook.ipynb`, and
>   `capstone.ipynb` top-to-bottom immediately before writing this report — all four ran with
>   zero errors and reproduced Precision@50 = 0.84 (baseline) / 0.74 (model), ROC-AUC = 0.6145.
