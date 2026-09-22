# tools/

Kept in the repo deliberately: both scripts lived in the session scratchpad,
which rotates and has silently deleted work three times in this project.

## preflight.py — RUN BEFORE EVERY MULTI-HOUR JOB

Runs the real pipeline on a small slice (~2 min) and fails the launch if:

1. a feature is the label or its bookkeeping (`forward_return`, `target_date`, `_cmp_*`)
2. any feature correlates >0.95 with the target
3. `--technicals-only` / `--fundamental-only` did not filter as claimed
4. the pipeline cannot complete one train+predict, or predictions are constant

Check 2 is the one that matters. A composite-target leak once cost 6.5 hours of
compute: `_cmp_r20` was the forward return itself, sitting in the feature matrix,
and produced IC +0.82 with 100% of days positive. Preflight catches that in two
minutes.

    python3 tools/preflight.py [technicals|fundamental|all]

Edit SRC at the top to point at the input you are about to train on. If the
input uses lowercase column names, the script reuses the forecasting script's
own PascalCase map rather than reimplementing it — reimplementing it is how an
earlier version of this preflight failed.

## score_runs.py — compare finished runs

Scores prediction parquets on rank IC and top-N basket excess by market-cap
band. **Outcomes are always recomputed from RefPriceClose**, never read from the
stored `forward_return`, which is forward-filled and untrustworthy at the recent
edge.

Two rules learned the hard way:

- **IC is not the objective; basket excess is.** They can point in opposite
  directions. Dropping stale labels raised IC (+0.0272 -> +0.0397) while lowering
  top-50 basket excess (+6.87% -> +6.29%) and the win rate (72% -> 58%).
- **Split by period before making a claim.** "Momentum beats the ML" held over
  the full window (+7.61% vs +7.17%) but inverted in 2026 (+10.90% vs +13.47%),
  which was the period that actually mattered.

Edit the RUNS dict to point at the parquets you want compared.
