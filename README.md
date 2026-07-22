# WC 2026 Prediction Dashboard

A self-learning prediction model with a live backend for the FIFA World Cup 2026.

**🔴 Live demo:** https://xuanbachmai.github.io/wc2026-prediction-dashboard/

## Result

The tournament ran its full course through this model, updating live off ESPN after
every match. Final tally across all **104 matches**:

- **72.1% match-outcome accuracy** (75/104 correct) — scored honestly, every prediction
  logged against the real result including the knockouts.
- **Champion called correctly:** the model's projected winner was **Spain**, and Spain
  beat Argentina 1–0 in the final.

## Architecture

The backend (`serve.py`) runs everything automatically — nothing is manual:

```
serve.py  (the backend process)
 ├─ watcher thread   — polls ESPN every 3 s, writes goals/results/penalties
 ├─ rebuild thread   — invokes engine.py after every change (retrain + re-render)
 ├─ backfill thread  — recovers missed results at boot and hourly
 ├─ full-refresh     — every 6 h refreshes news + corner data
 └─ HTTP + SSE       — serves the page and pushes live updates to browsers
```

`engine.py` is the model + rendering module the backend invokes: it retrains on
every new result (loop-learned ELO, online-learned xG biases), recomputes every
prediction, and renders the page. It is never run by hand.

## What the model does

- **Match predictions** — RandomForest outcome classifier + Poisson xG,
  blended with a Dixon-Coles scoreline matrix
- **Loop learning** — ELO (K=60 WC weight) and xG biases update after every result
- **Knockout progression** — bracket slots auto-resolve as winners are decided
- **Accuracy tracker** — every prediction scored against the real result
- **Goalscorers** — golden boot race, per-team scorers, walk-forward backtested
  scorer picks for upcoming matches
- **Corners model** — walk-forward learned corner predictions with O/U picks

## Run locally

```bash
pip install -r requirements.txt
python3 serve.py     # backend: builds if needed, serves at http://localhost:8765
```

Or just `./start.sh`.

## Hosting

- **Render** (live backend): `render.yaml` blueprint — deploys `serve.py` with the
  full watcher/rebuild/SSE loop. Auto-redeploys on every push.
- **GitHub Pages** (static mirror): a GitHub Actions cron fetches results, rebuilds,
  commits, and redeploys — keeps a free always-on copy current.

## Key files

| File | Purpose |
|---|---|
| `serve.py` | The backend: watcher, rebuilds, backfill, HTTP + SSE |
| `engine.py` | Model + rendering module (invoked by the backend) |
| `update_results.py` | ESPN result fetcher + knockout-slot resolver (backend + CI) |
| `schedule.py` | Master schedule — all 104 matches, results, penalties |
| `main.py` / `model.py` / `elo.py` / `features.py` | Training pipeline |
| `online_learner.py` | Post-match learning (ELO, xG bias, factor weights) |
| `factor_engine.py` / `team_intelligence.py` / `player_data.py` | Prediction factors |
| `corners_model.py` / `corners_learning.py` / `fetch_corner_stats.py` | Corners |
| `fetch_scorers.py` | Per-player goal data from ESPN |
