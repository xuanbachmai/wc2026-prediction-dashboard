"""
engine.py — Build a full interactive HTML prediction dashboard.

Output: output/dashboard.html  (single self-contained file, no server needed)

Sections
────────
  Overview      — tournament summary cards
  Group Stage   — all 72 matches with full reasoning
  Standings     — expected finish % per group
  Knockout      — projected bracket R32 → Final
  Team Profiles — scouting cards for all 48 teams
"""

import argparse, json, math, os
import numpy as np
from pathlib import Path
from itertools import combinations

import data, elo as elo_module, features as feat_module, model as model_module
from simulate import _build_match_features, precompute_lambdas
from team_intelligence import get_intel_features, TEAM_INTEL
from config import WC_2026_GROUPS, FIFA_DISPLAY_NAMES
from schedule import SCHEDULE, get_group_matches, is_played, result_str, KICKOFF_TIMES
from predict_all_matches import predict_match
from main import build_team_stats_lookup
from news_fetcher import (
    fetch_all_upcoming_news, aggregate_sentiment,
)
from factor_engine import compute_factors, apply_factor_adjustments
from match_context import get_match_context
from team_intelligence import TEAM_INTEL

# ── Actual corners loader ─────────────────────────────────────────────────────
ACTUAL_CORNERS_PATH = Path("data/actual_corners.json")

def load_actual_corners() -> dict:
    """Load actual corner counts (ESPN box scores)."""
    if ACTUAL_CORNERS_PATH.exists():
        try:
            return json.loads(ACTUAL_CORNERS_PATH.read_text())
        except Exception:
            pass
    return {}

# ── Reasoning engine ──────────────────────────────────────────────────────────

def build_reasoning(home: str, away: str, pred: dict,
                    elo: dict, stats: dict) -> dict:
    """Produce structured reasoning for a single match prediction."""
    hi   = get_intel_features(home)
    ai   = get_intel_features(away)
    hint = TEAM_INTEL.get(home, {})
    aint = TEAM_INTEL.get(away, {})
    h_elo = elo.get(home, 1500)
    a_elo = elo.get(away, 1500)
    hs    = stats.get(home, {})
    as_   = stats.get(away, {})

    factors  = []   # {icon, label, text, side}   side = "home"|"away"|"neutral"
    risks    = []   # plain strings

    # 1. ELO
    elo_diff = h_elo - a_elo
    if abs(elo_diff) >= 30:
        fav  = home if elo_diff > 0 else away
        side = "home" if elo_diff > 0 else "away"
        factors.append({"icon": "📈", "label": "Historical Rating (ELO)",
                        "text": f"{fav} rated {abs(elo_diff):.0f} pts higher "
                                f"({h_elo:.0f} vs {a_elo:.0f})", "side": side})

    # 2. Star quality
    sd = hi["star_rating"] - ai["star_rating"]
    if abs(sd) >= 0.4:
        fav  = home if sd > 0 else away
        fi   = hi if sd > 0 else ai
        side = "home" if sd > 0 else "away"
        kp   = (hint if sd > 0 else aint).get("key_players", [])[:2]
        factors.append({"icon": "⭐", "label": "Star Player Quality",
                        "text": f"{fav} leads ({fi['star_rating']:.1f}/10) — "
                                f"{', '.join(kp)}", "side": side})

    # 3. Current star form
    fd = hi["star_form"] - ai["star_form"]
    if abs(fd) >= 0.8:
        fav  = home if fd > 0 else away
        fi   = hi if fd > 0 else ai
        side = "home" if fd > 0 else "away"
        factors.append({"icon": "🔥", "label": "Star In-Form",
                        "text": f"{fav} stars in sharper form this season "
                                f"({fi['star_form']:.1f}/10)", "side": side})

    # 4. Tactical score
    td = hi["tactical_score"] - ai["tactical_score"]
    if abs(td) >= 0.7:
        fav  = home if td > 0 else away
        fi   = hi if td > 0 else ai
        side = "home" if td > 0 else "away"
        styles = {"tiki-taka":"possession-based tiki-taka",
                  "possession":"possession-dominant",
                  "high-press":"high-intensity pressing",
                  "counter-attack":"disciplined counter-attack",
                  "attacking":"direct attacking football",
                  "defensive":"compact defensive block"}
        style_label = styles.get((hint if fav==home else aint).get("style","defensive"),
                                  "tactical")
        factors.append({"icon": "🎯", "label": "Tactical Superiority",
                        "text": f"{fav} has better coaching & system "
                                f"({fi['tactical_score']:.1f}/10, {style_label})",
                        "side": side})

    # 5. Squad depth
    ddd = hi["squad_depth"] - ai["squad_depth"]
    if abs(ddd) >= 1.5:
        fav  = home if ddd > 0 else away
        fi   = hi if ddd > 0 else ai
        side = "home" if ddd > 0 else "away"
        factors.append({"icon": "👥", "label": "Squad Depth",
                        "text": f"{fav} has significantly more squad quality "
                                f"top-to-bottom ({fi['squad_depth']:.1f}/10)",
                        "side": side})

    # 6. Tournament motivation
    if hi["political_boost"] >= 7.5 or ai["political_boost"] >= 7.5:
        fav  = home if hi["political_boost"] >= ai["political_boost"] else away
        fi   = hi if fav == home else ai
        side = "home" if fav == home else "away"
        note = (hint if fav == home else aint).get("scouting_notes","")[:75]
        factors.append({"icon": "🏟️", "label": "Motivation / Context",
                        "text": f"{fav} ({fi['political_boost']:.1f}/10): {note}",
                        "side": side})

    # 7. Recent form (from rolling stats)
    h_form = hs.get("form", 1.5)
    a_form = as_.get("form", 1.5)
    rfm = h_form - a_form
    if abs(rfm) >= 0.25:
        fav  = home if rfm > 0 else away
        side = "home" if rfm > 0 else "away"
        factors.append({"icon": "📊", "label": "Recent Match Form",
                        "text": f"{fav} averaging more points per game recently "
                                f"({h_form:.2f} vs {a_form:.2f} pts/match)",
                        "side": side})

    # 8. Carrying risk flags
    for team, intel, ti in [(home, hi, hint), (away, ai, aint)]:
        if ti.get("carrying_factor", 0) >= 8.0:
            kp = ti.get("key_players", ["star player"])[0]
            risks.append(
                f"{team} rely heavily on {kp} "
                f"(dependency score {ti['carrying_factor']:.1f}/10) — "
                f"one bad game from him collapses their attack."
            )

    # Verdict
    pw, pa = pred["p_home_win"], pred["p_away_win"]
    if pw >= 70:
        verdict = (f"<strong>{home}</strong> are clear favourites ({pw:.0f}%). "
                   f"Expect them to control the game and win comfortably.")
    elif pw >= 55:
        verdict = (f"<strong>{home}</strong> are marginal favourites ({pw:.0f}%), "
                   f"but {away} are live underdogs. Could go either way.")
    elif pa >= 70:
        verdict = (f"<strong>{away}</strong> are clear favourites ({pa:.0f}%). "
                   f"Despite playing as 'away', the model sees them as significantly stronger.")
    elif pa >= 55:
        verdict = (f"<strong>{away}</strong> edge this ({pa:.0f}%), "
                   f"but {home} are competitive and could cause a surprise.")
    else:
        verdict = (f"Coin-flip contest. Both teams rated very closely. "
                   f"Match likely decided by fine margins or a set piece.")

    return {
        "factors":         factors,
        "risks":           risks,
        "verdict":         verdict,
        "home_notes":      hint.get("scouting_notes", ""),
        "away_notes":      aint.get("scouting_notes", ""),
        "home_players":    hint.get("key_players", []),
        "away_players":    aint.get("key_players", []),
        "home_style":      hint.get("style", ""),
        "away_style":      aint.get("style", ""),
        "home_star":       hi["star_rating"],
        "away_star":       ai["star_rating"],
        "home_depth":      hi["squad_depth"],
        "away_depth":      ai["squad_depth"],
        "home_tactics":    hi["tactical_score"],
        "away_tactics":    ai["tactical_score"],
        "home_motivation": hi["political_boost"],
        "away_motivation": ai["political_boost"],
        "home_carry":      hi["carrying_factor"],
        "away_carry":      ai["carrying_factor"],
        "home_elo":        round(elo.get(home, 1500)),
        "away_elo":        round(elo.get(away, 1500)),
    }


# ── Build all data payloads ───────────────────────────────────────────────────

def display_name(team: str) -> str:
    return FIFA_DISPLAY_NAMES.get(team, team)


def apply_loop_learning(elo: dict, schedule: list, K: float = 60.0) -> dict:
    """
    Update ELO ratings using actual WC 2026 results already played.
    K=60 (WC weight) so real results have strong influence on upcoming predictions.
    Returns a *new* dict — original ELO is not mutated.
    """
    updated = dict(elo)
    for m in schedule:
        if m["home_score"] is None:
            continue                           # not yet played
        home, away = m["home"], m["away"]
        hg, ag     = m["home_score"], m["away_score"]
        h_elo      = updated.get(home, 1500)
        a_elo      = updated.get(away, 1500)

        # Expected win probability (standard ELO formula)
        exp_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        exp_a = 1 - exp_h

        # Actual outcome score (1=win, 0.5=draw, 0=loss)
        if hg > ag:
            act_h, act_a = 1.0, 0.0
        elif ag > hg:
            act_h, act_a = 0.0, 1.0
        else:
            act_h, act_a = 0.5, 0.5

        updated[home] = h_elo + K * (act_h - exp_h)
        updated[away] = a_elo + K * (act_a - exp_a)

    return updated


def predict_enhanced(home: str, away: str, outcome_model, goals_model,
                     elo: dict, stats: dict,
                     factor_weights: dict | None = None,
                     xg_bias_home: float = 0.0,
                     xg_bias_away: float = 0.0,
                     extra_draw_inflation: float = 0.0) -> dict:
    """
    Prediction pipeline: base ML model → learned xG bias correction.

    xg_bias_home/away are computed by OnlineLearner from actual WC match
    prediction errors — the only real-data correction applied here.
    All manual factor adjustments (mismatch amplifier, draw inflator,
    DC-Rho, low-block, host bonus, etc.) have been removed.
    """
    pred = predict_match(home, away, outcome_model, goals_model, elo, stats)
    pred = dict(pred)

    # Apply learned xG bias correction (mean error from actual WC results)
    bias = (xg_bias_home + xg_bias_away) / 2  # symmetric — WC is neutral
    if bias:
        pred["xg_home"] = max(0.3, pred.get("xg_home", 1.2) + bias)
        pred["xg_away"] = max(0.3, pred.get("xg_away", 1.0) + bias)

    # Apply learned draw inflation: shift probability mass from H/A wins → draw.
    # This is purely data-driven — computed by OnlineLearner from actual WC results.
    if extra_draw_inflation > 0:
        ph  = pred.get("p_home_win", 33.3)
        pd_ = pred.get("p_draw",     33.3)
        pa  = pred.get("p_away_win", 33.3)
        # Take proportionally from each win bucket, add to draw
        shift = extra_draw_inflation * 100  # fraction → percentage points
        ph_new  = max(1.0, ph  - shift * ph  / (ph + pa))
        pa_new  = max(1.0, pa  - shift * pa  / (ph + pa))
        pd_new  = pd_ + (ph - ph_new) + (pa - pa_new)
        total   = ph_new + pd_new + pa_new
        pred["p_home_win"] = round(ph_new * 100 / total, 1)
        pred["p_draw"]     = round(pd_new * 100 / total, 1)
        pred["p_away_win"] = round(pa_new * 100 / total, 1)

    # Compute factor cards for display only — no xG adjustment applied
    factors = compute_factors(home, away, pred,
                              elo.get(home, 1500), elo.get(away, 1500),
                              factor_weights=None)
    pred["factor_cards"]   = factors.get("factor_cards", [])
    pred["factor_summary"] = factors.get("summary", {})

    # Direction-consistent score: most probable scoreline that agrees with predicted winner.
    # This avoids showing "1-1" when the model picks a team to win.
    from scipy.stats import poisson as _poisson
    _xg_h = pred.get("xg_home", 1.2)
    _xg_a = pred.get("xg_away", 1.0)
    _ph = pred.get("p_home_win", 34)
    _pd = pred.get("p_draw", 33)
    _pa = pred.get("p_away_win", 33)
    _dir = "home" if _ph >= _pd and _ph >= _pa else ("away" if _pa >= _ph and _pa >= _pd else "draw")
    _best_sc, _best_p = None, -1
    for _hg in range(7):
        for _ag in range(7):
            _matches_dir = (
                (_dir == "home" and _hg > _ag) or
                (_dir == "away" and _ag > _hg) or
                (_dir == "draw" and _hg == _ag)
            )
            if not _matches_dir:
                continue
            _p = _poisson.pmf(_hg, _xg_h) * _poisson.pmf(_ag, _xg_a)
            if _p > _best_p:
                _best_p = _p
                _best_sc = (_hg, _ag)
    pred["direction_score"] = f"{_best_sc[0]}-{_best_sc[1]}" if _best_sc else pred.get("likely_score", "1-0")

    return pred


def compute_deterministic_standings(outcome_model, goals_model, elo: dict, stats: dict,
                                    factor_weights=None, xg_bias_home=0.0,
                                    xg_bias_away=0.0, extra_draw_inflation=0.0) -> dict:
    """
    Compute expected group standings — 2026 FIFA tiebreaker rules.

    NEW in 2026 vs 2022:
      Primary tiebreaker is HEAD-TO-HEAD points (not overall goal difference).
      Full order: H2H pts → H2H GD → H2H GF → overall GD → overall GF → FIFA ranking.

    We approximate with expected values:
      sort key = (exp_pts, h2h_pts_vs_equal, overall_GD, overall_GF)
    """
    result = {}
    for grp, teams in WC_2026_GROUPS.items():
        exp_pts:    dict[str, float] = {t: 0.0 for t in teams}
        exp_gf:     dict[str, float] = {t: 0.0 for t in teams}
        exp_ga:     dict[str, float] = {t: 0.0 for t in teams}
        # H2H points matrix: h2h_pts[A][B] = expected pts A earns in the A vs B match
        h2h_pts: dict[str, dict[str, float]] = {t: {o: 0.0 for o in teams if o != t} for t in teams}
        h2h_gf:  dict[str, dict[str, float]] = {t: {o: 0.0 for o in teams if o != t} for t in teams}
        h2h_ga:  dict[str, dict[str, float]] = {t: {o: 0.0 for o in teams if o != t} for t in teams}

        match_preds: dict[tuple, dict] = {}
        for home, away in combinations(teams, 2):
            pred = predict_enhanced(home, away, outcome_model, goals_model, elo, stats,
                                    factor_weights=factor_weights,
                                    xg_bias_home=xg_bias_home, xg_bias_away=xg_bias_away,
                                    extra_draw_inflation=extra_draw_inflation)
            ph, pd_, pa = pred["p_home_win"] / 100, pred["p_draw"] / 100, pred["p_away_win"] / 100
            match_preds[(home, away)] = pred

            # Overall stats
            exp_pts[home] += ph * 3 + pd_
            exp_pts[away] += pa * 3 + pd_
            exp_gf[home]  += pred["xg_home"]
            exp_ga[home]  += pred["xg_away"]
            exp_gf[away]  += pred["xg_away"]
            exp_ga[away]  += pred["xg_home"]

            # H2H stats (used as tiebreaker — 2026 rule)
            h2h_pts[home][away] = ph * 3 + pd_
            h2h_pts[away][home] = pa * 3 + pd_
            h2h_gf[home][away]  = pred["xg_home"]
            h2h_gf[away][home]  = pred["xg_away"]
            h2h_ga[home][away]  = pred["xg_away"]
            h2h_ga[away][home]  = pred["xg_home"]

        def sort_key(t):
            # 2026 tiebreaker order:
            # 1. Overall points (always first)
            # 2. H2H points among tied teams (simplified: sum vs all opponents)
            # 3. H2H goal difference
            # 4. H2H goals scored
            # 5. Overall goal difference
            # 6. Overall goals scored
            h2h_pts_total = sum(h2h_pts[t].values())
            h2h_gd        = sum(h2h_gf[t].values()) - sum(h2h_ga[t].values())
            h2h_gf_total  = sum(h2h_gf[t].values())
            overall_gd    = exp_gf[t] - exp_ga[t]
            return (exp_pts[t], h2h_pts_total, h2h_gd, h2h_gf_total, overall_gd, exp_gf[t])

        order = sorted(teams, key=sort_key, reverse=True)

        exp_pos = {t: order.index(t) + 1 for t in teams}
        pts_sorted = [exp_pts[t] for t in order]
        pts_gap = {}
        for i, t in enumerate(order):
            pts_gap[t] = round(pts_sorted[0] - pts_sorted[1], 2) if i == 0 \
                         else round(pts_sorted[i-1] - pts_sorted[i], 2)

        result[grp] = {
            "order":    order,
            "exp_pts":  {t: round(exp_pts[t], 2) for t in teams},
            "exp_gf":   {t: round(exp_gf[t],  2) for t in teams},
            "exp_ga":   {t: round(exp_ga[t],  2) for t in teams},
            "exp_pos":  exp_pos,
            "pts_gap":  pts_gap,
            "h2h_pts":  {t: round(sum(h2h_pts[t].values()), 2) for t in teams},
        }
    return result


def build_deterministic_ko_bracket(
    det_standings: dict,
    outcome_model, goals_model, elo: dict, stats: dict,
    factor_weights=None, xg_bias_home=0.0, xg_bias_away=0.0, extra_draw_inflation=0.0,
) -> tuple[list[dict], str]:
    """
    Build a fully deterministic knockout bracket.
    Seeding: top 2 per group (24 teams) + best 8 third-place = 32 teams.
    Pairing:  ELO-seed bracket (1 vs 32, 2 vs 31, …) — no randomness.
    Each match winner = team with higher P(win) from the model.

    Returns (ko_rows, predicted_champion)
    where ko_rows = list of dicts with round/slot/team1/team2/pred/predicted_winner
    """
    firsts:  list[str] = []
    seconds: list[str] = []
    thirds:  list[dict] = []

    for grp, data_ in det_standings.items():
        order = data_["order"]
        firsts.append(order[0])
        seconds.append(order[1])
        thirds.append({
            "team":    order[2],
            "exp_pts": data_["exp_pts"][order[2]],
            "exp_gd":  data_["exp_gf"][order[2]] - data_["exp_ga"][order[2]],
            "exp_gf":  data_["exp_gf"][order[2]],
        })

    # 2026 rule: best 8 of 12 third-placed teams rank by pts → GD → GF → FIFA ranking
    best8 = sorted(thirds, key=lambda x: (x["exp_pts"], x["exp_gd"], x["exp_gf"]), reverse=True)[:8]
    third_teams = [x["team"] for x in best8]

    # All 32 qualified teams, seeded by ELO (descending)
    all_32 = firsts + seconds + third_teams
    seeded = sorted(all_32, key=lambda t: elo.get(t, 1500), reverse=True)

    def run_round(pairs: list[tuple[str, str]], rnd: str) -> tuple[list[dict], list[str], list[str]]:
        """Returns (rows, winners, losers). Uses full enhanced model."""
        rows:    list[dict] = []
        winners: list[str]  = []
        losers:  list[str]  = []
        for slot, (t1, t2) in enumerate(pairs, 1):
            pred   = predict_enhanced(t1, t2, outcome_model, goals_model, elo, stats,
                                      factor_weights=factor_weights,
                                      xg_bias_home=xg_bias_home, xg_bias_away=xg_bias_away,
                                      extra_draw_inflation=extra_draw_inflation)
            winner = t1 if pred["p_home_win"] >= pred["p_away_win"] else t2
            loser  = t2 if winner == t1 else t1
            rows.append({
                "round":             rnd,
                "slot":              slot,
                "team1":             t1,
                "team2":             t2,
                "predicted_winner":  winner,
                "p_home_win":        pred["p_home_win"],
                "p_draw":            pred["p_draw"],
                "p_away_win":        pred["p_away_win"],
                "xg_home":           pred["xg_home"],
                "xg_away":           pred["xg_away"],
                "likely_score":      pred["likely_score"],
                "expected_score":    pred.get("expected_score", pred["likely_score"]),
                "direction_score":   pred.get("direction_score", pred["likely_score"]),
                "likely_score_%":    pred["likely_score_%"],
                "favourite":         pred["favourite"],
                "favourite_p":       pred["favourite_p"],
                "meet_pct":          None,
            })
            winners.append(winner)
            losers.append(loser)
        return rows, winners, losers

    n = len(seeded)  # 32
    all_rows: list[dict] = []

    # R32 — 16 matches
    r32_pairs = [(seeded[i], seeded[n - 1 - i]) for i in range(n // 2)]
    rows, w32, _ = run_round(r32_pairs, "R32")
    all_rows.extend(rows)

    # R16 — 8 matches
    r16_pairs = [(w32[i], w32[i + 1]) for i in range(0, len(w32), 2)]
    rows, w16, _ = run_round(r16_pairs, "R16")
    all_rows.extend(rows)

    # QF — 4 matches
    qf_pairs = [(w16[i], w16[i + 1]) for i in range(0, len(w16), 2)]
    rows, wqf, _ = run_round(qf_pairs, "QF")
    all_rows.extend(rows)

    # SF — 2 matches → track losers for 3rd-place playoff
    sf_pairs = [(wqf[i], wqf[i + 1]) for i in range(0, len(wqf), 2)]
    rows, wsf, sf_losers = run_round(sf_pairs, "SF")
    all_rows.extend(rows)

    # 3rd place playoff — 1 match (SF losers)
    rows, third_winner, _ = run_round([(sf_losers[0], sf_losers[1])], "3rd")
    all_rows.extend(rows)

    # Final — 1 match
    rows, champs, _ = run_round([(wsf[0], wsf[1])], "Final")
    all_rows.extend(rows)

    return all_rows, champs[0]


def _accuracy_verdict(pred: dict, m: dict) -> dict:
    """Compare a pre-learning prediction with an actual result."""
    hs, as_ = m["home_score"], m["away_score"]
    if hs is None:
        return {}

    # Actual winner (team name or 'Draw')
    if hs > as_:
        actual_winner = m["home"]
    elif as_ > hs:
        actual_winner = m["away"]
    else:
        actual_winner = "Draw"

    # Predicted score = most probable scoreline consistent with predicted direction
    predicted_score = pred.get("direction_score") or pred.get("expected_score") or pred.get("likely_score", "1-1")
    actual_score    = f"{hs}-{as_}"

    # Predicted direction = highest aggregate outcome probability (simple argmax).
    ph = pred.get("p_home_win", 34)
    pd_ = pred.get("p_draw", 33)
    pa = pred.get("p_away_win", 33)
    if pd_ >= ph and pd_ >= pa:
        predicted_winner = "Draw"
    elif ph >= pa:
        predicted_winner = m["home"]
    else:
        predicted_winner = m["away"]

    direction_ok = (predicted_winner == actual_winner)
    score_ok     = (predicted_score == actual_score)

    return {
        "actual_winner":     actual_winner,
        "actual_score":      actual_score,
        "predicted_winner":  predicted_winner,
        "predicted_score":   predicted_score,
        "direction_correct": direction_ok,
        "score_correct":     score_ok,
        "pre_p_home":        pred["p_home_win"],
        "pre_p_draw":        pred["p_draw"],
        "pre_p_away":        pred["p_away_win"],
        "pre_xg_home":       pred["xg_home"],
        "pre_xg_away":       pred["xg_away"],
    }


def build_all_data(goals_model, outcome_model, elo, stats):
    sched_group = get_group_matches()

    # ── Step 1: Pre-learning predictions for ALL group matches ───────────────
    print("[dashboard] Step 1 — Pre-learning predictions (base ELO + factors, no online adj)...")
    pre_preds: dict[int, dict] = {}
    for m in sched_group:
        # Use full pipeline (DC correction + factor engine) but NO online learner
        # adjustments — these represent what we'd have predicted before seeing results.
        pre_preds[m["match_no"]] = predict_enhanced(
            m["home"], m["away"], outcome_model, goals_model, elo, stats)

    # ── Step 2: Apply loop learning — update ELO from actual results ─────────
    print("[dashboard] Step 2 — Applying loop learning from actual WC results...")
    from schedule import SCHEDULE as FULL_SCHEDULE
    learned_elo = apply_loop_learning(elo, FULL_SCHEDULE)

    # How many teams had ELO updated?
    elo_changes = {t: round(learned_elo[t] - elo.get(t, 1500), 1)
                   for t in learned_elo if abs(learned_elo[t] - elo.get(t, 1500)) > 0.01}
    print(f"[dashboard]   ELO updated for {len(elo_changes)} teams: "
          + ", ".join(f"{t}{'+' if d>0 else ''}{d}" for t, d in sorted(elo_changes.items(), key=lambda x: -abs(x[1]))))

    # ── Step 2.5: Online learner — self-learn from WC 2026 results ──────────
    print("[dashboard] Step 2.5 — Running self-learning engine (OnlineLearner)...")
    try:
        from online_learner import OnlineLearner
        learner = OnlineLearner()
        learner.process_all_played(FULL_SCHEDULE, pre_preds, elo)
        ol_adj = learner.get_adjustments()
        learner.print_summary()
        learner.save()
        # Apply WC-learned ELO on top of loop-learning ELO
        for team, wc_elo in ol_adj["elo_overrides"].items():
            learned_elo[team] = wc_elo
        online_factor_weights  = ol_adj["factor_weights"]
        online_xg_bias_home    = ol_adj["xg_bias_home"]
        online_xg_bias_away    = ol_adj["xg_bias_away"]
        online_draw_inflation  = ol_adj["draw_inflation"]
        print(f"[dashboard]   OnlineLearner adjustments applied — "
              f"draw_inflation={online_draw_inflation:+.3f}, "
              f"xg_bias=({online_xg_bias_home:+.3f}/{online_xg_bias_away:+.3f})")
    except Exception as _ol_err:
        print(f"[dashboard]   OnlineLearner skipped ({_ol_err}) — using base ELO only")
        online_factor_weights = None
        online_xg_bias_home = online_xg_bias_away = online_draw_inflation = 0.0

    # ── Step 3a: Fetch real news for upcoming matches (before building cards) ──
    print("[dashboard] Step 3a — Fetching real news for upcoming matches...")
    news_data = fetch_all_upcoming_news(FULL_SCHEDULE, max_matches=20)

    # ── Step 3: Group matches — played use pre-learning, unplayed use learned ─
    print("[dashboard] Step 3 — Building group match cards...")
    group_matches = []
    accuracy_rows = []

    for m in sched_group:
        home, away = m["home"], m["away"]
        played     = is_played(m)

        if played:
            # Show the pre-learning prediction alongside the real result
            pred   = pre_preds[m["match_no"]]
            av     = _accuracy_verdict(pred, m)
            reason = build_reasoning(home, away, pred, elo, stats)
        else:
            # Enhanced model: loop-learned ELO + online-learned adjustments + factors
            pred   = predict_enhanced(home, away, outcome_model, goals_model, learned_elo, stats,
                                      xg_bias_home=online_xg_bias_home,
                                      xg_bias_away=online_xg_bias_away)
            av     = {}
            reason = build_reasoning(home, away, pred, learned_elo, stats)

        # Attach news for unplayed matches
        articles = news_data.get(m["match_no"], []) if not played else []
        sentiment = aggregate_sentiment(articles, home, away) if articles else {}

        # Star player + discipline data
        from player_data import get_player_data as _gpd, YELLOW_CARDS
        hpd = _gpd(home)
        apd = _gpd(away)

        entry = {
            **pred,
            "home":        home,
            "away":        away,
            "match_no":    m["match_no"],
            "date":        m["date"],
            "group":       m["group"],
            "matchday":    m["matchday"],
            "stadium":     m["stadium"],
            "city":        m["city"],
            "kickoff_utc": KICKOFF_TIMES.get(m["match_no"]),
            "played":      played,
            "actual":      result_str(m),
            "home_display":display_name(home),
            "away_display":display_name(away),
            "reasoning":   reason,
            "news":        articles,
            "news_sentiment": sentiment,
            "factor_cards":  pred.get("factor_cards", []) if not played else [],
            "factor_summary": pred.get("factor_summary", {}) if not played else {},
            # Star player bios
            "home_star_player": hpd.get("star_player", {}),
            "away_star_player": apd.get("star_player", {}),
            # Discipline tracking
            "home_yellow_cards": YELLOW_CARDS.get(home, {}),
            "away_yellow_cards": YELLOW_CARDS.get(away, {}),
            "home_injured":  hpd.get("injured", []),
            "away_injured":  apd.get("injured", []),
            "home_suspended": hpd.get("suspended", []),
            "away_suspended": apd.get("suspended", []),
            "match_context": get_match_context(home, away),
            **av,          # accuracy keys merged in (empty {} for unplayed)
        }
        # Compute predicted_winner for unplayed matches (same argmax logic as _accuracy_verdict)
        if not played:
            _ph = pred.get("p_home_win", 34)
            _pd = pred.get("p_draw", 33)
            _pa = pred.get("p_away_win", 33)
            if _pd >= _ph and _pd >= _pa:
                entry["predicted_winner"] = "Draw"
            elif _ph >= _pa:
                entry["predicted_winner"] = home
            else:
                entry["predicted_winner"] = away
        # Fix xG display precision
        entry["xg_home"] = round(entry.get("xg_home", 0), 2)
        entry["xg_away"] = round(entry.get("xg_away", 0), 2)
        group_matches.append(entry)

        if played and av:
            accuracy_rows.append({
                "match_no":      m["match_no"],
                "date":          m["date"],
                "group":         m["group"],
                "home":          home,
                "away":          away,
                "home_display":  display_name(home),
                "away_display":  display_name(away),
                **av,
            })

    # ── Accuracy summary (base model) ────────────────────────────────────────
    total_played    = len(accuracy_rows)
    dir_correct     = sum(1 for r in accuracy_rows if r["direction_correct"])
    score_correct   = sum(1 for r in accuracy_rows if r["score_correct"])

    # ── Retrospective: re-run with ALL factors on played matches ─────────────
    print("[dashboard] Retrospective — re-running factors on played matches...")
    retro_rows: list[dict] = []
    for m in sched_group:
        if not is_played(m):
            continue
        home, away = m["home"], m["away"]
        base_pred  = pre_preds[m["match_no"]]
        enhanced   = base_pred   # no manual adjustments — model output is final
        enh_av     = _accuracy_verdict(enhanced, m)
        base_av    = _accuracy_verdict(base_pred, m)

        retro_rows.append({
            "match_no":         m["match_no"],
            "home_display":     display_name(home),
            "away_display":     display_name(away),
            "actual_score":     base_av.get("actual_score",""),
            "actual_winner":    base_av.get("actual_winner",""),
            # Base model
            "base_winner":      base_av.get("predicted_winner",""),
            "base_score":       base_av.get("predicted_score",""),
            "base_correct":     base_av.get("direction_correct", False),
            "base_p_home":      base_pred.get("p_home_win", 0),
            "base_p_draw":      base_pred.get("p_draw", 0),
            "base_p_away":      base_pred.get("p_away_win", 0),
            "base_xg_h":        base_pred.get("xg_home", 0),
            "base_xg_a":        base_pred.get("xg_away", 0),
            # Factor-enhanced model
            "enh_winner":       enh_av.get("predicted_winner",""),
            "enh_score":        enh_av.get("predicted_score",""),
            "enh_correct":      enh_av.get("direction_correct", False),
            "enh_p_home":       enhanced.get("p_home_win", 0),
            "enh_p_draw":       enhanced.get("p_draw", 0),
            "enh_p_away":       enhanced.get("p_away_win", 0),
            "enh_xg_h":         enhanced.get("xg_home", 0),
            "enh_xg_a":         enhanced.get("xg_away", 0),
            "xg_adj_home":      0,
            "xg_adj_away":      0,
            "factor_cards":     [],
        })

    retro_base_correct = sum(1 for r in retro_rows if r["base_correct"])
    retro_enh_correct  = sum(1 for r in retro_rows if r["enh_correct"])
    retro_total        = len(retro_rows)
    print(f"[dashboard]   Retrospective: Base {retro_base_correct}/{retro_total} "
          f"({100*retro_base_correct/max(retro_total,1):.1f}%) → "
          f"Enhanced {retro_enh_correct}/{retro_total} "
          f"({100*retro_enh_correct/max(retro_total,1):.1f}%)")

    # accuracy_summary built after actual_ko_schedule (Step 5b) to include KO results
    _acc_group_rows   = accuracy_rows
    _acc_retro_rows   = retro_rows
    _acc_retro_base   = retro_base_correct
    _acc_retro_enh    = retro_enh_correct
    _acc_retro_total  = retro_total

    # ── Step 4: Deterministic group standings (expected points, no RNG) ───────
    print("[dashboard] Step 4 — Computing deterministic group standings...")
    det_standings = compute_deterministic_standings(
        outcome_model, goals_model, learned_elo, stats,
        factor_weights=online_factor_weights,
        xg_bias_home=online_xg_bias_home,
        xg_bias_away=online_xg_bias_away,
        extra_draw_inflation=online_draw_inflation,
    )

    standings = {}
    for grp, data_ in det_standings.items():
        order = data_["order"]
        rows  = []
        for pos, team in enumerate(order, 1):
            ti = get_intel_features(team)
            rows.append({
                "team":     team,
                "exp_pos":  pos,                          # 1st / 2nd / 3rd / 4th
                "advances": pos <= 2,                     # top 2 guaranteed advance
                "exp_pts":  data_["exp_pts"][team],
                "exp_gf":   data_["exp_gf"][team],
                "exp_ga":   data_["exp_ga"][team],
                "exp_gd":   round(data_["exp_gf"][team] - data_["exp_ga"][team], 2),
                "pts_gap":  data_["pts_gap"][team],
                "elo":      round(learned_elo.get(team, 1500)),
                "star":     ti["star_rating"],
            })
        standings[grp] = rows

    # ── Step 5: Deterministic KO bracket (no RNG) ────────────────────────────
    print("[dashboard] Step 5 — Building deterministic knockout bracket...")
    ko_rows, champion = build_deterministic_ko_bracket(
        det_standings, outcome_model, goals_model, learned_elo, stats,
        factor_weights=online_factor_weights,
        xg_bias_home=online_xg_bias_home,
        xg_bias_away=online_xg_bias_away,
        extra_draw_inflation=online_draw_inflation,
    )
    print(f"[dashboard]   Predicted champion: {champion}")

    ko_matches = []
    for r in ko_rows:
        pred_dict = {
            "home_team":    r["team1"],
            "away_team":    r["team2"],
            "p_home_win":   r["p_home_win"],
            "p_draw":       r["p_draw"],
            "p_away_win":   r["p_away_win"],
            "xg_home":      r["xg_home"],
            "xg_away":      r["xg_away"],
            "likely_score": r["likely_score"],
            "expected_score": r.get("expected_score", r["likely_score"]),
            "direction_score": r.get("direction_score", r["likely_score"]),
            "likely_score_%": r["likely_score_%"],
            "favourite":    r["favourite"],
            "favourite_p":  r["favourite_p"],
        }
        reason = build_reasoning(r["team1"], r["team2"], pred_dict, learned_elo, stats)
        ko_matches.append({
            **pred_dict,
            "round":             r["round"],
            "slot":              r["slot"],
            "meet_pct":          None,
            "predicted_winner":  r["predicted_winner"],
            "reasoning":         reason,
        })

    # ── Step 5b: Actual knockout schedule from schedule.py ────────────────────
    print("[dashboard] Step 5b — Building actual knockout schedule from schedule...")
    KO_ROUND_LABELS = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-Finals",
                       "SF": "Semi-Finals", "3rd": "3rd Place", "Final": "Final"}
    actual_ko_schedule = []
    for m in SCHEDULE:
        rnd = m.get("group", "")
        if rnd not in KO_ROUND_LABELS:
            continue
        home = m["home"]
        away = m["away"]
        home_tbd = home.startswith("TBD")
        away_tbd = away.startswith("TBD")
        entry = {
            "match_no":  m["match_no"],
            "date":      m["date"],
            "round":     rnd,
            "group":     rnd,
            "slot":      m["match_no"] - 72,
            "home":      home,
            "away":      away,
            "home_display": home if home_tbd else display_name(home),
            "away_display": away if away_tbd else display_name(away),
            "home_tbd":  home_tbd,
            "away_tbd":  away_tbd,
            "home_score": m["home_score"],
            "away_score": m["away_score"],
            "home_pens":  m.get("home_pens"),
            "away_pens":  m.get("away_pens"),
            "played":    m["home_score"] is not None,
            "stadium":   m["stadium"],
            "city":      m["city"],
            "kickoff_utc": KICKOFF_TIMES.get(m["match_no"]),
            "actual":    result_str(m),
        }
        if not home_tbd and not away_tbd:
            pred = predict_enhanced(home, away, outcome_model, goals_model, learned_elo, stats,
                                    factor_weights=online_factor_weights,
                                    xg_bias_home=online_xg_bias_home, xg_bias_away=online_xg_bias_away,
                                    extra_draw_inflation=online_draw_inflation)
            entry.update({
                "p_home_win":   pred["p_home_win"],
                "p_draw":       pred["p_draw"],
                "p_away_win":   pred["p_away_win"],
                "xg_home":      round(pred["xg_home"], 2),
                "xg_away":      round(pred["xg_away"], 2),
                "likely_score": pred["likely_score"],
                "favourite":    pred["favourite"],
                "favourite_p":  pred["favourite_p"],
            })
            # Same conventions as group cards → matchCard + modal work directly
            _ph, _pd, _pa = pred["p_home_win"], pred["p_draw"], pred["p_away_win"]
            if _pd >= _ph and _pd >= _pa:
                entry["predicted_winner"] = "Draw"
            elif _ph >= _pa:
                entry["predicted_winner"] = home
            else:
                entry["predicted_winner"] = away
            if entry["played"]:
                entry.update(_accuracy_verdict(pred, m))
        else:
            entry.update({
                "p_home_win": None, "p_draw": None, "p_away_win": None,
                "xg_home": None, "xg_away": None,
                "likely_score": None, "favourite": None, "favourite_p": None,
            })
        actual_ko_schedule.append(entry)
    print(f"[dashboard]   Actual KO schedule: {len(actual_ko_schedule)} entries")

    # ── Step 5b2: Project the ACTUAL bracket forward ─────────────────────────
    # Played matches → real winner. Unplayed → model favourite. TBD slots
    # resolve recursively from earlier projections. Eliminated teams can never
    # appear in later rounds (unlike the pre-tournament simulation above).
    proj_winner: dict[int, str] = {}
    proj_loser:  dict[int, str] = {}
    projected_final = {"home": None, "away": None}

    def _resolve_slot(name: str):
        if name.startswith("TBD ("):
            token = name[5:-1]              # "W96" / "L101"
            try:
                src = int(token[1:])
            except ValueError:
                return None
            return (proj_winner if token[0] == "W" else proj_loser).get(src)
        return name

    for m in sorted([x for x in SCHEDULE if x.get("group") in KO_ROUND_LABELS],
                    key=lambda x: x["match_no"]):
        home = _resolve_slot(m["home"])
        away = _resolve_slot(m["away"])
        if not home or not away:
            continue
        if m["match_no"] == 104:
            projected_final = {"home": home, "away": away}
        hs, as_ = m["home_score"], m["away_score"]
        if hs is not None and as_ is not None and is_played(m):
            hp, ap = m.get("home_pens"), m.get("away_pens")
            if hs > as_:
                w, l = home, away
            elif as_ > hs:
                w, l = away, home
            elif hp is not None and ap is not None:
                w, l = (home, away) if hp > ap else (away, home)
            else:
                continue
        else:
            _p = predict_enhanced(home, away, outcome_model, goals_model, learned_elo, stats,
                                  factor_weights=online_factor_weights,
                                  xg_bias_home=online_xg_bias_home,
                                  xg_bias_away=online_xg_bias_away,
                                  extra_draw_inflation=online_draw_inflation)
            w = home if _p["p_home_win"] >= _p["p_away_win"] else away
            l = away if w == home else home
        proj_winner[m["match_no"]] = w
        proj_loser[m["match_no"]]  = l

    if proj_winner.get(104):
        champion = proj_winner[104]
    print(f"[dashboard]   Actual-bracket projection: final "
          f"{projected_final['home']} vs {projected_final['away']} → champion {champion}")

    # ── Add played KO matches to accuracy tracking ───────────────────────────
    ko_accuracy_rows = []
    for entry in actual_ko_schedule:
        if not entry["played"] or entry["home_tbd"] or entry["away_tbd"]:
            continue
        if entry["p_home_win"] is None:
            continue
        m_sched = next((x for x in SCHEDULE if x["match_no"] == entry["match_no"]), None)
        if m_sched is None:
            continue
        av = _accuracy_verdict(entry, m_sched)
        if av:
            ko_accuracy_rows.append({
                "match_no":     entry["match_no"],
                "date":         entry["date"],
                "group":        entry["round"],
                "home":         entry["home"],
                "away":         entry["away"],
                "home_display": display_name(entry["home"]),
                "away_display": display_name(entry["away"]),
                **av,
            })
    accuracy_rows = _acc_group_rows + ko_accuracy_rows

    total_played  = len(accuracy_rows)
    dir_correct   = sum(1 for r in accuracy_rows if r["direction_correct"])
    score_correct = sum(1 for r in accuracy_rows if r["score_correct"])
    print(f"[dashboard]   Accuracy (incl KO): {dir_correct}/{total_played} direction correct")

    accuracy_summary = {
        "total_played":      total_played,
        "direction_correct": dir_correct,
        "direction_pct":     round(100 * dir_correct / max(total_played, 1), 1),
        "score_correct":     score_correct,
        "score_pct":         round(100 * score_correct / max(total_played, 1), 1),
        "matches":           accuracy_rows,
        "elo_changes":       elo_changes,
        "retro": {
            "total":       _acc_retro_total,
            "base_correct":_acc_retro_base,
            "enh_correct": _acc_retro_enh,
            "base_pct":    round(100 * _acc_retro_base / max(_acc_retro_total, 1), 1),
            "enh_pct":     round(100 * _acc_retro_enh  / max(_acc_retro_total, 1), 1),
            "improvement": _acc_retro_enh - _acc_retro_base,
            "rows":        _acc_retro_rows,
        },
    }

    # ── Step 5c: Goalscorer analytics — golden boot, backtest, scorer picks ──
    print("[dashboard] Step 5c — Building goalscorer analytics...")
    from fetch_scorers import fetch_all_scorers
    from update_results import NAME_MAP as _ESPN_NAME_MAP

    def _team_name(espn: str) -> str:
        return _ESPN_NAME_MAP.get(espn.lower(), espn)

    scorers_raw = fetch_all_scorers()

    # Full tally: player → {team, goals, pens}
    player_tally: dict[tuple, dict] = {}
    team_goals_total: dict[str, int] = {}
    for v in scorers_raw.values():
        for g in v["goals"]:
            if g["kind"] == "og" or not g["scorer"]:
                continue
            team = _team_name(g["team"])
            key = (g["scorer"], team)
            rec = player_tally.setdefault(key, {"player": g["scorer"], "team": team,
                                                "goals": 0, "pens": 0})
            rec["goals"] += 1
            if g["kind"] == "pen":
                rec["pens"] += 1
            team_goals_total[team] = team_goals_total.get(team, 0) + 1

    golden_boot = sorted(player_tally.values(), key=lambda r: -r["goals"])[:20]

    by_team: dict[str, list] = {}
    for rec in player_tally.values():
        by_team.setdefault(rec["team"], []).append(
            {"player": rec["player"], "goals": rec["goals"], "pens": rec["pens"]})
    for t in by_team:
        by_team[t].sort(key=lambda r: -r["goals"])
    team_rank = sorted(by_team.items(), key=lambda kv: -sum(r["goals"] for r in kv[1]))
    by_team_list = [{"team": t, "total": sum(r["goals"] for r in rows), "players": rows}
                    for t, rows in team_rank]

    # Walk-forward backtest: pick each team's top scorer using only PRIOR data
    ordered = sorted(scorers_raw.values(), key=lambda v: (v["date"], v["match_no"]))
    running: dict[tuple, int] = {}
    team_matches_seen: dict[str, int] = {}
    bt_picks = 0
    bt_hits  = 0
    bt_rows  = []
    for v in ordered:
        scored_this = {g["scorer"] for g in v["goals"] if g["kind"] != "og"}
        for side in ("home", "away"):
            team = v[side]
            cands = [(p, n) for (p, t), n in running.items() if t == team]
            seen = team_matches_seen.get(team, 0)
            if cands and seen >= 1:
                pick, prior_goals = max(cands, key=lambda x: x[1])
                hit = pick in scored_this
                bt_picks += 1
                bt_hits  += int(hit)
                bt_rows.append({"match_no": v["match_no"], "date": v["date"],
                                "team": team, "pick": pick,
                                "prior_goals": prior_goals, "hit": hit})
        for g in v["goals"]:
            if g["kind"] == "og" or not g["scorer"]:
                continue
            running[(g["scorer"], _team_name(g["team"]))] = \
                running.get((g["scorer"], _team_name(g["team"])), 0) + 1
        for side in ("home", "away"):
            team_matches_seen[v[side]] = team_matches_seen.get(v[side], 0) + 1

    # Matches played per team (for scoring-rate normalisation)
    team_played: dict[str, int] = {}
    for m in SCHEDULE:
        if m["home_score"] is None:
            continue
        for side in ("home", "away"):
            team_played[m[side]] = team_played.get(m[side], 0) + 1

    # Scorer picks for upcoming matches with confirmed pairings
    def _score_prob(player_goals: int, team: str, xg_side: float) -> float:
        tg = team_goals_total.get(team, 0)
        if tg == 0:
            return 0.0
        share = player_goals / tg
        return 1.0 - math.exp(-max(xg_side, 0.1) * share * 1.15)  # 1.15 finish factor

    upcoming_picks = []
    for entry in actual_ko_schedule:
        if entry["played"] or entry["home_tbd"] or entry["away_tbd"]:
            continue
        if entry.get("xg_home") is None:
            continue
        pick_entry = {"match_no": entry["match_no"], "date": entry["date"],
                      "round": entry["round"], "home": entry["home"], "away": entry["away"],
                      "picks": []}
        for side, xg in (("home", entry["xg_home"]), ("away", entry["xg_away"])):
            team = entry[side]
            cands = sorted([r for r in player_tally.values() if r["team"] == team],
                           key=lambda r: -r["goals"])[:3]
            for r in cands:
                p = _score_prob(r["goals"], team, xg)
                if p > 0.05:
                    pick_entry["picks"].append({
                        "player": r["player"], "team": team, "goals": r["goals"],
                        "prob": round(p * 100, 1)})
        pick_entry["picks"].sort(key=lambda x: -x["prob"])
        if pick_entry["picks"]:
            upcoming_picks.append(pick_entry)

    scorers_payload = {
        "golden_boot":   golden_boot,
        "by_team":       by_team_list,
        "total_goals":   sum(team_goals_total.values()),
        "matches_with_data": len(scorers_raw),
        "backtest": {
            "picks": bt_picks, "hits": bt_hits,
            "hit_pct": round(100 * bt_hits / max(bt_picks, 1), 1),
            "rows": bt_rows[-40:],   # most recent 40 for display
        },
        "upcoming": upcoming_picks,
    }
    print(f"[dashboard]   Scorers: {len(player_tally)} players, "
          f"backtest {bt_hits}/{bt_picks} ({100*bt_hits/max(bt_picks,1):.1f}%), "
          f"{len(upcoming_picks)} upcoming matches with picks")

    # ── Step 5d: Team profiles — squads, 2026 stats, scorers, WC history ─────
    print("[dashboard] Step 5d — Building team profiles...")
    from fetch_squads import fetch_all_squads
    from fetch_team_stats import fetch_all_team_stats
    import csv as _csv

    squads_raw  = fetch_all_squads()
    match_stats = fetch_all_team_stats()

    # Assists tally from scorer events
    assist_tally: dict[tuple, int] = {}
    for v in scorers_raw.values():
        for g in v["goals"]:
            if g.get("assist") and g["kind"] != "og":
                k = (g["assist"], _team_name(g["team"]))
                assist_tally[k] = assist_tally.get(k, 0) + 1

    # Aggregate 2026 tournament stats per team
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    team_stats_2026: dict[str, dict] = {}
    for key, ms in match_stats.items():
        sched = next((x for x in SCHEDULE if x["match_no"] == ms["match_no"]), None)
        if sched is None or sched["home_score"] is None:
            continue
        for side, opp in (("home", "away"), ("away", "home")):
            team = ms[side]
            st   = ms.get(f"{side}_stats", {})
            agg  = team_stats_2026.setdefault(team, {
                "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0,
                "shots": 0, "sot": 0, "poss_sum": 0.0, "pass_pct_sum": 0.0,
                "corners": 0, "fouls": 0, "yellows": 0, "reds": 0,
                "offsides": 0, "saves": 0,
            })
            gf = sched[f"{side}_score"]; ga = sched[f"{opp}_score"]
            agg["p"]  += 1
            agg["gf"] += gf; agg["ga"] += ga
            agg["w"]  += int(gf > ga); agg["d"] += int(gf == ga); agg["l"] += int(gf < ga)
            agg["shots"]    += int(_f(st.get("totalShots")))
            agg["sot"]      += int(_f(st.get("shotsOnTarget")))
            agg["poss_sum"] += _f(st.get("possessionPct"))
            agg["pass_pct_sum"] += _f(st.get("passPct"))
            agg["corners"]  += int(_f(st.get("wonCorners")))
            agg["fouls"]    += int(_f(st.get("foulsCommitted")))
            agg["yellows"]  += int(_f(st.get("yellowCards")))
            agg["reds"]     += int(_f(st.get("redCards")))
            agg["offsides"] += int(_f(st.get("offsides")))
            agg["saves"]    += int(_f(st.get("saves")))

    # WC participation history from the historical results CSV
    _hist_alias = {"Congo DR": "DR Congo", "Curacao": "Curaçao"}
    wc_years: dict[str, set] = {}
    try:
        with open("data/results.csv") as _fh:
            for r in _csv.DictReader(_fh):
                if r["tournament"] != "FIFA World Cup":
                    continue
                yr = r["date"][:4]
                for tname in (r["home_team"], r["away_team"]):
                    wc_years.setdefault(tname, set()).add(yr)
    except Exception as e:
        print(f"[dashboard]   WC history error: {e}")

    WC_TITLES = {"Brazil": 5, "Germany": 4, "Italy": 4, "Argentina": 3,
                 "France": 2, "Uruguay": 2, "England": 1, "Spain": 1}

    team_profiles: dict[str, dict] = {}
    for team in [t for grp in WC_2026_GROUPS.values() for t in grp]:
        sq = squads_raw.get(team, {})
        players = sq.get("players", [])
        ages    = [p["age"] for p in players if p.get("age")]
        heights = [p["height_cm"] for p in players if p.get("height_cm")]
        s = team_stats_2026.get(team, {})
        p_ = max(s.get("p", 0), 1)
        hist_name = _hist_alias.get(team, team)
        years = sorted(wc_years.get(hist_name, set()))
        team_profiles[team] = {
            "coach":      sq.get("coach"),
            "squad_size": len(players),
            "avg_age":    round(sum(ages) / len(ages), 1) if ages else None,
            "avg_height": round(sum(heights) / len(heights)) if heights else None,
            "players":    players,
            "stats": ({
                "p": s.get("p", 0), "w": s.get("w", 0), "d": s.get("d", 0), "l": s.get("l", 0),
                "gf": s.get("gf", 0), "ga": s.get("ga", 0),
                "shots": s.get("shots", 0), "sot": s.get("sot", 0),
                "poss": round(s.get("poss_sum", 0) / p_, 1),
                "pass_pct": round(s.get("pass_pct_sum", 0) / p_ * 100, 1),
                "corners": s.get("corners", 0), "fouls": s.get("fouls", 0),
                "yellows": s.get("yellows", 0), "reds": s.get("reds", 0),
                "offsides": s.get("offsides", 0), "saves": s.get("saves", 0),
            } if s.get("p") else None),
            "top_scorers": sorted(
                [{"player": r["player"], "goals": r["goals"], "pens": r["pens"]}
                 for r in player_tally.values() if r["team"] == team],
                key=lambda x: -x["goals"])[:5],
            "top_assists": sorted(
                [{"player": pl, "assists": n}
                 for (pl, tm), n in assist_tally.items() if tm == team],
                key=lambda x: -x["assists"])[:5],
            "wc_appearances": len(years) + (0 if "2026" in years else 1),  # incl. 2026
            "wc_first":       years[0] if years else "2026",
            "wc_titles":      WC_TITLES.get(team, 0),
        }
    print(f"[dashboard]   Team profiles built for {len(team_profiles)} teams "
          f"({sum(1 for t in team_profiles.values() if t['players'])} with squads)")

    # ── Step 6: Team overview table (ELO + model-derived probabilities) ───────
    all_teams = [t for grp in WC_2026_GROUPS.values() for t in grp]
    # Build round-reach prob from deterministic bracket path
    team_ko_rounds: dict[str, str] = {}   # team → deepest round predicted
    for r in ko_rows:
        team_ko_rounds[r["predicted_winner"]] = r["round"]

    # Expected group position per team
    team_exp_pos: dict[str, int] = {}
    for grp, rows in standings.items():
        for row in rows:
            team_exp_pos[row["team"]] = row["exp_pos"]

    win_probs = []
    for team in all_teams:
        ti   = get_intel_features(team)
        hint = TEAM_INTEL.get(team, {})
        grp  = next(g for g, ts in WC_2026_GROUPS.items() if team in ts)
        elo_ = round(learned_elo.get(team, 1500))
        pos  = team_exp_pos.get(team, 4)

        is_champ = (team == champion)
        in_final = is_champ or any(r["round"] == "Final" and team in (r["team1"], r["team2"]) for r in ko_rows)
        in_sf    = in_final or any(r["round"] == "SF"    and team in (r["team1"], r["team2"]) for r in ko_rows)
        in_qf    = in_sf    or any(r["round"] == "QF"    and team in (r["team1"], r["team2"]) for r in ko_rows)
        in_r16   = in_qf    or any(r["round"] == "R16"   and team in (r["team1"], r["team2"]) for r in ko_rows)
        in_r32   = in_r16   or any(r["round"] == "R32"   and team in (r["team1"], r["team2"]) for r in ko_rows)
        # Best 3rd place teams also go to R32 even though they don't finish top-2
        advances = pos <= 2 or in_r32

        win_probs.append({
            "team":         team,
            "group":        grp,
            "elo":          elo_,
            "exp_pos":      pos,         # expected group finish 1–4
            "advances":     advances,    # True = makes knockout round
            "win_pct":      100.0 if is_champ else 0.0,
            "final_pct":    100.0 if in_final else 0.0,
            "sf_pct":       100.0 if in_sf    else 0.0,
            "qf_pct":       100.0 if in_qf    else 0.0,
            "star_rating":  ti["star_rating"],
            "star_form":    ti["star_form"],
            "squad_depth":  ti["squad_depth"],
            "tactical":     ti["tactical_score"],
            "motivation":   ti["political_boost"],
            "carry":        ti["carrying_factor"],
            "style":        hint.get("style", ""),
            "key_players":  hint.get("key_players", []),
            "notes":        hint.get("scouting_notes", ""),
        })
    # Sort by: champion first, then by deepest KO round, then by ELO
    round_order = {"Final": 5, "SF": 4, "QF": 3, "R16": 2, "R32": 1, "": 0}
    win_probs.sort(key=lambda x: (
        1 if x["team"] == champion else 0,
        round_order.get(team_ko_rounds.get(x["team"], ""), 0),
        x["elo"],
    ), reverse=True)

    # ── Date index for all matches (group + knockout) ────────────────────────
    from collections import OrderedDict
    dates_index = OrderedDict()
    for m in sorted(SCHEDULE, key=lambda x: (x["date"], x["match_no"])):
        d = m["date"]
        if d not in dates_index:
            dates_index[d] = []
        dates_index[d].append(m["match_no"])

    # ── Step 6c: Corners predictions — all 104 matches ───────────────────────
    print("[dashboard] Step 6c — Computing corners predictions (all 104 matches)...")
    from corners_model import predict_corners as _predict_corners
    # Actual corners come from ESPN box scores (exact counts)
    try:
        from fetch_corner_stats import fetch_all as _fetch_espn_corners
        espn_data = _fetch_espn_corners(force=False)   # uses cache; only fetches new
    except Exception:
        espn_data = {}
    merged_corners = dict(espn_data)
    # Only keep entries for COMPLETED schedule matches (filter live/future matches)
    _played_keys = {
        f"{m['home'].strip().lower()} vs {m['away'].strip().lower()}"
        for m in SCHEDULE if m.get("home_score") is not None
    }
    merged_corners = {k: v for k, v in merged_corners.items() if k in _played_keys}
    if merged_corners:
        ACTUAL_CORNERS_PATH.write_text(json.dumps(merged_corners, indent=2, ensure_ascii=False))
    actual_corners = load_actual_corners()

    # ── Walk-forward online learning: update team biases from played matches ──
    from corners_learning import run_learning as _run_corners_learning
    try:
        _learning = _run_corners_learning(elo=learned_elo, team_intel=TEAM_INTEL)
        learned_corner_bias = _learning["learned_bias"]
        corner_walk_forward = _learning["walk_forward"]
        print(f"[dashboard]   Corner learning: MAE={_learning['mae']}  "
              f"±2={_learning['within2']}/{_learning['n']}  "
              f"±3={_learning['within3']}/{_learning['n']}")
    except Exception as _le:
        print(f"[dashboard]   Corner learning error: {_le}")
        learned_corner_bias = {}
        corner_walk_forward = []
        _learning = {}

    def _norm_key(h, a):
        return f"{h.strip().lower()} vs {a.strip().lower()}"

    # Build walk_forward lookup by (home, away) for enriching played cards
    _wf_map = {(w["home"].lower(), w["away"].lower()): w for w in corner_walk_forward}

    corners_list = []

    # Group stage (72 matches) — real team names
    for m in sched_group:
        home, away = m["home"], m["away"]
        cp = _predict_corners(
            home=home, away=away,
            home_elo=learned_elo.get(home, 1500),
            away_elo=learned_elo.get(away, 1500),
            home_intel=TEAM_INTEL.get(home, {}),
            away_intel=TEAM_INTEL.get(away, {}),
            extra_bias=learned_corner_bias,   # ← live learned biases for upcoming
        )
        cp["match_no"]     = m["match_no"]
        cp["date"]         = m.get("date", "")
        cp["group"]        = m.get("group", "")
        cp["home_display"] = display_name(home)
        cp["away_display"] = display_name(away)
        cp["played"]       = is_played(m)
        cp["round"]        = "Group"
        if is_played(m):
            cp["actual_score"] = result_str(m)
            ak = actual_corners.get(_norm_key(display_name(home), away)) or \
                 actual_corners.get(_norm_key(home, away))
            if ak:
                cp["actual_total"]        = ak.get("total")
                cp["actual_total_min"]    = ak.get("total_min")
                cp["actual_total_max"]    = ak.get("total_max")
                cp["actual_home_corners"] = ak.get("home_corners")
                cp["actual_away_corners"] = ak.get("away_corners")
            else:
                cp["actual_total"] = None
            # Merge walk-forward learning data (bias before/after, residuals)
            wf = _wf_map.get((home.lower(), away.lower())) or _wf_map.get((away.lower(), home.lower()))
            if wf:
                cp["wf_bias_home_before"] = wf.get("bias_home_at_pred", 0)
                cp["wf_bias_away_before"] = wf.get("bias_away_at_pred", 0)
                cp["wf_bias_home_after"]  = wf.get("bias_home_after")
                cp["wf_bias_away_after"]  = wf.get("bias_away_after")
                cp["wf_res_h"]            = wf.get("res_h")
                cp["wf_res_a"]            = wf.get("res_a")
        corners_list.append(cp)

    # Knockout matches (32) — REAL pairings from the schedule, with results
    for entry in actual_ko_schedule:
        if entry["home_tbd"] or entry["away_tbd"]:
            continue   # pairing not confirmed yet — no corners pick possible
        home, away = entry["home"], entry["away"]
        cp = _predict_corners(
            home=home, away=away,
            home_elo=learned_elo.get(home, 1500),
            away_elo=learned_elo.get(away, 1500),
            home_intel=TEAM_INTEL.get(home, {}),
            away_intel=TEAM_INTEL.get(away, {}),
            extra_bias=learned_corner_bias,
        )
        cp["match_no"]     = entry["match_no"]
        cp["date"]         = entry["date"]
        cp["group"]        = entry["round"]
        cp["home_display"] = display_name(home)
        cp["away_display"] = display_name(away)
        cp["played"]       = entry["played"]
        cp["round"]        = entry["round"]
        if entry["played"]:
            score = f"{entry['home_score']}–{entry['away_score']}"
            if entry.get("home_pens") is not None:
                score += f" (pens {entry['home_pens']}–{entry['away_pens']})"
            cp["actual_score"] = score
            ak = actual_corners.get(_norm_key(display_name(home), away)) or \
                 actual_corners.get(_norm_key(home, away))
            if ak:
                cp["actual_total"]        = ak.get("total")
                cp["actual_total_min"]    = ak.get("total_min")
                cp["actual_total_max"]    = ak.get("total_max")
                cp["actual_home_corners"] = ak.get("home_corners")
                cp["actual_away_corners"] = ak.get("away_corners")
            else:
                cp["actual_total"] = None
        corners_list.append(cp)

    print(f"[dashboard]   Corners predictions built for {len(corners_list)} matches")

    # ── Online learner summary for dashboard display ──────────────────────────
    try:
        ol_summary = learner.get_summary()
    except Exception:
        ol_summary = {}

    return {
        "group_matches":   group_matches,
        "dates_index":     dates_index,
        "standings":       standings,
        "ko_matches":      ko_matches,
        "win_probs":       win_probs,
        "champion":        champion,
        "projected_final": projected_final,
        "team_profiles":   team_profiles,
        "accuracy":        accuracy_summary,
        "method":          "deterministic",
        "corners":         corners_list,
        "scorers":         scorers_payload,
        "corners_learning": {
            "walk_forward":  corner_walk_forward,
            "learned_bias":  learned_corner_bias,
            "mae":           _learning.get("mae"),
            "within2":       _learning.get("within2", 0),
            "within3":       _learning.get("within3", 0),
            "n":             _learning.get("n", 0),
            "alpha":         _learning.get("alpha", 0.45),
        },
        "online_learning": ol_summary,
        "actual_ko_schedule": actual_ko_schedule,
    }


# ── HTML generator ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="Self-learning AI model predicting every 2026 World Cup match — retrains after every result. Random forest + Poisson xG, loop-learned ELO, live ESPN data.">
<title>World Cup 2026 — AI Prediction Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0b1120; --surface: #131c31; --surface2: #1b2740;
    --border: #263351; --text: #e5eaf3; --muted: #8b9bb4;
    --home: #3b82f6; --away: #f97316; --draw: #6b7280;
    --green: #22c55e; --yellow: #eab308; --red: #ef4444;
    --gold: #f59e0b; --purple: #a855f7; --accent: #60a5fa;
    --radius: 12px;
    --shadow: 0 1px 2px rgba(0,0,0,.25), 0 6px 24px rgba(0,0,0,.22);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         font-size: 14px; line-height: 1.5;
         -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
  ::selection { background: #2563eb66; }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }
  ::-webkit-scrollbar-thumb:hover { background: #35507e; }
  button { font-family: inherit; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }

  /* ── Header / hero ── */
  header { background:
             radial-gradient(1000px 340px at 12% -40%, rgba(37,99,235,.28), transparent 60%),
             radial-gradient(800px 300px at 88% -50%, rgba(168,85,247,.14), transparent 55%),
             linear-gradient(180deg, #0e1729 0%, var(--bg) 100%);
           border-bottom: 1px solid var(--border); padding: 26px 24px 20px; }
  .hero-wrap { max-width: 1400px; margin: 0 auto; }
  header h1 { font-size: 26px; font-weight: 800; letter-spacing: -.02em; color: #fff;
              display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  header .sub { color: var(--muted); font-size: 13.5px; margin-top: 6px; max-width: 720px; }
  header .badge { background: rgba(34,197,94,.14); color: #4ade80; border: 1px solid rgba(34,197,94,.4);
                  border-radius: 999px; padding: 3px 12px 3px 9px; font-size: 11px; font-weight: 700;
                  display: inline-flex; align-items: center; gap: 6px; letter-spacing: .04em; }
  .live-dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e;
              animation: pulse 1.6s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,.5); }
                     50% { opacity: .7; box-shadow: 0 0 0 5px rgba(34,197,94,0); } }
  .hero-stats { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
  .hero-chip { background: rgba(19,28,49,.75); border: 1px solid var(--border);
               border-radius: 10px; padding: 8px 14px; backdrop-filter: blur(4px); }
  .hero-chip .hc-val { font-size: 17px; font-weight: 800; font-variant-numeric: tabular-nums; }
  .hero-chip .hc-lbl { font-size: 10.5px; color: var(--muted); text-transform: uppercase;
                       letter-spacing: .06em; margin-top: 1px; }
  .hdr-btn { background: rgba(19,28,49,.75); border: 1px solid var(--border); color: var(--text);
             border-radius: 10px; padding: 7px 14px; font-size: 12.5px; font-weight: 600;
             cursor: pointer; white-space: nowrap; transition: all .18s;
             display: inline-flex; align-items: center; gap: 7px; text-decoration: none; }
  .hdr-btn:hover { border-color: var(--accent); color: #fff; transform: translateY(-1px); }

  /* ── Nav tabs ── */
  nav { background: rgba(11,17,32,.86); backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border); display: flex; gap: 4px;
        padding: 8px 16px; overflow-x: auto; position: sticky; top: 0; z-index: 90;
        scrollbar-width: none; }
  nav::-webkit-scrollbar { display: none; }
  nav button { background: none; border: 1px solid transparent; color: var(--muted); cursor: pointer;
               padding: 8px 15px; font-size: 13px; font-weight: 600; white-space: nowrap;
               border-radius: 999px; transition: all .18s; }
  nav button:hover { color: var(--text); background: var(--surface); }
  nav button.active { color: #fff; background: linear-gradient(135deg, #2563eb, #1d4ed8);
                      border-color: #3b82f6; box-shadow: 0 2px 12px rgba(37,99,235,.35); }

  /* ── Layout ── */
  main { padding: 20px; max-width: 1400px; margin: 0 auto; }
  .tab-content { display: none; } .tab-content.active { display: block; }

  /* ── Cards ── */
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: var(--radius); padding: 16px; margin-bottom: 12px;
          box-shadow: var(--shadow); }
  .card-header { display: flex; justify-content: space-between; align-items: center;
                 margin-bottom: 12px; }
  .card-title { font-weight: 600; font-size: 14px; }

  /* ── Match card (compact, clickable) ── */
  .match-card { background: var(--surface); border: 1px solid var(--border);
                border-radius: var(--radius); margin-bottom: 8px; overflow: hidden;
                cursor: pointer; transition: border-color .15s, transform .1s, box-shadow .15s; }
  .match-card:hover { border-color: var(--accent); transform: translateY(-1px);
                      box-shadow: 0 4px 16px rgba(0,0,0,.3); }
  .match-card:active { transform: translateY(0); }

  /* ── Detail Modal ── */
  #match-modal-overlay {
    display: none; position: fixed; inset: 0; z-index: 9000;
    background: rgba(0,0,0,.65); backdrop-filter: blur(3px);
  }
  #match-modal-overlay.open { display: flex; align-items: flex-start; justify-content: flex-end; }
  #match-modal {
    width: min(680px, 100vw); height: 100vh; overflow-y: auto;
    background: var(--surface); border-left: 1px solid var(--border);
    transform: translateX(100%); transition: transform .22s cubic-bezier(.4,0,.2,1);
    display: flex; flex-direction: column;
  }
  #match-modal-overlay.open #match-modal { transform: translateX(0); }
  #modal-header {
    position: sticky; top: 0; z-index: 10;
    background: var(--surface2); border-bottom: 1px solid var(--border);
    padding: 14px 18px; display: flex; align-items: center; gap: 12px;
  }
  #modal-close {
    margin-left: auto; background: none; border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; padding: 4px 12px; cursor: pointer;
    font-size: 13px; transition: background .15s;
  }
  #modal-close:hover { background: var(--border); }
  #modal-body { padding: 20px 18px; flex: 1; }

  /* Modal sections */
  .modal-section { margin-bottom: 20px; }
  .modal-section-title {
    font-size: 11px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: .07em;
    margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }
  .modal-teams-row {
    display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px;
    align-items: center; margin-bottom: 14px;
  }
  .modal-team { text-align: center; }
  .modal-team-name { font-size: 22px; font-weight: 700; }
  .modal-team-xg { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .modal-vs-block { text-align: center; }
  .modal-score { font-size: 28px; font-weight: 800; color: var(--gold); }
  .modal-score-hint { font-size: 11px; color: var(--muted); }
  .modal-prob-row { display: flex; gap: 8px; justify-content: center; margin: 8px 0 4px; flex-wrap: wrap; }
  .modal-prob-chip {
    padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;
    background: var(--surface2); border: 1px solid var(--border);
  }
  .modal-prob-chip.win-h { background: rgba(96,165,250,.15); border-color: #60a5fa; color: #60a5fa; }
  .modal-prob-chip.win-a { background: rgba(251,146,60,.15); border-color: #fb923c; color: #fb923c; }
  .modal-prob-chip.draw  { background: rgba(148,163,184,.1); border-color: var(--muted); color: var(--muted); }
  .modal-meta-chips { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-top: 10px; font-size: 12px; }
  .modal-chip { background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; color: var(--muted); }
  /* ── Redesigned match card layout ── */
  .mc-meta { display:flex; align-items:center; gap:6px; padding:7px 14px 5px;
             border-bottom:1px solid var(--border); flex-wrap:wrap; }
  .mc-body  { display:grid; grid-template-columns:1fr 140px 1fr;
              align-items:center; padding:10px 14px; gap:8px; }
  .mc-team  { display:flex; flex-direction:column; gap:2px; }
  .mc-team.away { align-items:flex-end; text-align:right; }
  .mc-team-name { font-size:15px; font-weight:700; }
  .mc-team-xg   { font-size:11px; color:var(--muted); }
  .mc-center { text-align:center; }
  .mc-score  { font-size:18px; font-weight:800; color:var(--gold); line-height:1; }
  .mc-vs     { font-size:10px; color:var(--muted); margin:2px 0; }
  .mc-hint   { font-size:10px; color:var(--muted); }
  .mc-favpct { font-size:20px; font-weight:800; line-height:1; }
  .mc-ft-score { font-size:20px; font-weight:800; color:var(--green); }
  .mc-badges { padding:4px 14px 8px; display:flex; gap:5px; flex-wrap:wrap; }
  .mc-badge  { font-size:10px; border-radius:3px; padding:1px 6px; }

  .match-header { display: grid; grid-template-columns: 1fr auto 1fr;
                  gap: 12px; align-items: center; padding: 14px 16px; }
  .team-block { display: flex; flex-direction: column; gap: 4px; }
  .team-block.away { text-align: right; align-items: flex-end; }
  .team-name { font-weight: 700; font-size: 15px; }
  .team-meta { color: var(--muted); font-size: 11px; }
  .match-center { text-align: center; }
  .prob-badge { font-size: 22px; font-weight: 800; }
  .vs-badge { color: var(--muted); font-size: 11px; }
  .score-hint { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* ── Probability bar ── */
  .prob-bar-wrap { padding: 0 16px 10px; }
  .prob-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden;
              background: var(--border); }
  .pb-home { background: var(--home); }
  .pb-draw { background: var(--draw); }
  .pb-away { background: var(--away); }
  .prob-labels { display: flex; justify-content: space-between;
                 font-size: 11px; color: var(--muted); margin-top: 4px; }
  .prob-labels .lbl-home { color: var(--home); font-weight: 600; }
  .prob-labels .lbl-away { color: var(--away); font-weight: 600; }

  /* ── Reasoning toggle ── */
  .reasoning-toggle { width: 100%; background: none; border: none; border-top: 1px solid var(--border);
                       color: #60a5fa; cursor: pointer; padding: 8px 16px;
                       font-size: 12px; text-align: left; display: flex;
                       justify-content: space-between; align-items: center; }
  .reasoning-toggle:hover { background: var(--surface2); }
  .reasoning-body { display: none; padding: 16px; border-top: 1px solid var(--border);
                    background: var(--surface2); }
  .reasoning-body.open { display: block; }

  /* ── Factor chips ── */
  .factors-grid { display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px; }
  .factor { display: flex; align-items: flex-start; gap: 8px; padding: 8px 10px;
            border-radius: 6px; font-size: 12px; }
  .factor.home { border-left: 3px solid var(--home); background: rgba(59,130,246,.1); }
  .factor.away { border-left: 3px solid var(--away); background: rgba(249,115,22,.1); }
  .factor.neutral { border-left: 3px solid var(--muted); background: rgba(148,163,184,.07); }
  .factor-label { font-weight: 600; color: var(--text); white-space: nowrap; min-width: 150px; }
  .factor-text  { color: var(--muted); }

  /* ── Stat grid ── */
  .stat-grid { display: grid; grid-template-columns: 1fr auto 1fr; gap: 4px 8px;
               margin-bottom: 12px; }
  .sg-home { text-align: right; color: #93c5fd; font-size: 12px; }
  .sg-label { text-align: center; color: var(--muted); font-size: 11px; white-space: nowrap; }
  .sg-away { text-align: left; color: #fed7aa; font-size: 12px; }
  .sg-bar-home { height: 4px; background: var(--home); border-radius: 2px;
                 margin-left: auto; }
  .sg-bar-center { display: flex; align-items: center; justify-content: center; }
  .sg-bar-away { height: 4px; background: var(--away); border-radius: 2px; }

  /* ── Risk box ── */
  .risk-box { background: rgba(239,68,68,.1); border: 1px solid rgba(239,68,68,.3);
              border-radius: 6px; padding: 8px 12px; font-size: 12px; margin-bottom: 10px; }
  .risk-box h5 { color: var(--red); font-size: 11px; font-weight: 700; margin-bottom: 4px; }
  .risk-box p { color: var(--muted); }

  /* ── Verdict ── */
  .verdict { background: rgba(96,165,250,.08); border-radius: 6px;
             padding: 10px 14px; font-size: 12px; color: var(--text);
             border-left: 3px solid #60a5fa; }

  /* ── Scouting notes ── */
  .scout-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
  .scout-card { background: var(--bg); border-radius: 6px; padding: 10px; }
  .scout-card h6 { font-size: 11px; font-weight: 700; color: var(--muted);
                   margin-bottom: 4px; text-transform: uppercase; letter-spacing: .05em; }
  .scout-card .players { color: #a78bfa; font-size: 12px; margin-bottom: 4px; }
  .scout-card .note { color: var(--muted); font-size: 11px; line-height: 1.5; }

  /* ── Group grid ── */
  .groups-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr));
                 gap: 16px; margin-bottom: 24px; }
  .group-filter-bar { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
  .group-btn { background: var(--surface); border: 1px solid var(--border); color: var(--muted);
               border-radius: 6px; padding: 5px 12px; cursor: pointer; font-size: 12px; }
  .group-btn:hover, .group-btn.active { background: #1d4ed8; color: #fff; border-color: #1d4ed8; }

  /* ── Standings table ── */
  .standings-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px,1fr)); gap: 16px; }
  .stand-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .stand-table th { color: var(--muted); font-size: 10px; text-transform: uppercase;
                    letter-spacing: .05em; padding: 6px 8px; text-align: right; }
  .stand-table th:first-child { text-align: left; }
  .stand-table td { padding: 7px 8px; border-top: 1px solid var(--border); text-align: right; }
  .stand-table td:first-child { text-align: left; font-weight: 600; }
  .stand-table tr:hover td { background: var(--surface2); }
  .qualify-bar { height: 4px; background: var(--green); border-radius: 2px;
                 display: inline-block; margin-left: 4px; }
  .pct-cell { position: relative; }
  .heat-0  { color: #6b7280; }
  .heat-1  { color: #84cc16; }
  .heat-2  { color: #22c55e; }
  .heat-3  { color: #10b981; }
  .heat-4  { color: #06b6d4; }

  /* ── KO bracket ── */
  .ko-tabs { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
  .ko-tab { background: var(--surface); border: 1px solid var(--border); color: var(--muted);
            border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 13px; }
  .ko-tab.active { background: #7c3aed; color: #fff; border-color: #7c3aed; }

  /* ── Overview ── */
  .overview-top { display: grid; grid-template-columns: repeat(auto-fill,minmax(200px,1fr));
                  gap: 14px; margin-bottom: 24px; }
  .ov-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
             padding: 18px; }
  .ov-card .label { color: var(--muted); font-size: 11px; text-transform: uppercase;
                    letter-spacing: .05em; margin-bottom: 6px; }
  .ov-card .value { font-size: 28px; font-weight: 800; }
  .ov-card .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }

  /* ── Win probability table ── */
  .win-table { width: 100%; border-collapse: collapse; }
  .win-table th { color: var(--muted); font-size: 11px; text-transform: uppercase;
                  letter-spacing: .04em; padding: 8px 10px; text-align: right;
                  border-bottom: 1px solid var(--border); }
  .win-table th:first-child { text-align: left; }
  .win-table th:nth-child(2) { text-align: left; }
  .win-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right; }
  .win-table td:first-child { text-align: left; font-weight: 700; font-size: 14px; }
  .win-table td:nth-child(2) { text-align: left; color: var(--muted); font-size: 12px; }
  .win-table tr:hover td { background: var(--surface2); }
  .rank-num { color: var(--muted); font-size: 12px; font-weight: normal; }
  .wbar { display: inline-block; height: 6px; background: var(--gold);
          border-radius: 3px; vertical-align: middle; margin-left: 6px; }

  /* ── Team profiles (list + detail layout) ── */
  .tp-shell { display: flex; gap: 0; height: calc(100vh - 140px); min-height: 500px; }
  .tp-sidebar {
    width: 260px; min-width: 220px; flex-shrink: 0;
    display: flex; flex-direction: column;
    border-right: 1px solid var(--border);
  }
  .tp-search-wrap { padding: 12px 12px 8px; }
  .tp-search {
    width: 100%; box-sizing: border-box;
    background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); padding: 8px 12px; font-size: 13px; outline: none;
  }
  .tp-search:focus { border-color: var(--home); }
  .tp-sort-row { display: flex; gap: 4px; padding: 0 12px 8px; flex-wrap: wrap; }
  .tp-sort-btn {
    font-size: 10px; padding: 3px 8px; border-radius: 20px; cursor: pointer;
    background: var(--surface2); border: 1px solid var(--border); color: var(--muted);
    transition: all .15s;
  }
  .tp-sort-btn.active { background: var(--home); border-color: var(--home); color: #fff; }
  .tp-list { flex: 1; overflow-y: auto; padding: 0 6px 6px; }
  .tp-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 10px; border-radius: 8px; cursor: pointer;
    transition: background .12s; margin-bottom: 2px;
  }
  .tp-item:hover { background: var(--surface2); }
  .tp-item.selected { background: rgba(99,102,241,.2); border: 1px solid rgba(99,102,241,.4); }
  .tp-flag { font-size: 20px; width: 28px; text-align: center; flex-shrink: 0; }
  .tp-item-info { flex: 1; min-width: 0; }
  .tp-item-name { font-size: 13px; font-weight: 600; white-space: nowrap;
                  overflow: hidden; text-overflow: ellipsis; }
  .tp-item-sub { font-size: 11px; color: var(--muted); }
  .tp-item-elo { font-size: 12px; font-weight: 700; color: var(--gold); flex-shrink: 0; }

  .tp-detail {
    flex: 1; overflow-y: auto; padding: 24px 28px;
    display: flex; flex-direction: column; gap: 20px;
  }
  .tp-empty { display: flex; align-items: center; justify-content: center;
              height: 100%; color: var(--muted); font-size: 14px; }
  .tp-d-hero { display: flex; align-items: flex-start; gap: 20px; }
  .tp-d-flag { font-size: 56px; line-height: 1; }
  .tp-d-title { flex: 1; }
  .tp-d-name { font-size: 26px; font-weight: 800; margin-bottom: 4px; }
  .tp-d-meta { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .tp-d-badge {
    padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
    background: var(--surface2); color: var(--muted); border: 1px solid var(--border);
  }
  .tp-d-elo-big { font-size: 14px; font-weight: 700; color: var(--gold); }
  .tp-win-big { font-size: 13px; color: var(--green); font-weight: 600; }

  .tp-section-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--muted); margin-bottom: 8px; font-weight: 600;
  }
  .tp-d-notes {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; color: var(--text); font-size: 13px; line-height: 1.6;
  }
  .tp-d-players { display: flex; flex-wrap: wrap; gap: 6px; }
  .tp-player-chip {
    background: rgba(139,92,246,.15); border: 1px solid rgba(139,92,246,.3);
    color: #c4b5fd; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 500;
  }
  .tp-d-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px,1fr)); gap: 10px; }
  .tp-stat-box {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px;
  }
  .tp-stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase;
                   letter-spacing: .06em; margin-bottom: 4px; }
  .tp-stat-value { font-size: 22px; font-weight: 800; margin-bottom: 4px; }
  .tp-stat-bar { height: 4px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .tp-stat-fill { height: 100%; border-radius: 3px; }
  .tp-d-prob { display: flex; gap: 10px; flex-wrap: wrap; }
  .tp-prob-box {
    flex: 1; min-width: 100px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px; text-align: center;
  }
  .tp-prob-val { font-size: 20px; font-weight: 800; }
  .tp-prob-label { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* ── Style badge ── */
  .style-badge { display: inline-block; border-radius: 4px; padding: 1px 7px; font-size: 10px;
                 font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
  .style-attacking { background: rgba(239,68,68,.2); color: #fca5a5; }
  .style-tiki-taka { background: rgba(59,130,246,.2); color: #93c5fd; }
  .style-possession { background: rgba(59,130,246,.15); color: #bfdbfe; }
  .style-high-press { background: rgba(234,179,8,.15); color: #fde68a; }
  .style-counter-attack { background: rgba(168,85,247,.15); color: #d8b4fe; }
  .style-defensive { background: rgba(107,114,128,.2); color: #9ca3af; }

  /* ── Date view ── */
  .date-section { margin-bottom: 28px; }
  .date-header { display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
                 padding-bottom: 8px; border-bottom: 2px solid var(--border); }
  .date-label { font-size: 18px; font-weight: 800; color: #fff; }
  .date-sub { color: var(--muted); font-size: 13px; }
  .date-badge { background: var(--surface2); border-radius: 5px; padding: 3px 9px;
                font-size: 11px; color: var(--muted); }
  .today-badge { background: var(--green); color: #000; font-weight: 700; border-radius: 5px;
                 padding: 3px 9px; font-size: 11px; }
  .played-badge { background: rgba(107,114,128,.3); color: var(--muted); border-radius: 5px;
                  padding: 3px 9px; font-size: 11px; }
  .matchday-row { display: grid; grid-template-columns: repeat(auto-fill,minmax(420px,1fr));
                  gap: 10px; }
  .group-pill { display: inline-block; background: #1d4ed8; color: #fff; border-radius: 4px;
                padding: 1px 8px; font-size: 11px; font-weight: 700; margin-right: 4px; }
  .match-meta { display: flex; gap: 10px; align-items: center; padding: 6px 16px 0;
                font-size: 11px; color: var(--muted); }
  .actual-result { background: rgba(34,197,94,.15); border: 1px solid rgba(34,197,94,.4);
                   color: #86efac; border-radius: 6px; padding: 2px 10px; font-weight: 700;
                   font-size: 13px; }
  .filter-bar { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
  .filter-btn { background: var(--surface); border: 1px solid var(--border); color: var(--muted);
                border-radius: 6px; padding: 5px 12px; cursor: pointer; font-size: 12px; }
  .filter-btn:hover, .filter-btn.active { background: #1d4ed8; color: #fff; border-color: #1d4ed8; }

  /* ── Match context panel ── */
  .ctx-panel { border-top: 1px solid var(--border); padding: 14px 16px;
               background: rgba(168,85,247,.04); }
  .ctx-intensity-high   { border-left: 4px solid #ef4444; }
  .ctx-intensity-medium { border-left: 4px solid #eab308; }
  .ctx-intensity-low    { border-left: 4px solid var(--muted); }
  .ctx-headline { font-size: 14px; font-weight: 800; color: #e2e8f0; margin-bottom: 10px; }
  .ctx-section  { margin-bottom: 10px; }
  .ctx-section-title { font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase;
                        letter-spacing:.05em; margin-bottom: 5px; }
  .ctx-bullet { font-size: 12px; color: var(--muted); line-height: 1.6; padding: 4px 0;
                border-bottom: 1px solid rgba(255,255,255,.04); }
  .ctx-bullet:last-child { border-bottom: none; }
  .ctx-pitch { background: var(--bg); border-radius: 6px; padding: 8px 10px; font-size: 12px;
               color: var(--muted); }
  .ctx-impact { background: rgba(168,85,247,.1); border-left: 3px solid #a855f7;
                border-radius: 0 6px 6px 0; padding: 8px 12px; font-size: 12px;
                color: var(--text); margin-top: 8px; }
  .ctx-badge-high   { background: rgba(239,68,68,.2);  color: #fca5a5; border: 1px solid rgba(239,68,68,.4);
                      border-radius: 4px; padding: 1px 8px; font-size: 10px; font-weight: 700; }
  .ctx-badge-medium { background: rgba(234,179,8,.2);  color: #fde68a; border: 1px solid rgba(234,179,8,.4);
                      border-radius: 4px; padding: 1px 8px; font-size: 10px; font-weight: 700; }
  .ctx-badge-low    { background: rgba(148,163,184,.15); color: var(--muted); border: 1px solid var(--border);
                      border-radius: 4px; padding: 1px 8px; font-size: 10px; font-weight: 700; }

    /* ── Star player panel ── */
  .star-panel { border-top: 1px solid var(--border); padding: 12px 16px;
                background: rgba(245,158,11,.04); }
  .star-panel-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .star-card { background: var(--surface); border-radius: 8px; padding: 12px; position: relative;
               border: 1px solid var(--border); }
  .star-card.home-star { border-top: 3px solid var(--home); }
  .star-card.away-star { border-top: 3px solid var(--away); }
  .star-name { font-size: 15px; font-weight: 800; color: #fff; margin-bottom: 2px; }
  .star-club { font-size: 11px; color: var(--gold); font-weight: 600; margin-bottom: 4px; }
  .star-achievement { font-size: 11px; color: #a78bfa; margin-bottom: 6px; font-style: italic; }
  .star-stats { font-size: 11px; color: var(--green); font-weight: 600; margin-bottom: 6px; }
  .star-why { font-size: 12px; color: var(--muted); line-height: 1.5; margin-bottom: 8px; }
  .star-bars { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .star-bar-row { display: flex; flex-direction: column; gap: 2px; }
  .star-bar-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing:.04em; }
  .star-bar-track { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .star-bar-fill { height: 100%; border-radius: 2px; }
  .star-no-data { color: var(--muted); font-size: 12px; font-style: italic; }

  /* ── Discipline panel ── */
  .disc-panel { margin-top: 8px; background: var(--bg); border-radius: 6px; padding: 8px 10px; }
  .disc-title { font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase;
                letter-spacing:.05em; margin-bottom: 6px; }
  .disc-row  { display: flex; flex-wrap: wrap; gap: 6px; }
  .yellow-chip { display: inline-flex; align-items: center; gap: 4px; background: rgba(234,179,8,.15);
                 border: 1px solid rgba(234,179,8,.4); border-radius: 4px; padding: 2px 8px;
                 font-size: 11px; font-weight: 600; color: #fde68a; }
  .red-chip    { display: inline-flex; align-items: center; gap: 4px; background: rgba(239,68,68,.15);
                 border: 1px solid rgba(239,68,68,.4); border-radius: 4px; padding: 2px 8px;
                 font-size: 11px; font-weight: 600; color: #fca5a5; }
  .warn-chip   { background: rgba(239,68,68,.25); border-color: rgba(239,68,68,.6); color: #f87171; }
  .disc-clean  { color: var(--muted); font-size: 11px; font-style: italic; }

    /* ── Factor panel ── */
  .factor-panel { border-top: 1px solid var(--border); padding: 12px 16px;
                  background: rgba(15,23,42,.5); }
  .factor-panel-header { font-size: 11px; font-weight: 700; color: var(--muted);
                          text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px;
                          display: flex; align-items: center; gap: 8px; }
  .factor-chip { display: inline-flex; align-items: center; gap: 4px; border-radius: 4px;
                 padding: 2px 8px; font-size: 10px; font-weight: 700; }
  .factor-chip.player  { background: rgba(168,85,247,.2); color: #d8b4fe; }
  .factor-chip.coach   { background: rgba(59,130,246,.2); color: #93c5fd; }
  .factor-chip.counter { background: rgba(234,179,8,.2);  color: #fde68a; }
  .factor-chip.political { background: rgba(239,68,68,.15); color: #fca5a5; }
  .factor-row { display: flex; align-items: flex-start; gap: 8px; padding: 6px 8px;
                border-radius: 6px; margin-bottom: 4px; font-size: 12px; }
  .factor-row.home-side  { background: rgba(59,130,246,.08); border-left: 3px solid var(--home); }
  .factor-row.away-side  { background: rgba(249,115,22,.08); border-left: 3px solid var(--away); }
  .factor-row.neutral-side { background: rgba(148,163,184,.06); border-left: 3px solid var(--muted); }
  .factor-row-text { color: var(--muted); flex: 1; }
  .factor-coach-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
                      margin-top: 8px; }
  .factor-coach-card { background: var(--bg); border-radius: 6px; padding: 8px 10px; }
  .fc-label { color: var(--muted); font-size: 10px; text-transform: uppercase;
              letter-spacing: .05em; margin-bottom: 2px; }
  .fc-coach { font-size: 13px; font-weight: 700; color: var(--text); }
  .fc-form  { font-size: 11px; color: #93c5fd; }
  .fc-morale { font-size: 11px; }

  /* ── News panel ── */
  .news-panel { border-top: 1px solid var(--border); padding: 12px 16px;
                background: rgba(15,23,42,.6); }
  .news-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .news-header .nh-title { font-size: 12px; font-weight: 600; color: var(--muted);
                           text-transform: uppercase; letter-spacing: .05em; }
  .sentiment-bar { display: flex; gap: 6px; align-items: center; }
  .sent-chip { border-radius: 4px; padding: 2px 7px; font-size: 10px; font-weight: 700;
               text-transform: uppercase; }
  .sent-chip.bullish  { background: rgba(34,197,94,.2);  color: #86efac; }
  .sent-chip.bearish  { background: rgba(239,68,68,.2);  color: #fca5a5; }
  .sent-chip.neutral  { background: rgba(148,163,184,.15); color: var(--muted); }
  .bias-vs-model { font-size: 11px; color: var(--yellow); margin-left: 4px; }
  .news-articles { display: flex; flex-direction: column; gap: 6px; }
  .news-item { display: flex; flex-direction: column; gap: 2px; padding: 7px 10px;
               background: var(--surface); border-radius: 6px; border-left: 3px solid var(--border); }
  .news-item.bullish-home  { border-left-color: var(--home); }
  .news-item.bearish-home  { border-left-color: var(--red); }
  .news-item.bullish-away  { border-left-color: var(--away); }
  .news-item.bearish-away  { border-left-color: var(--red); }
  .ni-title  { font-size: 12px; color: var(--text); line-height: 1.4; }
  .ni-meta   { font-size: 10px; color: var(--muted); display: flex; gap: 8px; }
  .ni-sents  { display: flex; gap: 4px; }
  .no-news   { color: var(--muted); font-size: 11px; font-style: italic; }

  /* ── Accuracy tab ── */
  .acc-summary { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px,1fr));
                 gap: 14px; margin-bottom: 24px; }
  .acc-stat { background: var(--surface); border: 1px solid var(--border);
              border-radius: var(--radius); padding: 18px; text-align: center; }
  .acc-stat .as-val { font-size: 36px; font-weight: 800; }
  .acc-stat .as-label { color: var(--muted); font-size: 11px; text-transform: uppercase;
                        letter-spacing: .05em; margin-top: 4px; }
  .acc-match { background: var(--surface); border: 1px solid var(--border);
               border-radius: var(--radius); padding: 16px; margin-bottom: 12px;
               display: grid; grid-template-columns: auto 1fr; gap: 16px; align-items: start; }
  .acc-icon { font-size: 32px; line-height: 1; }
  .acc-teams { font-size: 15px; font-weight: 700; margin-bottom: 6px; }
  .acc-row { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 8px; }
  .acc-box { background: var(--bg); border-radius: 6px; padding: 8px 12px; }
  .acc-box .ab-label { color: var(--muted); font-size: 10px; text-transform: uppercase;
                       letter-spacing: .05em; margin-bottom: 3px; }
  .acc-box .ab-val { font-size: 14px; font-weight: 700; }
  .correct-badge { display: inline-block; border-radius: 4px; padding: 2px 8px;
                   font-size: 11px; font-weight: 700; }
  .correct-badge.ok { background: rgba(34,197,94,.2); color: #86efac; }
  .correct-badge.fail { background: rgba(239,68,68,.2); color: #fca5a5; }
  .elo-change-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .elo-chip { border-radius: 6px; padding: 4px 10px; font-size: 12px; font-weight: 600; }
  .elo-chip.up   { background: rgba(34,197,94,.15); color: #86efac; }
  .elo-chip.down { background: rgba(239,68,68,.15); color: #fca5a5; }
  /* Played match accuracy badge in match card */
  .acc-verdict { display: flex; align-items: center; gap: 6px; padding: 6px 16px 8px;
                 font-size: 12px; border-top: 1px solid var(--border); background: var(--surface2); }
  .acc-verdict .av-pre { color: var(--muted); }
  .acc-verdict .av-score { font-weight: 700; }

  /* ── Markets tab ── */
  .mkts-no-data { text-align:center; padding:60px 20px; color:var(--muted); }
  .mkts-no-data h2 { font-size:22px; margin-bottom:12px; }
  .mkts-no-data code { background:var(--surface2); padding:2px 8px; border-radius:4px; font-family:monospace; }
  .mkts-summary { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:18px; }
  .mkts-kpi { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
              padding:14px 18px; flex:1; min-width:130px; }
  .mkts-kpi .k-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .mkts-kpi .k-val   { font-size:22px; font-weight:700; margin-top:4px; }
  .mkts-edge-table { width:100%; border-collapse:collapse; font-size:13px; }
  .mkts-edge-table th { color:var(--muted); font-weight:500; padding:6px 10px; text-align:left;
                        border-bottom:1px solid var(--border); font-size:11px; text-transform:uppercase; }
  .mkts-edge-table td { padding:8px 10px; border-bottom:1px solid var(--border); vertical-align:middle; }
  .mkts-edge-table tr:hover td { background:var(--surface2); }
  .edge-pos { color:#22c55e; font-weight:700; }
  .edge-neg { color:#ef4444; }
  .flow-bar-wrap { display:flex; align-items:center; gap:6px; min-width:120px; }
  .flow-bar { height:8px; border-radius:4px; flex:1; display:flex; overflow:hidden; }
  .flow-bid { background:#22c55e; }
  .flow-ask { background:#ef4444; }
  .flow-label { font-size:11px; color:var(--muted); white-space:nowrap; }
  .bet-chip { display:inline-block; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700; }
  .bet-yes  { background:rgba(34,197,94,.2); color:#86efac; border:1px solid #22c55e55; }
  .bet-no   { background:rgba(100,100,100,.15); color:var(--muted); border:1px solid var(--border); }
  .sparkline { display:inline-block; vertical-align:middle; }

  /* ── (legacy chips) ── */
  .props-match-selector { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:20px; }
  .props-match-btn { background:var(--surface); border:1px solid var(--border); color:var(--muted);
                     padding:6px 12px; border-radius:6px; cursor:pointer; font-size:12px; text-align:left; }
  .props-match-btn.active { background:rgba(96,165,250,.15); color:#60a5fa; border-color:#60a5fa55; }
  .props-match-btn .pmb-score { font-size:10px; color:var(--muted); display:block; }
  .props-layout { display:grid; grid-template-columns: 260px 1fr; gap:16px; align-items:start; }
  .props-cat-nav { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
                   overflow:hidden; position:sticky; top:12px; }
  .props-cat-btn { display:block; width:100%; text-align:left; background:none; border:none;
                   color:var(--muted); cursor:pointer; padding:10px 16px; font-size:13px;
                   border-bottom:1px solid var(--border); transition:all .15s; }
  .props-cat-btn:hover { background:var(--surface2); color:var(--text); }
  .props-cat-btn.active { background:rgba(96,165,250,.12); color:#60a5fa; font-weight:600; }
  .props-panel { }
  .props-section { background:var(--surface); border:1px solid var(--border);
                   border-radius:var(--radius); margin-bottom:16px; overflow:hidden; }
  .props-section-header { background:var(--surface2); padding:10px 16px;
                           font-size:13px; font-weight:700; border-bottom:1px solid var(--border);
                           display:flex; align-items:center; gap:8px; }
  .props-grid { display:grid; grid-template-columns: 1fr 1fr; gap:0; }
  .prop-row { display:flex; align-items:center; padding:8px 14px; border-bottom:1px solid var(--border); gap:8px; }
  .prop-row:last-child { border-bottom:none; }
  .prop-row:hover { background:var(--surface2); }
  .prop-label { flex:1; font-size:12px; color:var(--text); }
  .prop-prob-bar { width:80px; height:6px; background:var(--border); border-radius:3px; overflow:hidden; }
  .prop-prob-fill { height:100%; border-radius:3px; transition:width .3s; }
  .prop-prob-pct { width:42px; text-align:right; font-size:12px; font-weight:600; }
  .prop-odds { width:44px; text-align:right; font-size:11px; color:var(--muted); }
  .prop-tip { font-size:10px; white-space:nowrap; }
  .prop-high { color:#22c55e; }
  .prop-low  { color:#ef4444; }
  .prop-mid  { color:#eab308; }
  .props-xg-strip { display:flex; gap:16px; padding:12px 16px; background:var(--surface2);
                    border-bottom:1px solid var(--border); font-size:12px; flex-wrap:wrap; }
  .props-xg-item { }
  .props-xg-item .xi-label { color:var(--muted); font-size:10px; }
  .props-xg-item .xi-val   { font-weight:700; font-size:16px; }
  .exact-score-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0; }
  .exact-score-cell { padding:8px 12px; border:1px solid var(--border); text-align:center;
                      cursor:default; transition:background .15s; }
  .exact-score-cell:hover { background:var(--surface2); }
  .esc-score { font-size:14px; font-weight:700; }
  .esc-prob  { font-size:11px; color:var(--muted); }
  .esc-odds  { font-size:10px; color:#60a5fa; }
  .esc-h     { border-left:2px solid #3b82f655; }
  .esc-d     { border-left:2px solid #6b728055; }
  .esc-a     { border-left:2px solid #f9731655; }

  /* ── Live Flow tab ── */
  .flow-no-data { text-align:center; padding:60px 20px; color:var(--muted); }
  .flow-no-data h2 { font-size:22px; margin-bottom:12px; }
  .flow-no-data code { background:var(--surface2); padding:2px 8px; border-radius:4px; font-family:monospace; }
  .flow-match-card { background:var(--surface); border:1px solid var(--border);
                     border-radius:var(--radius); margin-bottom:20px; overflow:hidden; }
  .flow-card-header { display:flex; align-items:center; justify-content:space-between;
                      padding:12px 16px; background:var(--surface2); border-bottom:1px solid var(--border); }
  .flow-teams { font-size:16px; font-weight:700; }
  .flow-meta { color:var(--muted); font-size:11px; display:flex; gap:12px; align-items:center; }
  .flow-badge-live { background:#ef4444; color:#fff; border-radius:4px; padding:2px 8px; font-size:10px; font-weight:700; animation:pulse 1s infinite; }
  .flow-badge-done { background:var(--border); color:var(--muted); border-radius:4px; padding:2px 8px; font-size:10px; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.5; } }
  .flow-chart-wrap { padding:16px; }
  .flow-chart-title { font-size:12px; color:var(--muted); margin-bottom:8px; text-transform:uppercase; letter-spacing:.05em; }
  .flow-prob-chart { width:100%; display:block; }
  .flow-vol-chart  { width:100%; display:block; margin-top:8px; }
  .flow-events { display:flex; flex-wrap:wrap; gap:6px; padding:0 16px 12px; }
  .flow-event-chip { border-radius:4px; padding:3px 10px; font-size:11px; font-weight:600; }
  .flow-event-home { background:rgba(59,130,246,.2); color:#93c5fd; border:1px solid #3b82f655; }
  .flow-event-away { background:rgba(249,115,22,.2); color:#fed7aa; border:1px solid #f9731655; }
  .flow-event-draw { background:rgba(107,114,128,.2); color:var(--muted); border:1px solid var(--border); }
  .flow-probs-summary { display:flex; gap:12px; padding:12px 16px; border-top:1px solid var(--border); }
  .flow-prob-pill { flex:1; text-align:center; padding:8px; border-radius:8px; }
  .flow-prob-pill .fp-label { font-size:10px; color:var(--muted); text-transform:uppercase; }
  .flow-prob-pill .fp-pre  { font-size:12px; color:var(--muted); }
  .flow-prob-pill .fp-now  { font-size:18px; font-weight:700; }
  .flow-filter-bar { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }
  .flow-filter-btn { background:var(--surface); border:1px solid var(--border); color:var(--muted);
                     padding:6px 14px; border-radius:6px; cursor:pointer; font-size:12px; }
  .flow-filter-btn.active { background:rgba(96,165,250,.15); color:#60a5fa; border-color:#60a5fa55; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Footer ── */
  footer { border-top: 1px solid var(--border); margin-top: 48px;
           padding: 28px 24px 36px; background: linear-gradient(180deg, var(--bg), #0d1526); }
  .foot-wrap { max-width: 1400px; margin: 0 auto; display: flex;
               justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap; }
  .foot-title { font-weight: 700; font-size: 14px; margin-bottom: 6px; }
  .foot-sub { color: var(--muted); font-size: 12.5px; max-width: 520px; line-height: 1.6; }
  .tech-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
  .tech-chip { background: var(--surface); border: 1px solid var(--border); color: var(--muted);
               border-radius: 999px; padding: 3px 11px; font-size: 11px; font-weight: 600; }

  /* ── About modal ── */
  #about-overlay { display: none; position: fixed; inset: 0; background: rgba(4,8,18,.72);
                   backdrop-filter: blur(6px); z-index: 200; overflow-y: auto; padding: 5vh 16px; }
  #about-overlay.open { display: block; }
  #about-modal { max-width: 680px; margin: 0 auto; background: var(--surface);
                 border: 1px solid var(--border); border-radius: 16px; padding: 28px;
                 box-shadow: 0 24px 80px rgba(0,0,0,.5); }
  #about-modal h2 { font-size: 20px; font-weight: 800; letter-spacing: -.01em; margin-bottom: 4px; }
  #about-modal h3 { font-size: 13px; font-weight: 700; color: var(--accent);
                    text-transform: uppercase; letter-spacing: .07em; margin: 20px 0 8px; }
  #about-modal p, #about-modal li { color: #c4cede; font-size: 13.5px; line-height: 1.65; }
  #about-modal ul { padding-left: 20px; }
  #about-modal code { background: var(--surface2); border: 1px solid var(--border);
                      padding: 1px 6px; border-radius: 5px; font-size: 12px; }

  /* ── Responsive ── */
  @media(max-width:640px) {
    .scout-row { grid-template-columns: 1fr; }
    .stat-grid { font-size: 11px; }
    .matchday-row { grid-template-columns: 1fr; }
    header { padding: 20px 16px 16px; }
    header h1 { font-size: 20px; }
    header .sub { font-size: 12px; }
    .hero-side { display: none; }
    .hero-stats { gap: 7px; }
    .hero-chip { padding: 6px 11px; }
    .hero-chip .hc-val { font-size: 14px; }
    main { padding: 14px 12px; }
  }
</style>
</head>
<body>

<header>
  <div class="hero-wrap">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
      <div>
        <h1>⚽ World Cup 2026 — AI Prediction Engine
          <span class="badge"><span class="live-dot"></span>LIVE</span>
        </h1>
        <div class="sub">A self-learning machine-learning model that predicts every match of the 2026 World Cup —
          and retrains itself after every final whistle. Random forest + Poisson xG, loop-learned ELO,
          live ESPN data.</div>
      </div>
      <div class="hero-side" style="display:flex;flex-direction:column;align-items:flex-end;gap:8px">
        <div id="utc7-clock" style="text-align:right;color:#cbd5e1;font-size:13px;min-width:150px">
          <div style="font-size:10.5px;color:#8b9bb4;text-transform:uppercase;letter-spacing:.05em">🕐 Vietnam (UTC+7)</div>
          <div id="utc7-time" style="font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;color:#f1f5f9">--:--:--</div>
          <div id="utc7-date" style="font-size:11px;color:#8b9bb4"></div>
        </div>
        <div style="display:flex;gap:8px">
          <a class="hdr-btn" href="https://github.com/xuanbachmai/wc2026-prediction-dashboard" target="_blank" rel="noopener">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
            Source
          </a>
          <button class="hdr-btn" onclick="openAbout()">ℹ️ How it works</button>
          <button class="hdr-btn" id="rebuild-btn" onclick="triggerRebuild()">⚡ Refresh</button>
        </div>
      </div>
    </div>
    <div class="hero-stats" id="hero-stats"></div>
  </div>
</header>

<nav>
  <button class="active" onclick="showTab('overview')">🏆 Overview</button>
  <button onclick="showTab('groups')">📅 Schedule by Date</button>
  <button onclick="showTab('bygroup')">📋 By Group</button>
  <button onclick="showTab('standings')">📊 Standings</button>
  <button onclick="showTab('knockout')">⚔️ Knockout</button>
  <button onclick="showTab('teams')">👤 Team Profiles</button>
  <button onclick="showTab('accuracy')" id="acc-tab-btn">🎯 Accuracy</button>
  <button onclick="showTab('scorers')" id="scorers-tab-btn">⚽ Scorers</button>
  <button onclick="showTab('corners')" id="corners-tab-btn">🚩 Corners Model</button>
</nav>

<main>

<!-- ── Match Detail Modal ─────────────────────────────────────────────── -->
<div id="match-modal-overlay" onclick="closeModal(event)">
  <div id="match-modal">
    <div id="modal-header">
      <span id="modal-title" style="font-weight:700;font-size:15px"></span>
      <button id="modal-close" onclick="closeMatchModal()">✕ Close</button>
    </div>
    <div id="modal-body"></div>
  </div>
</div>

<!-- ══════════════════════ OVERVIEW ══════════════════════ -->
<div id="tab-overview" class="tab-content active">
  <div id="overview-top" class="overview-top"></div>
  <div class="card">
    <div class="card-header">
      <span class="card-title">🏆 Tournament Win Probability — All 48 Teams</span>
    </div>
    <table class="win-table" id="win-table"></table>
  </div>
</div>

<!-- ══════════════════════ SCHEDULE BY DATE ══════════════════════ -->
<div id="tab-groups" class="tab-content">
  <div class="filter-bar" id="date-filter-bar">
    <button class="filter-btn active" onclick="filterDate('ALL', this)">All Dates</button>
    <button class="filter-btn" onclick="filterDate('MD1', this)">Matchday 1</button>
    <button class="filter-btn" onclick="filterDate('MD2', this)">Matchday 2</button>
    <button class="filter-btn" onclick="filterDate('MD3', this)">Matchday 3</button>
  </div>
  <div id="date-matches-container"></div>
</div>

<!-- ══════════════════════ BY GROUP ══════════════════════ -->
<div id="tab-bygroup" class="tab-content">
  <div class="filter-bar" id="group-filter-bar">
    <button class="filter-btn active" onclick="filterGroup('ALL', this)">All Groups</button>
  </div>
  <div id="group-matches-container"></div>
</div>

<!-- ══════════════════════ STANDINGS ══════════════════════ -->
<div id="tab-standings" class="tab-content">
  <div class="standings-grid" id="standings-grid"></div>
</div>


<!-- ══════════════════════ KNOCKOUT ══════════════════════ -->
<div id="tab-knockout" class="tab-content">
  <div class="ko-tabs" id="ko-sched-tabs"></div>
  <div id="ko-sched-container"></div>
</div>

<!-- ══════════════════════ TEAM PROFILES ══════════════════════ -->
<div id="tab-teams" class="tab-content">
  <div class="tp-shell">
    <!-- Sidebar: search + sorted list -->
    <div class="tp-sidebar">
      <div class="tp-search-wrap">
        <input class="tp-search" id="tp-search" placeholder="🔍  Search team…" oninput="filterTeams()" autocomplete="off">
      </div>
      <div class="tp-sort-row">
        <button class="tp-sort-btn active" onclick="sortTeams('win_pct')"  id="tsb-win">🏆 Win%</button>
        <button class="tp-sort-btn"        onclick="sortTeams('elo')"      id="tsb-elo">📈 ELO</button>
        <button class="tp-sort-btn"        onclick="sortTeams('group')"    id="tsb-grp">🔤 Group</button>
        <button class="tp-sort-btn"        onclick="sortTeams('name')"     id="tsb-name">A–Z</button>
      </div>
      <div class="tp-list" id="tp-list"></div>
    </div>
    <!-- Detail panel -->
    <div class="tp-detail" id="tp-detail">
      <div class="tp-empty">← Select a team to see their profile</div>
    </div>
  </div>
</div>

<!-- ══════════════════════ ACCURACY TRACKER ══════════════════════ -->
<div id="tab-accuracy" class="tab-content">
  <div class="acc-summary" id="acc-summary"></div>
  <div class="card" style="margin-bottom:20px">
    <div class="card-header">
      <span class="card-title">🔄 Loop Learning — ELO Updates from Actual Results</span>
    </div>
    <p style="color:var(--muted);font-size:12px;margin-bottom:8px">
      After every played match, the model updates each team's ELO rating (K=60 WC weight).
      These updated ratings feed directly into all upcoming match predictions.
    </p>
    <div class="elo-change-list" id="elo-change-list"></div>
  </div>
  <div id="acc-matches"></div>

  <!-- ── Self-Learning Engine Status ──────────────────────────────────── -->
  <div class="card" style="margin-top:24px" id="ol-card">
    <div class="card-header">
      <span class="card-title">🧠 Self-Learning Engine — What the Model Has Learned</span>
    </div>
    <div id="ol-content">
      <p style="color:var(--muted);font-size:13px">Loading…</p>
    </div>
  </div>
</div>

<!-- ══════════════════════ 🚩 CORNERS & MARKETS ══════════════════════ -->
<div id="tab-scorers" class="tab-content">
  <div id="scorers-container"></div>
</div>

<div id="tab-corners" class="tab-content">

  <!-- Country corner search -->
  <div style="margin-bottom:16px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
      <input id="corner-search" type="text" placeholder="🔍  Search country…"
        oninput="filterCornerSearch()"
        style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px 14px;color:var(--fg);font-size:13px;width:240px;outline:none">
      <span id="corner-search-hint" style="font-size:11px;color:var(--muted)">Type a team name to see their corner history</span>
    </div>
    <div id="corner-search-results"></div>
  </div>

  <div id="corners-container"></div>
</div>


</main>

<!-- ── About modal ── -->
<div id="about-overlay" onclick="if(event.target===this)closeAbout()">
  <div id="about-modal">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
      <h2>How this project works</h2>
      <button class="hdr-btn" onclick="closeAbout()">✕</button>
    </div>
    <p style="color:var(--muted);font-size:12.5px">A self-learning World Cup prediction system — model, backend, and live data pipeline.</p>

    <h3>The model</h3>
    <ul>
      <li>A <b>random-forest classifier</b> predicts home / draw / away from ~50k historical internationals; a <b>Poisson regression</b> estimates expected goals (xG) per side.</li>
      <li>A <b>Dixon-Coles matrix</b> converts xG into full scoreline probabilities.</li>
      <li>Prediction factors: loop-learned ELO, squad quality, star-player form, tactics, injuries/suspensions and news sentiment.</li>
    </ul>

    <h3>The learning loop</h3>
    <ul>
      <li>After every final whistle the backend re-trains: <b>ELO ratings update</b> (K=60 World Cup weight), online-learned xG biases shift, and every remaining prediction is recomputed.</li>
      <li>An <b>accuracy tracker</b> scores every prediction against the real result — nothing is hidden or retro-fitted.</li>
      <li>The corners model and goalscorer picks are <b>walk-forward backtested</b>: each pick uses only data available before that match.</li>
    </ul>

    <h3>The live backend</h3>
    <ul>
      <li><code>serve.py</code> polls ESPN every 3 seconds during matches, writes goals, penalties and shootouts into the schedule, resolves bracket slots, and pushes <b>SSE live updates</b> into open browsers.</li>
      <li>Missed results self-heal: a backfill pass re-fetches by date at boot and hourly, and a verify pass re-checks every recorded score against ESPN.</li>
      <li>A GitHub Actions cron mirrors the same pipeline in CI, keeping the hosted page current around the clock.</li>
    </ul>

    <h3>Stack</h3>
    <p>Python · scikit-learn · SciPy · pandas · vanilla JS (zero front-end dependencies) · GitHub Actions · Render</p>
  </div>
</div>

<footer>
  <div class="foot-wrap">
    <div>
      <div class="foot-title">⚽ World Cup 2026 — AI Prediction Engine</div>
      <div class="foot-sub">A self-learning football prediction model built end-to-end: data pipeline, ML model,
        live backend and this dashboard. Predictions are recomputed automatically after every match.
        Educational / portfolio project — not gambling advice.</div>
      <div class="tech-chips">
        <span class="tech-chip">Python</span><span class="tech-chip">scikit-learn</span>
        <span class="tech-chip">Poisson + Dixon-Coles</span><span class="tech-chip">ELO loop-learning</span>
        <span class="tech-chip">SSE live updates</span><span class="tech-chip">GitHub Actions</span>
      </div>
    </div>
    <div style="text-align:right">
      <div class="foot-title">Xuan Bach Mai</div>
      <a class="hdr-btn" href="https://github.com/xuanbachmai/wc2026-prediction-dashboard" target="_blank" rel="noopener" style="margin-top:6px">
        View source on GitHub →
      </a>
    </div>
  </div>
</footer>

<script>
const DATA = {{DATA_JSON}};

// ── Corner search by country ───────────────────────────────────────────────
function filterCornerSearch() {
  const q = document.getElementById('corner-search').value.trim().toLowerCase();
  const el = document.getElementById('corner-search-results');
  const hint = document.getElementById('corner-search-hint');
  if (!q) { el.innerHTML = ''; hint.style.display = ''; return; }
  hint.style.display = 'none';

  const corners = DATA.corners || [];
  const matches = corners.filter(c =>
    c.actual_home_corners != null &&  // played only
    (c.home.toLowerCase().includes(q) || c.away.toLowerCase().includes(q))
  );

  if (!matches.length) {
    el.innerHTML = `<div style="color:var(--muted);font-size:12px;padding:8px 0">No played matches found for "<b>${q}</b>"</div>`;
    return;
  }

  let totalFor = 0, totalAgainst = 0, played = 0;
  const rows = matches.map(c => {
    const isHome = c.home.toLowerCase().includes(q);
    const team   = isHome ? c.home : c.away;
    const opp    = isHome ? c.away : c.home;
    const forC   = isHome ? c.actual_home_corners : c.actual_away_corners;
    const agstC  = isHome ? c.actual_away_corners : c.actual_home_corners;
    const total  = c.actual_total;
    if (forC != null) { totalFor += forC; totalAgainst += agstC; played++; }
    const ha = isHome ? 'H' : 'A';
    const forCol  = forC >= 6 ? '#22c55e' : forC >= 4 ? '#f59e0b' : '#ef4444';
    const agstCol = agstC >= 6 ? '#ef4444' : agstC >= 4 ? '#f59e0b' : '#22c55e';
    return `<div style="display:flex;align-items:center;gap:10px;padding:6px 10px;border-radius:6px;background:var(--card);margin-bottom:4px">
      <span style="font-size:10px;color:var(--muted);min-width:80px">${c.date}</span>
      <span style="font-size:9px;color:var(--muted);background:var(--border);padding:1px 5px;border-radius:4px">${ha}</span>
      <span style="font-size:12px;font-weight:600;min-width:120px">${team}</span>
      <span style="font-size:11px;color:var(--muted)">vs ${opp}</span>
      <span style="margin-left:auto;display:flex;gap:14px;align-items:center">
        <span title="Corners for"><span style="font-size:10px;color:var(--muted)">CK </span><b style="color:${forCol};font-size:15px">${forC ?? '?'}</b></span>
        <span title="Corners against"><span style="font-size:10px;color:var(--muted)">vs </span><b style="color:${agstCol};font-size:15px">${agstC ?? '?'}</b></span>
        <span title="Total"><span style="font-size:10px;color:var(--muted)">tot </span><b style="font-size:13px">${total ?? '?'}</b></span>
      </span>
    </div>`;
  }).join('');

  const avgFor   = played ? (totalFor   / played).toFixed(1) : '—';
  const avgAgst  = played ? (totalAgainst / played).toFixed(1) : '—';
  const avgTotal = played ? ((totalFor + totalAgainst) / played).toFixed(1) : '—';

  el.innerHTML = `
    <div style="margin-bottom:8px;padding:8px 12px;border-radius:8px;background:var(--card);border:1px solid var(--border);display:flex;gap:24px;align-items:center">
      <span style="font-size:12px;font-weight:700">${matches[0].home.toLowerCase().includes(q)?matches[0].home:matches[0].away}</span>
      <span style="font-size:11px;color:var(--muted)">${played} match${played>1?'es':''}</span>
      <span><span style="font-size:10px;color:var(--muted)">Avg CK for </span><b style="color:#22c55e">${avgFor}</b></span>
      <span><span style="font-size:10px;color:var(--muted)">Avg CK against </span><b style="color:#ef4444">${avgAgst}</b></span>
      <span><span style="font-size:10px;color:var(--muted)">Avg total </span><b>${avgTotal}</b></span>
    </div>
    ${rows}`;
}

// ── Tab navigation ─────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

// ── Helpers ────────────────────────────────────────────────────────────────
function pct(v) { return v.toFixed(1) + '%'; }
function bar(v, max=10) {
  return `<div class="ts-bar"><div class="ts-fill" style="width:${v/max*100}%"></div></div>`;
}
function heatClass(v) {
  if (v >= 70) return 'heat-4';
  if (v >= 50) return 'heat-3';
  if (v >= 30) return 'heat-2';
  if (v >= 15) return 'heat-1';
  return 'heat-0';
}
function styleBadge(s) {
  return s ? `<span class="style-badge style-${s.replace(' ','-')}">${s}</span>` : '';
}

// ── Probability bar HTML ───────────────────────────────────────────────────
function probBar(ph, pd, pa) {
  ph = ph ?? 0; pd = pd ?? 0; pa = pa ?? 0;
  const total = (ph + pd + pa) || 1;
  const wh = ph/total*100, wd = pd/total*100, wa = pa/total*100;
  return `
    <div class="prob-bar-wrap">
      <div class="prob-bar">
        <div class="pb-home" style="width:${wh}%"></div>
        <div class="pb-draw" style="width:${wd}%"></div>
        <div class="pb-away" style="width:${wa}%"></div>
      </div>
      <div class="prob-labels">
        <span class="lbl-home">${ph.toFixed(1)}% Win</span>
        <span>${pd.toFixed(1)}% Draw</span>
        <span class="lbl-away">${pa.toFixed(1)}% Win</span>
      </div>
    </div>`;
}

// ── Reasoning HTML ─────────────────────────────────────────────────────────
function reasoningHTML(r, m) {
  if (!r || !r.factors) return '<div style="color:var(--muted);font-size:12px">Prediction from loop-learned ELO + online-learned model — detailed factor breakdown available for group-stage matches.</div>';
  const factors = r.factors.map(f => `
    <div class="factor ${f.side}">
      <span>${f.icon}</span>
      <span class="factor-label">${f.label}</span>
      <span class="factor-text">${f.text}</span>
    </div>`).join('');

  const risks = r.risks.length ? `
    <div class="risk-box">
      <h5>⚠️ Key Risks</h5>
      ${r.risks.map(rk => `<p>${rk}</p>`).join('')}
    </div>` : '';

  const statRows = [
    ['ELO Rating', r.home_elo, r.away_elo, 1200, 2000],
    ['⭐ Star Rating', r.home_star, r.away_star, 0, 10],
    ['👥 Squad Depth', r.home_depth, r.away_depth, 0, 10],
    ['🎯 Tactics Score', r.home_tactics, r.away_tactics, 0, 10],
    ['🏟️ Motivation', r.home_motivation, r.away_motivation, 0, 10],
    ['⚠️ Carry Risk', r.home_carry, r.away_carry, 0, 10],
  ].map(([label, hv, av, mn, mx]) => {
    const range = mx - mn;
    const hw = ((hv - mn) / range * 100).toFixed(1);
    const aw = ((av - mn) / range * 100).toFixed(1);
    const hc = hv >= av ? '#93c5fd' : '#94a3b8';
    const ac = av > hv ? '#fed7aa' : '#94a3b8';
    return `
      <div class="sg-home" style="color:${hc}">${typeof hv === 'number' && hv > 100 ? hv : hv.toFixed ? hv.toFixed(1) : hv}</div>
      <div class="sg-label">${label}</div>
      <div class="sg-away" style="color:${ac}">${typeof av === 'number' && av > 100 ? av : av.toFixed ? av.toFixed(1) : av}</div>`;
  }).join('');

  return `
    <div class="factors-grid">${factors || '<p style="color:var(--muted);font-size:12px">Teams evenly matched across all factors.</p>'}</div>
    ${risks}
    <div class="verdict">${r.verdict}</div>
    <div class="scout-row">
      <div class="scout-card">
        <h6>${m.home_team}</h6>
        <div class="players">🎯 ${(r.home_players||[]).join(' · ')}</div>
        <div class="note">${r.home_notes}</div>
      </div>
      <div class="scout-card">
        <h6>${m.away_team}</h6>
        <div class="players">🎯 ${(r.away_players||[]).join(' · ')}</div>
        <div class="note">${r.away_notes}</div>
      </div>
    </div>
    <div class="stat-grid" style="margin-top:12px">${statRows}</div>`;
}

// ── Match card HTML ────────────────────────────────────────────────────────
// ── Match lookup table (keyed by match_no for modal access) ─────────────────
const MATCH_LOOKUP = {};
(DATA.group_matches||[]).forEach(m => { MATCH_LOOKUP[m.match_no] = m; });
(DATA.ko_matches||[]).forEach(m => { MATCH_LOOKUP[m.match_no] = m; });

// ── Compact match card (click to open modal) ────────────────────────────────
function matchCard(m, showMeet=false) {
  const hName = m.home_display || m.home_team;
  const aName = m.away_display || m.away_team;
  const favSide = m.p_home_win > m.p_away_win ? 'home' : (m.p_away_win > m.p_home_win ? 'away' : null);
  const favProb = Math.max(m.p_home_win||0, m.p_away_win||0);
  const hColor  = favSide==='home' ? '#60a5fa' : 'var(--text)';
  const aColor  = favSide==='away' ? '#fb923c' : 'var(--text)';
  const borderColor = m.played ? '#22c55e' : (favSide==='home' ? '#60a5fa' : (favSide==='away' ? '#fb923c' : 'var(--border)'));

  // VN time
  let vnTime = '';
  if (m.kickoff_utc) {
    const d = new Date(m.kickoff_utc);
    const vn = new Date(d.getTime() + 7*3600000);
    vnTime = `${String(vn.getUTCHours()).padStart(2,'0')}:${String(vn.getUTCMinutes()).padStart(2,'0')} VN`;
  }

  // Meta row
  const groupChip = m.group && m.group.length===1
    ? `<span class="group-pill" style="font-size:10px;padding:1px 7px">Grp ${m.group}${m.matchday?` · MD${m.matchday}`:''}</span>` : '';
  const timeChip = vnTime ? `<span style="font-size:11px;font-weight:700;color:var(--gold)">🕐 ${vnTime}</span>` : '';
  const meetChip = showMeet && m.meet_pct ? `<span style="font-size:10px;color:var(--muted)">Meet ${m.meet_pct}%</span>` : '';

  let accChip = '';
  if (m.played && m.direction_correct !== undefined) {
    // Show predicted winner (direction) not just the Poisson mode scoreline
    const predWLabel = !m.predicted_winner ? '?' :
      m.predicted_winner === m.home ? (m.home_display || m.home) :
      m.predicted_winner === m.away ? (m.away_display || m.away) : 'Draw';
    const predProb = m.predicted_winner === m.home   ? Math.round(m.pre_p_home||m.p_home_win||0) :
                     m.predicted_winner === m.away   ? Math.round(m.pre_p_away||m.p_away_win||0) :
                                                       Math.round(m.pre_p_draw||m.p_draw||0);
    const predHint = predProb ? `${predWLabel} ${predProb}%` : predWLabel;
    accChip = `<span class="${m.direction_correct?'correct-badge ok':'correct-badge fail'}" style="font-size:10px;padding:1px 7px;margin-left:auto">${m.direction_correct?'✅':'❌'} ${m.actual} <span style="opacity:.6;font-size:9px">(pred ${predHint})</span></span>`;
  }

  const metaRow = `<div class="mc-meta">
    ${groupChip}
    <span style="font-size:11px;color:var(--muted)">📍 ${m.city}</span>
    ${timeChip}${meetChip}
    ${m.played ? '<span style="font-size:11px;color:#22c55e;font-weight:600">✓ FT</span>' : ''}
    ${accChip}
  </div>`;

  // Center block
  // For upcoming: show PREDICTION (winner + %) clearly. xG shown under team names.
  // Avoid showing mode scoreline as a "prediction" — it's almost always 1-1.
  const predWinnerLabel = !m.predicted_winner ? 'Even'
    : m.predicted_winner === m.home ? (m.home_display||m.home)
    : m.predicted_winner === m.away ? (m.away_display||m.away)
    : 'Draw';
  const predColor = !m.predicted_winner ? 'var(--muted)'
    : m.predicted_winner === m.home ? '#60a5fa'
    : m.predicted_winner === m.away ? '#fb923c'
    : '#a3e635';
  const predPct = m.predicted_winner === m.home   ? Math.round(m.p_home_win)
                : m.predicted_winner === m.away   ? Math.round(m.p_away_win)
                : Math.round(m.p_draw);

  const centerBlock = m.played
    ? `<div class="mc-center">
         <div class="mc-ft-score">${m.actual}</div>
         <div class="mc-vs">Full Time</div>
       </div>`
    : `<div class="mc-center">
         <div class="mc-vs" style="font-size:10px;color:var(--muted);margin-bottom:2px">PREDICTION</div>
         <div class="mc-pred-winner" style="color:${predColor};font-size:15px;font-weight:800;line-height:1.1">${predWinnerLabel}</div>
         <div style="font-size:20px;font-weight:900;color:${predColor}">${predPct}%</div>
         <div class="mc-vs" style="font-size:10px;margin-top:2px">confidence</div>
       </div>`;

  // Prob bar
  const bar = !m.played
    ? `<div style="padding:0 14px 8px">${probBar(m.p_home_win, m.p_draw, m.p_away_win)}</div>` : '';

  // Bottom badges
  const badges = [];
  if (!m.played && m.factor_cards && m.factor_cards.length)
    badges.push(`<span class="mc-badge" style="background:rgba(99,102,241,.12);color:#818cf8">${m.factor_cards.length} factors</span>`);
  if (m.match_context)
    badges.push(`<span class="mc-badge" style="background:rgba(168,85,247,.12);color:#c084fc">📜 context</span>`);
  if (!m.played && m.home_star_player && m.home_star_player.name)
    badges.push(`<span class="mc-badge" style="background:rgba(234,179,8,.1);color:#ca8a04">⭐ stars</span>`);
  const badgeRow = `<div class="mc-badges">${badges.join('')}<span style="margin-left:auto;font-size:10px;color:var(--muted)">tap for details →</span></div>`;

  // Register in lookup
  if (m.match_no) MATCH_LOOKUP[m.match_no] = m;

  return `
  <div class="match-card" style="border-top:3px solid ${borderColor}${m.played?';opacity:.93':''}"
       onclick="openMatchModal(${JSON.stringify(m.match_no)})">
    ${metaRow}
    <div class="mc-body">
      <div class="mc-team">
        <div class="mc-team-name" style="color:${hColor}">${hName}</div>
        <div class="mc-team-xg">xG ${m.xg_home}</div>
      </div>
      ${centerBlock}
      <div class="mc-team away">
        <div class="mc-team-name" style="color:${aColor}">${aName}</div>
        <div class="mc-team-xg">xG ${m.xg_away}</div>
      </div>
    </div>
    ${bar}
    ${badgeRow}
  </div>`;
}


// ── Match Detail Modal ───────────────────────────────────────────────────────
function openMatchModal(matchNo) {
  const m = MATCH_LOOKUP[matchNo];
  if (!m) return;
  const hName = m.home_display || m.home_team;
  const aName = m.away_display || m.away_team;
  const favSide = m.p_home_win > m.p_away_win ? 'home' : (m.p_away_win > m.p_home_win ? 'away' : null);

  // Header
  document.getElementById('modal-title').textContent = `${hName} vs ${aName}`;

  // ── Build modal body ────────────────────────────────────────────────────
  let html = '';

  // 1. Prediction summary
  const vnTimeStr = m.kickoff_utc ? (() => {
    const d = new Date(m.kickoff_utc);
    const vn = new Date(d.getTime()+7*3600000);
    const days=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    const months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return `${days[vn.getUTCDay()]} ${months[vn.getUTCMonth()]} ${vn.getUTCDate()} · ${String(vn.getUTCHours()).padStart(2,'0')}:${String(vn.getUTCMinutes()).padStart(2,'0')} VN`;
  })() : '';

  // For upcoming matches: show PREDICTION clearly, not a confusing "expected score"
  // The mode scoreline (e.g. 1-1) is the single most probable scoreline from Poisson,
  // which is almost always 1-1 even when one team is a heavy favourite — don't show it
  // as "the prediction". The REAL prediction is the outcome probability distribution.
  const modalPredWinner = !m.predicted_winner ? null
    : m.predicted_winner === m.home ? hName
    : m.predicted_winner === m.away ? aName : 'Draw';
  const modalPredPct = !m.predicted_winner ? null
    : m.predicted_winner === m.home   ? (m.p_home_win||0).toFixed(1)
    : m.predicted_winner === m.away   ? (m.p_away_win||0).toFixed(1)
    : (m.p_draw||0).toFixed(1);
  const modalPredColor = !m.predicted_winner ? 'var(--muted)'
    : m.predicted_winner === m.home ? '#60a5fa'
    : m.predicted_winner === m.away ? '#fb923c' : '#a3e635';

  const centerScore = m.played
    ? `<div class="modal-score">${m.actual}</div><div class="modal-score-hint" style="color:var(--green)">Full Time</div>`
    : modalPredWinner
      ? `<div style="font-size:11px;color:var(--muted);margin-bottom:2px">PREDICTED WINNER</div>
         <div class="modal-score" style="color:${modalPredColor};font-size:22px">${modalPredWinner}</div>
         <div class="modal-score-hint" style="color:${modalPredColor}">${modalPredPct}% confidence</div>`
      : `<div class="modal-score">vs</div>`;

  html += `<div class="modal-section">
    <div class="modal-teams-row">
      <div class="modal-team" style="text-align:left">
        <div class="modal-team-name" style="color:${favSide==='home'?'#60a5fa':'var(--text)'}">${hName}</div>
        <div class="modal-team-xg">xG ${m.xg_home} <span style="opacity:.5;font-size:9px">(expected goals)</span></div>
      </div>
      <div class="modal-vs-block">${centerScore}</div>
      <div class="modal-team" style="text-align:right">
        <div class="modal-team-name" style="color:${favSide==='away'?'#fb923c':'var(--text)'}">${aName}</div>
        <div class="modal-team-xg">xG ${m.xg_away} <span style="opacity:.5;font-size:9px">(expected goals)</span></div>
      </div>
    </div>
    <div class="modal-prob-row">
      <div class="modal-prob-chip ${favSide==='home'?'win-h':''}">${hName} Win <strong>${(m.p_home_win||0).toFixed(1)}%</strong></div>
      <div class="modal-prob-chip draw">Draw <strong>${(m.p_draw||0).toFixed(1)}%</strong></div>
      <div class="modal-prob-chip ${favSide==='away'?'win-a':''}">${aName} Win <strong>${(m.p_away_win||0).toFixed(1)}%</strong></div>
    </div>
    ${!m.played ? probBar(m.p_home_win, m.p_draw, m.p_away_win) : ''}
    <div class="modal-meta-chips">
      ${m.group && m.group.length===1 ? `<span class="modal-chip">Group ${m.group} · MD${m.matchday||'?'}</span>` : ''}
      <span class="modal-chip">📍 ${m.city}</span>
      <span class="modal-chip">${m.stadium}</span>
      ${vnTimeStr ? `<span class="modal-chip" style="color:var(--gold);border-color:var(--gold)">🕐 ${vnTimeStr}</span>` : ''}
    </div>
  </div>`;

  // 2. Accuracy (played)
  if (m.played && m.direction_correct !== undefined) {
    const dirOk = m.direction_correct;
    const predW = m.predicted_winner === m.home_team ? hName : (m.predicted_winner === m.away_team ? aName : 'Draw');
    const actW  = m.actual_winner  === m.home_team ? hName : (m.actual_winner  === m.away_team ? aName : 'Draw');
    html += `<div class="modal-section">
      <div class="modal-section-title">🎯 Model Verdict</div>
      <div class="acc-verdict">
        <span class="${dirOk?'correct-badge ok':'correct-badge fail'}">${dirOk?'✅ Correct direction':'❌ Wrong direction'}</span>
        <span class="av-pre">Predicted: <span class="av-score">${predW}</span>
          <span style="opacity:.6;font-size:10px">(H ${Math.round(m.pre_p_home||m.p_home_win||0)}% · D ${Math.round(m.pre_p_draw||m.p_draw||0)}% · A ${Math.round(m.pre_p_away||m.p_away_win||0)}%)</span></span>
        <span style="color:var(--muted)">·</span>
        <span class="av-pre">Actual: <span class="av-score" style="color:var(--green)">${actW} (${m.actual_score})</span></span>
        ${m.score_correct ? '<span class="correct-badge ok">🎯 Exact Score!</span>' : ''}
      </div>
    </div>`;
  }

  // 3. Factor adjustments
  const fs = m.factor_summary || {};
  html += `<div class="modal-section">
    <div class="modal-section-title">🔬 Contextual Factors <span style="font-weight:400;color:var(--muted)">(xG adjustments on top of ELO + ML)</span></div>
    <div class="factor-coach-row">
      <div class="factor-coach-card">
        <div class="fc-label">Home — ${hName}</div>
        <div class="fc-coach">${fs.home_coach||'—'}</div>
        <div class="fc-form">${fs.home_formation||''}  ·  Style: ${fs.home_style_code||'?'}</div>
        <div class="fc-morale" style="color:${(fs.home_morale||7)>=8?'var(--green)':(fs.home_morale||7)>=6?'var(--yellow)':'var(--red)'}">
          Morale ${(fs.home_morale||7).toFixed(1)}/10
          ${fs.home_injury?'<span style="color:var(--red)"> · 🏥 Injury</span>':''}
        </div>
        ${m.factor_adj_home!==undefined?`<div style="font-size:11px;color:${m.factor_adj_home>=0?'var(--green)':'var(--red)'}">xG adj: ${m.factor_adj_home>=0?'+':''}${(m.factor_adj_home||0).toFixed(2)}</div>`:''}
      </div>
      <div class="factor-coach-card">
        <div class="fc-label">Away — ${aName}</div>
        <div class="fc-coach">${fs.away_coach||'—'}</div>
        <div class="fc-form">${fs.away_formation||''}  ·  Style: ${fs.away_style_code||'?'}</div>
        <div class="fc-morale" style="color:${(fs.away_morale||7)>=8?'var(--green)':(fs.away_morale||7)>=6?'var(--yellow)':'var(--red)'}">
          Morale ${(fs.away_morale||7).toFixed(1)}/10
          ${fs.away_injury?'<span style="color:var(--red)"> · 🏥 Injury</span>':''}
        </div>
        ${m.factor_adj_away!==undefined?`<div style="font-size:11px;color:${m.factor_adj_away>=0?'var(--green)':'var(--red)'}">xG adj: ${m.factor_adj_away>=0?'+':''}${(m.factor_adj_away||0).toFixed(2)}</div>`:''}
      </div>
    </div>`;

  if (m.factor_cards && m.factor_cards.length) {
    const typeChip = t => `<span class="factor-chip ${t}">${
      t==='player'?'🏥 Player':t==='coach'?'🧠 Coach':t==='counter'?'⚔️ Counter':t==='host'?'🏟️ Host':t==='confederation'?'📊 Conf':t==='lowblock'?'🚌 LowBlock':'🏛️ Political'
    }</span>`;
    html += m.factor_cards.map(fc => `
      <div class="factor-row ${fc.side}-side" style="margin-top:5px">
        <span>${fc.icon}</span>
        ${typeChip(fc.type)}
        <span class="factor-row-text">${fc.text}</span>
        <span style="color:var(--muted);font-size:10px;white-space:nowrap">Δ${(fc.magnitude*100).toFixed(0)}%xG</span>
      </div>`).join('');
  } else {
    html += `<div style="color:var(--muted);font-size:12px;margin-top:8px">No significant factor adjustments — teams evenly matched on contextual metrics.</div>`;
  }
  html += `</div>`;

  // 4. Star Players
  function starCardModal(sp, yc, injured, suspended, side, teamName) {
    if (!sp || !sp.name) return `<div class="star-card ${side}-star"><div class="star-no-data">${teamName} — no star player data</div></div>`;
    const fatCol  = sp.fatigue_level>=7?'#ef4444':sp.fatigue_level>=5?'#eab308':'#22c55e';
    const riskCol = sp.injury_risk>=6?'#ef4444':sp.injury_risk>=3?'#eab308':'#22c55e';
    const allYC = Object.entries(yc||{});
    const discRows = allYC.length
      ? allYC.map(([p,cnt])=>`<span class="${cnt>=2?'yellow-chip warn-chip':'yellow-chip'}">🟨×${cnt} ${p}${cnt>=2?' ⚠️ BAN RISK':''}</span>`).join('')
      : `<span class="disc-clean">Clean — no bookings</span>`;
    const suspRows = (suspended||[]).map(p=>`<span class="red-chip">🟥 SUSPENDED: ${p}</span>`).join('');
    const injRows  = (injured||[]).map(p=>`<span class="red-chip">🏥 OUT: ${p}</span>`).join('');
    return `<div class="star-card ${side}-star">
      <div class="star-name">⭐ ${sp.name}</div>
      <div class="star-club">🏟️ ${sp.club}</div>
      <div class="star-achievement">🏆 ${sp.club_achievement}</div>
      <div class="star-stats">📊 ${sp.season_stats}</div>
      <div class="star-why">${sp.why_carrying}</div>
      <div class="star-bars">
        <div class="star-bar-row"><div class="star-bar-label">Fatigue</div>
          <div class="star-bar-track"><div class="star-bar-fill" style="width:${(sp.fatigue_level||0)*10}%;background:${fatCol}"></div></div>
          <div style="font-size:10px;color:${fatCol}">${sp.fatigue_level}/10</div></div>
        <div class="star-bar-row"><div class="star-bar-label">Injury Risk</div>
          <div class="star-bar-track"><div class="star-bar-fill" style="width:${(sp.injury_risk||0)*10}%;background:${riskCol}"></div></div>
          <div style="font-size:10px;color:${riskCol}">${sp.injury_risk}/10</div></div>
      </div>
      <div class="disc-panel"><div class="disc-title">🟨 Discipline</div>
        <div class="disc-row">${suspRows}${injRows}${discRows}</div>
      </div>
    </div>`;
  }

  html += `<div class="modal-section">
    <div class="modal-section-title">⭐ Star Players</div>
    <div class="star-panel-grid">
      ${starCardModal(m.home_star_player, m.home_yellow_cards, m.home_injured, m.home_suspended, 'home', hName)}
      ${starCardModal(m.away_star_player, m.away_yellow_cards, m.away_injured, m.away_suspended, 'away', aName)}
    </div>
  </div>`;

  // 5. Historical / Political context
  if (m.match_context) {
    const ctx = m.match_context;
    const intBadge = `<span class="ctx-badge-${ctx.intensity}">${ctx.intensity==='high'?'🔴 HIGH':ctx.intensity==='medium'?'🟡 MEDIUM':'🟢 LOW'} INTENSITY</span>`;
    const histBullets = (ctx.history||[]).map(b=>`<div class="ctx-bullet">${b}</div>`).join('');
    const geoBullets  = (ctx.geopolitical||[]).map(b=>`<div class="ctx-bullet">${b}</div>`).join('');
    const pitchBullets= (ctx.on_pitch||[]).map(b=>`<li style="margin-bottom:3px">${b}</li>`).join('');
    html += `<div class="modal-section">
      <div class="modal-section-title">🏛️ Historical & Political Context ${intBadge}</div>
      <div class="ctx-panel ctx-intensity-${ctx.intensity}" style="margin-top:0">
        <div class="ctx-headline">${ctx.headline}</div>
        <div class="ctx-section"><div class="ctx-section-title">📜 Historical Relationship</div>${histBullets}</div>
        <div class="ctx-section"><div class="ctx-section-title">🌐 Current Geopolitics (2025–26)</div>${geoBullets}</div>
        <div class="ctx-section"><div class="ctx-section-title">⚽ Prior WC Meetings</div>
          <div class="ctx-pitch"><ul style="padding-left:14px;margin:0">${pitchBullets}</ul></div></div>
        <div class="ctx-impact">💡 <strong>Match Impact:</strong> ${ctx.impact_note}</div>
      </div>
    </div>`;
  }

  // 6. News
  if (!m.played && m.news && m.news.length) {
    const sent = m.news_sentiment||{};
    const homeBias = sent.home_bias||'neutral';
    const awayBias = sent.away_bias||'neutral';
    const modelFavHome = m.p_home_win > m.p_away_win;
    const biasNote = (homeBias!=='neutral'||awayBias!=='neutral')
      ? `<span class="bias-vs-model">${modelFavHome===(homeBias==='bullish')?'✓ Media agrees':'⚠️ Media disagrees'} with model</span>` : '';
    const articles = m.news.map(a => {
      const hSent = a.home_sentiment!=='neutral' ? `<span class="sent-chip ${a.home_sentiment==='positive'?'bullish':'bearish'}">${hName} ${a.home_sentiment}</span>` : '';
      const aSent = a.away_sentiment!=='neutral' ? `<span class="sent-chip ${a.away_sentiment==='positive'?'bullish':'bearish'}">${aName} ${a.away_sentiment}</span>` : '';
      return `<div class="news-item">
        <div class="ni-title">${a.title}</div>
        <div class="ni-meta"><span>${a.source}</span><span>${a.pub_date}</span><div class="ni-sents">${hSent}${aSent}</div></div>
      </div>`;
    }).join('');
    html += `<div class="modal-section">
      <div class="modal-section-title">📰 Latest News & Sentiment</div>
      <div class="news-panel" style="margin:0">
        <div class="news-header">
          <div class="sentiment-bar">
            <span class="sent-chip ${homeBias}">${hName}: ${homeBias}</span>
            <span class="sent-chip ${awayBias}">${aName}: ${awayBias}</span>
            ${biasNote}
          </div>
        </div>
        <div class="news-articles">${articles}</div>
      </div>
    </div>`;
  }

  // 7. Full reasoning
  html += `<div class="modal-section">
    <div class="modal-section-title">💡 Model Reasoning & Scouting</div>
    <div class="reasoning-body open" style="border:none;padding:0">${reasoningHTML(m.reasoning, m)}</div>
  </div>`;

  document.getElementById('modal-body').innerHTML = html;
  const overlay = document.getElementById('match-modal-overlay');
  overlay.classList.add('open');
  // Animate in
  requestAnimationFrame(() => {
    document.getElementById('match-modal').style.transform = '';
  });
  document.body.style.overflow = 'hidden';
}

function closeMatchModal() {
  document.getElementById('match-modal-overlay').classList.remove('open');
  document.body.style.overflow = '';
}

function closeModal(e) {
  if (e.target === document.getElementById('match-modal-overlay')) closeMatchModal();
}

// Close on Escape key
document.addEventListener('keydown', e => { if (e.key==='Escape') closeMatchModal(); });

// ── Overview ───────────────────────────────────────────────────────────────
function renderOverview() {
  const wp     = DATA.win_probs;
  const champ  = DATA.champion;
  const final2 = DATA.projected_final && DATA.projected_final.home
    ? { home_team: DATA.projected_final.home, away_team: DATA.projected_final.away }
    : DATA.ko_matches.find(m => m.round === 'Final');
  const acc    = DATA.accuracy;

  const accCard = acc && acc.total_played ? `
    <div class="ov-card">
      <div class="label">Model Accuracy</div>
      <div class="value" style="color:${acc.direction_pct>=70?'var(--green)':acc.direction_pct>=50?'var(--yellow)':'var(--red)'}">${acc.direction_pct}%</div>
      <div class="sub">${acc.direction_correct}/${acc.total_played} correct · <a href="#" onclick="showTabAcc();return false;" style="color:#60a5fa">See tracker →</a></div>
    </div>` : '';

  document.getElementById('overview-top').innerHTML = `
    <div class="ov-card">
      <div class="label">Predicted Champion</div>
      <div class="value" style="color:var(--gold)">${champ}</div>
      <div class="sub">Actual bracket + model picks for unplayed rounds</div>
    </div>
    <div class="ov-card">
      <div class="label">Predicted Final</div>
      <div class="value" style="font-size:18px">${final2 ? (final2.home_team + ' vs ' + final2.away_team) : '—'}</div>
      <div class="sub">Real results so far — model favourites onward</div>
    </div>
    <div class="ov-card">
      <div class="label">Matches Predicted</div>
      <div class="value" style="color:var(--green)">104</div>
      <div class="sub">72 group + 31 knockout + 1 third-place playoff</div>
    </div>
    <div class="ov-card">
      <div class="label">Method</div>
      <div class="value" style="color:var(--purple);font-size:16px">Enhanced v2</div>
      <div class="sub">ELO + ML + Player + Coach + Counter + Political</div>
    </div>
    ${accCard}`;

  const rows = wp.map((t, i) => {
    const posLabel  = ['🥇 1st','🥈 2nd','🥉 3rd','4th'][t.exp_pos - 1] || '—';
    const advBadge  = t.advances
      ? `<span style="color:var(--green);font-size:11px">✓ Advances</span>`
      : `<span style="color:var(--muted);font-size:11px">Eliminated</span>`;
    const roundLabel = t.win_pct === 100 ? '🏆 Champion' :
                       t.final_pct === 100 ? '⭐ Final' :
                       t.sf_pct === 100 ? 'Semi-Final' :
                       t.qf_pct === 100 ? 'Quarter-Final' :
                       t.advances ? 'Round of 32' : 'Group stage exit';
    const eloChg = DATA.accuracy && DATA.accuracy.elo_changes && DATA.accuracy.elo_changes[t.team];
    const eloTag = eloChg ? `<span style="color:${eloChg>0?'var(--green)':'var(--red)'};font-size:10px">${eloChg>0?'+':''}${eloChg}</span>` : '';
    return `<tr>
      <td><span class="rank-num">${i+1}.</span> ${t.team}</td>
      <td>Group ${t.group} <span class="style-badge style-${t.style.replace(' ','-')}">${t.style}</span></td>
      <td>${posLabel} ${advBadge}</td>
      <td style="color:var(--gold);font-weight:700">${roundLabel}</td>
      <td>${t.elo} ${eloTag}</td>
      <td>${t.star_rating.toFixed(1)}</td>
    </tr>`;
  }).join('');
  document.getElementById('win-table').innerHTML = `
    <thead><tr>
      <th style="text-align:left">Team</th>
      <th style="text-align:left">Group</th>
      <th>Expected Group Finish</th>
      <th>Predicted KO Path</th>
      <th>ELO (Δ)</th>
      <th>⭐ Star</th>
    </tr></thead><tbody>${rows}</tbody>`;
}

// ── Schedule by Date ──────────────────────────────────────────────────────
const TODAY = '2026-06-14';
const MD1_DATES = ['2026-06-11','2026-06-12','2026-06-13','2026-06-14','2026-06-15','2026-06-16','2026-06-17'];
const MD2_DATES = ['2026-06-18','2026-06-19','2026-06-20','2026-06-21','2026-06-22','2026-06-23','2026-06-24'];
const MD3_DATES = ['2026-06-25','2026-06-26','2026-06-27'];
let currentDateFilter = 'ALL';

function filterDate(filter, btn) {
  currentDateFilter = filter;
  document.querySelectorAll('#date-filter-bar .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderDateMatches();
}

function fmtDate(d) {
  const dt = new Date(d + 'T12:00:00Z');
  return dt.toLocaleDateString('en-US', {weekday:'long', month:'long', day:'numeric', year:'numeric'});
}

function renderDateMatches() {
  const matchByNo = {};
  DATA.group_matches.forEach(m => matchByNo[m.match_no] = m);
  (DATA.actual_ko_schedule||[]).forEach(m => matchByNo[m.match_no] = m);

  let allowedDates;
  if (currentDateFilter === 'MD1') allowedDates = new Set(MD1_DATES);
  else if (currentDateFilter === 'MD2') allowedDates = new Set(MD2_DATES);
  else if (currentDateFilter === 'MD3') allowedDates = new Set(MD3_DATES);
  else allowedDates = null;

  const con = document.getElementById('date-matches-container');
  const sections = [];

  for (const [date, matchNos] of Object.entries(DATA.dates_index)) {
    if (allowedDates && !allowedDates.has(date)) continue;
    const matches = matchNos.map(no => matchByNo[no]).filter(Boolean)
      .sort((a,b) => {
        const ta = a.kickoff_utc ? new Date(a.kickoff_utc).getTime() : 0;
        const tb = b.kickoff_utc ? new Date(b.kickoff_utc).getTime() : 0;
        return ta - tb || a.match_no - b.match_no;
      });
    if (!matches.length) continue;

    const isPast    = date < TODAY;
    const isToday   = date === TODAY;
    const groups    = [...new Set(matches.map(m => m.group))].sort();
    const statusBadge = isPast
      ? `<span class="played-badge">✓ Results in</span>`
      : (isToday ? `<span class="today-badge">📍 TODAY</span>` : '');

    const cards = matches.map(m => {
      // TBD knockout pairings get a placeholder; confirmed ones use the full card
      if (m.round && (m.home_tbd || m.away_tbd || m.p_home_win == null)) {
        const rnd = {R32:'Round of 32',R16:'Round of 16',QF:'Quarter-Final',SF:'Semi-Final','3rd':'3rd Place','Final':'Final'}[m.round] || m.round;
        const hasPens = m.home_pens != null && m.away_pens != null;
        const penWinner = hasPens ? (m.home_pens > m.away_pens ? m.home : m.away) : null;
        const score = m.played
          ? `<div style="text-align:center"><span style="font-weight:800;font-size:17px">${m.home_score} – ${m.away_score}</span>${hasPens ? `<div style="font-size:10px;color:#f59e0b">pens ${m.home_pens}–${m.away_pens}</div><div style="font-size:10px;color:#22c55e">${penWinner} advances</div>` : ''}</div>`
          : `<span style="color:var(--muted);font-size:13px">vs</span>`;
        const predRow = (m.p_home_win != null && !m.played) ? `
          <div style="display:flex;gap:14px;margin-top:8px;font-size:12px;color:var(--muted)">
            <span>🏠 ${m.p_home_win.toFixed(1)}%</span><span>🤝 ${m.p_draw.toFixed(1)}%</span><span>✈️ ${m.p_away_win.toFixed(1)}%</span>
            ${m.likely_score ? `<span style="margin-left:auto">📊 ${m.likely_score}</span>` : ''}
          </div>` : '';
        return `<div class="card" style="padding:14px 16px">
          <div style="font-size:11px;color:var(--muted);margin-bottom:6px">M${m.match_no} · ${rnd} · ${m.city||''} ${m.played ? '<span style="color:#22c55e">✓ FT</span>' : ''}</div>
          <div style="display:flex;align-items:center;gap:10px">
            <span style="flex:1;font-weight:700">${teamFlag(m.home)} ${m.home}</span>
            ${score}
            <span style="flex:1;text-align:right;font-weight:700">${m.away} ${teamFlag(m.away)}</span>
          </div>
          ${predRow}
        </div>`;
      }
      return matchCard(m);
    }).join('');

    sections.push(`
      <div class="date-section">
        <div class="date-header">
          <span class="date-label">${fmtDate(date)}</span>
          ${statusBadge}
          <span class="date-sub">${matches.length} match${matches.length>1?'es':''}</span>
          ${groups.filter(g => g).map(g => `<span class="group-pill">${g.length===1 ? 'Group '+g : ({R32:'Round of 32',R16:'Round of 16',QF:'Quarter-Finals',SF:'Semi-Finals','3rd':'3rd Place','Final':'Final'}[g]||g)}</span>`).join('')}
        </div>
        <div class="matchday-row">${cards}</div>
      </div>`);
  }
  con.innerHTML = sections.join('');
}

// ── By Group ───────────────────────────────────────────────────────────────
let currentGroup = 'ALL';
function filterGroup(grp, btn) {
  currentGroup = grp;
  document.querySelectorAll('#group-filter-bar .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderGroupMatches();
}

function renderGroupMatches() {
  const grps = [...new Set(DATA.group_matches.map(m => m.group))].sort();
  const filterBar = document.getElementById('group-filter-bar');
  if (filterBar.children.length === 1) {
    grps.forEach(g => {
      const b = document.createElement('button');
      b.className = 'filter-btn';
      b.textContent = 'Group ' + g;
      b.onclick = (e) => filterGroup(g, e.target);
      filterBar.appendChild(b);
    });
  }
  const filtered = currentGroup === 'ALL' ? grps : [currentGroup];
  document.getElementById('group-matches-container').innerHTML = filtered.map(grp => {
    const ms = DATA.group_matches.filter(m => m.group === grp)
      .sort((a,b) => {
        const ta = a.kickoff_utc ? new Date(a.kickoff_utc).getTime() : 0;
        const tb = b.kickoff_utc ? new Date(b.kickoff_utc).getTime() : 0;
        return ta - tb || a.match_no - b.match_no;
      });
    return `<div class="card">
      <div class="card-header">
        <span class="card-title">Group ${grp}</span>
        <span class="date-sub">${ms.length} matches · MD1, MD2, MD3</span>
      </div>
      ${ms.map(m => matchCard(m)).join('')}
    </div>`;
  }).join('');
}

// ── Standings ──────────────────────────────────────────────────────────────
function renderStandings() {
  const grps = Object.keys(DATA.standings).sort();
  document.getElementById('standings-grid').innerHTML = grps.map(grp => {
    const rows = DATA.standings[grp].map((r, i) => {
      const medals = ['🥇','🥈','🥉',''];
      const medal  = medals[i] || '';
      const status = r.advances
        ? `<span style="color:var(--green);font-weight:600">✓ Advances to R32</span>`
        : `<span style="color:var(--muted)">Eliminated</span>`;
      const gapTxt = i === 0
        ? `<span style="color:var(--muted);font-size:10px">+${r.pts_gap} ahead of 2nd</span>`
        : `<span style="color:var(--muted);font-size:10px">${r.pts_gap} behind ${i===1?'1st':'above'}</span>`;
      return `<tr>
        <td>${medal} ${r.team}</td>
        <td style="color:var(--gold);font-weight:700">${r.exp_pts}</td>
        <td>${r.exp_gf}</td>
        <td>${r.exp_ga}</td>
        <td style="color:${r.exp_gd>=0?'var(--green)':'var(--red)'}">${r.exp_gd>=0?'+':''}${r.exp_gd}</td>
        <td>${gapTxt}</td>
        <td>${status}</td>
      </tr>`;
    }).join('');
    return `
      <div class="card">
        <div class="card-header">
          <span class="card-title">Group ${grp}</span>
          <span style="color:var(--muted);font-size:11px">xPts = probability-weighted expected points per match</span>
        </div>
        <table class="stand-table">
          <thead><tr>
            <th>Team</th><th>xPts</th><th>xGF</th><th>xGA</th><th>xGD</th><th>Gap</th><th>Status</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }).join('');
}

// ── Knockout ───────────────────────────────────────────────────────────────
const KO_ROUNDS = [
  {key:'R32',   label:'Round of 32'},
  {key:'R16',   label:'Round of 16'},
  {key:'QF',    label:'Quarter-Finals'},
  {key:'SF',    label:'Semi-Finals'},
  {key:'3rd',   label:'🥉 3rd Place'},
  {key:'Final', label:'⭐ The Final'},
];
let currentSchedRound = 'R32';

function renderKO() {
  const schedTabsEl = document.getElementById('ko-sched-tabs');
  const schedRounds = [...new Set((DATA.actual_ko_schedule||[]).map(m=>m.round))];
  const orderedRounds = ['R32','R16','QF','SF','3rd','Final'].filter(r=>schedRounds.includes(r));
  const roundLabels = {R32:'Round of 32',R16:'Round of 16',QF:'Quarter-Finals',SF:'Semi-Finals','3rd':'🥉 3rd Place','Final':'⭐ The Final'};
  // Default to first unplayed round, or last round if all played
  const firstUnplayed = orderedRounds.find(r => (DATA.actual_ko_schedule||[]).filter(m=>m.round===r).some(m=>!m.played));
  currentSchedRound = firstUnplayed || orderedRounds[orderedRounds.length-1] || 'R32';
  schedTabsEl.innerHTML = orderedRounds.map(r =>
    `<button class="ko-tab ${r===currentSchedRound?'active':''}" onclick="switchSchedKO('${r}',this)">${roundLabels[r]||r}</button>`
  ).join('');
  renderSchedKOMatches(currentSchedRound);
}

function switchSchedKO(key, btn) {
  currentSchedRound = key;
  document.getElementById('ko-sched-tabs').querySelectorAll('.ko-tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  renderSchedKOMatches(key);
}

function renderSchedKOMatches(round) {
  const ms = (DATA.actual_ko_schedule||[]).filter(m=>m.round===round).sort((a,b)=>a.match_no-b.match_no);
  document.getElementById('ko-sched-container').innerHTML = ms.map(m => {
    // Confirmed pairings use the full prediction card (tap for details)
    if (!m.home_tbd && !m.away_tbd && m.p_home_win != null) return matchCard(m);
    const homeTbd = m.home_tbd, awayTbd = m.away_tbd;
    const hFlag = homeTbd ? '🏳️' : (FLAG_MAP[m.home]||'🏳️');
    const aFlag = awayTbd ? '🏳️' : (FLAG_MAP[m.away]||'🏳️');
    const hName = homeTbd ? `<span style="color:var(--muted);font-style:italic">${m.home}</span>` : `<strong>${m.home}</strong>`;
    const aName = awayTbd ? `<span style="color:var(--muted);font-style:italic">${m.away}</span>` : `<strong>${m.away}</strong>`;
    const hasPens = m.home_pens != null && m.away_pens != null;
    const penWinner = hasPens ? (m.home_pens > m.away_pens ? m.home : m.away) : null;
    const scoreStr = m.played
      ? `<div style="text-align:center"><span class="score-live">${m.home_score} – ${m.away_score}</span>${hasPens ? `<div style="font-size:10px;color:#f59e0b;margin-top:2px">pens ${m.home_pens}–${m.away_pens}</div><div style="font-size:10px;color:#22c55e">${penWinner} advances</div>` : ''}</div>`
      : `<span style="color:var(--muted);font-size:12px">vs</span>`;
    const predRow = (!homeTbd && !awayTbd && m.p_home_win != null) ? `
      <div style="display:flex;gap:16px;margin-top:8px;font-size:12px;color:var(--muted)">
        <span>🏠 ${m.p_home_win.toFixed(1)}%</span>
        <span>🤝 ${m.p_draw.toFixed(1)}%</span>
        <span>✈️ ${m.p_away_win.toFixed(1)}%</span>
        ${m.likely_score ? `<span style="margin-left:auto">📊 ${m.likely_score}</span>` : ''}
        ${m.xg_home != null ? `<span>xG ${m.xg_home.toFixed(2)}–${m.xg_away.toFixed(2)}</span>` : ''}
      </div>` : '';
    const favBadge = (!homeTbd && !awayTbd && m.favourite && m.favourite_p > 0) ? `
      <span style="font-size:10px;background:#7c3aed22;color:#a78bfa;padding:2px 6px;border-radius:4px;margin-left:6px">
        ⭐ ${m.favourite} ${m.favourite_p.toFixed(0)}%
      </span>` : '';
    return `<div class="card" style="margin-bottom:12px;padding:14px 16px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span style="font-size:11px;color:var(--muted)">M${m.match_no} · ${m.date} · ${m.city}</span>
        <span style="font-size:10px;color:var(--muted)">${m.stadium}</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:6px">
        <span style="font-size:22px">${hFlag}</span>
        <span style="flex:1">${hName}</span>
        ${scoreStr}
        <span style="flex:1;text-align:right">${aName}</span>
        <span style="font-size:22px">${aFlag}</span>
      </div>
      ${favBadge ? `<div style="margin-top:6px">${favBadge}</div>` : ''}
      ${predRow}
    </div>`;
  }).join('') || `<div style="color:var(--muted);padding:20px">No matches scheduled yet for this round.</div>`;
}


// ── Teams ──────────────────────────────────────────────────────────────────
const FLAG_MAP = {
  'Argentina':'🇦🇷','Australia':'🇦🇺','Austria':'🇦🇹','Belgium':'🇧🇪','Bolivia':'🇧🇴',
  'Bosnia and Herzegovina':'🇧🇦','Brazil':'🇧🇷','Canada':'🇨🇦','Cabo Verde':'🇨🇻',
  'Chile':'🇨🇱','Colombia':'🇨🇴','Congo DR':'🇨🇩','Croatia':'🇭🇷','Czech Republic':'🇨🇿',
  'Ecuador':'🇪🇨','Egypt':'🇪🇬','England':'󠁧󠁢󠁥󠁮󠁧󠁿🏴󠁧󠁢󠁥󠁮󠁧󠁿','France':'🇫🇷','Georgia':'🇬🇪',
  'Germany':'🇩🇪','Ghana':'🇬🇭','Haiti':'🇭🇹','Honduras':'🇭🇳','Hungary':'🇭🇺',
  'Indonesia':'🇮🇩','Iran':'🇮🇷','Iraq':'🇮🇶','Ireland Republic':'🇮🇪','Ivory Coast':'🇨🇮',
  'Jamaica':'🇯🇲','Japan':'🇯🇵','Jordan':'🇯🇴','Mexico':'🇲🇽','Morocco':'🇲🇦',
  'Netherlands':'🇳🇱','New Zealand':'🇳🇿','Nigeria':'🇳🇬','Norway':'🇳🇴','Panama':'🇵🇦',
  'Paraguay':'🇵🇾','Peru':'🇵🇪','Portugal':'🇵🇹','Qatar':'🇶🇦','Romania':'🇷🇴',
  'Saudi Arabia':'🇸🇦','Scotland':'🏴󠁧󠁢󠁳󠁣󠁴󠁿','Senegal':'🇸🇳','Slovakia':'🇸🇰',
  'South Africa':'🇿🇦','South Korea':'🇰🇷','Spain':'🇪🇸','Sweden':'🇸🇪','Switzerland':'🇨🇭',
  'Tunisia':'🇹🇳','Turkey':'🇹🇷','United States':'🇺🇸','USA':'🇺🇸','Uruguay':'🇺🇾','Uzbekistan':'🇺🇿',
  'Venezuela':'🇻🇪','Wales':'🏴󠁧󠁢󠁷󠁬󠁳󠁿','Algeria':'🇩🇿','Curacao':'🇨🇼',
};
function teamFlag(name) { return FLAG_MAP[name] || '🏳️'; }

let _tpSort = 'win_pct';
let _tpSelected = null;

function renderTeams() {
  sortTeams(_tpSort, true);
  // Auto-select first team
  const first = DATA.win_probs[0];
  if (first) selectTeam(first.team);
}

function sortTeams(key, skipRender) {
  _tpSort = key;
  document.querySelectorAll('.tp-sort-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('tsb-' + {win_pct:'win',elo:'elo',group:'grp',name:'name'}[key]);
  if (btn) btn.classList.add('active');
  const q = (document.getElementById('tp-search')?.value || '').toLowerCase();
  renderTeamList(q);
}

function filterTeams() {
  const q = document.getElementById('tp-search').value.toLowerCase();
  renderTeamList(q);
}

function renderTeamList(q) {
  let teams = [...DATA.win_probs];
  if (q) teams = teams.filter(t => t.team.toLowerCase().includes(q) ||
    (t.key_players||[]).join(' ').toLowerCase().includes(q));

  teams.sort((a,b) => {
    if (_tpSort === 'win_pct') return b.win_pct - a.win_pct;
    if (_tpSort === 'elo')     return b.elo - a.elo;
    if (_tpSort === 'group')   return a.group.localeCompare(b.group) || a.team.localeCompare(b.team);
    return a.team.localeCompare(b.team);
  });

  document.getElementById('tp-list').innerHTML = teams.map(t => `
    <div class="tp-item ${_tpSelected === t.team ? 'selected' : ''}"
         onclick="selectTeam('${t.team.replace(/'/g,"\\'")}')">
      <div class="tp-flag">${teamFlag(t.team)}</div>
      <div class="tp-item-info">
        <div class="tp-item-name">${t.team}</div>
        <div class="tp-item-sub">Group ${t.group} · ${styleBadge(t.style)}</div>
      </div>
      <div class="tp-item-elo">${pct(t.win_pct)}</div>
    </div>`).join('');
}

function selectTeam(name) {
  _tpSelected = name;
  const t = DATA.win_probs.find(x => x.team === name);
  if (!t) return;

  // Re-render list to update selection highlight
  const q = (document.getElementById('tp-search')?.value || '').toLowerCase();
  renderTeamList(q);

  // Stat rows — only real data-derived stats
  const statRows = [
    { label:'⚡ ELO Rating',     value: t.elo,                       color:'#fbbf24', max:2200, min:1300 },
    { label:'🏆 Tournament Win%',value: (t.win_pct*100).toFixed(1)+'%', color:'#34d399', raw:true },
    { label:'🎯 Final%',         value: (t.final_pct*100).toFixed(1)+'%', color:'#818cf8', raw:true },
    { label:'🥈 Semi-final%',    value: (t.sf_pct*100).toFixed(1)+'%',   color:'#38bdf8', raw:true },
  ];

  const eloNorm = Math.min(1, Math.max(0, (t.elo - 1300) / 900));
  const winNorm = Math.min(1, t.win_pct * 5);

  document.getElementById('tp-detail').innerHTML = `
    <!-- Hero -->
    <div class="tp-d-hero">
      <div class="tp-d-flag">${teamFlag(name)}</div>
      <div class="tp-d-title">
        <div class="tp-d-name">${name}</div>
        <div class="tp-d-meta">
          <span class="tp-d-badge">Group ${t.group}</span>
          ${styleBadge(t.style)}
          <span class="tp-d-elo-big">ELO ${t.elo}</span>
          <span class="tp-win-big">Win ${pct(t.win_pct)}</span>
        </div>
      </div>
    </div>

    <!-- Tournament odds -->
    <div>
      <div class="tp-section-label">Tournament Probabilities</div>
      <div class="tp-d-prob">
        <div class="tp-prob-box">
          <div class="tp-prob-val" style="color:#fbbf24">${pct(t.win_pct)}</div>
          <div class="tp-prob-label">🏆 Win</div>
        </div>
        <div class="tp-prob-box">
          <div class="tp-prob-val" style="color:#818cf8">${pct(t.final_pct)}</div>
          <div class="tp-prob-label">🎯 Final</div>
        </div>
        <div class="tp-prob-box">
          <div class="tp-prob-val" style="color:#38bdf8">${pct(t.sf_pct)}</div>
          <div class="tp-prob-label">🥈 Semi</div>
        </div>
        <div class="tp-prob-box">
          <div class="tp-prob-val" style="color:#34d399">${pct(t.qf_pct||0)}</div>
          <div class="tp-prob-label">⚽ QF</div>
        </div>
      </div>
    </div>

    <!-- ELO bar -->
    <div>
      <div class="tp-section-label">ELO Rating  <span style="color:var(--text);font-size:13px;font-weight:700">${t.elo}</span></div>
      <div style="background:var(--border);border-radius:4px;height:8px;overflow:hidden">
        <div style="height:100%;width:${(eloNorm*100).toFixed(1)}%;background:linear-gradient(90deg,#fbbf24,#f97316);border-radius:4px;transition:width .4s"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:3px">
        <span>1300</span><span>2200</span>
      </div>
    </div>

    <!-- Key players -->
    ${(t.key_players||[]).length ? `
    <div>
      <div class="tp-section-label">Key Players</div>
      <div class="tp-d-players">
        ${(t.key_players||[]).map(p=>`<span class="tp-player-chip">⚽ ${p}</span>`).join('')}
      </div>
    </div>` : ''}

    <!-- Scouting notes -->
    ${t.notes ? `
    <div>
      <div class="tp-section-label">Scouting Notes</div>
      <div class="tp-d-notes">${t.notes}</div>
    </div>` : ''}

    <!-- Next matches -->
    <div>
      <div class="tp-section-label">Group ${t.group} Matches</div>
      <div style="display:flex;flex-direction:column;gap:6px">
        ${(DATA.group_matches||[]).filter(m=>m.home===name||m.away===name).map(m=>{
          const opp = m.home===name ? m.away : m.home;
          const side = m.home===name ? 'vs' : 'vs';
          const played = m.played;
          const score = played ? `<span style="color:var(--green)">${m.actual||''}</span>` :
                                 `<span style="color:var(--muted)">upcoming</span>`;
          return `<div style="display:flex;align-items:center;gap:10px;background:var(--surface);
                              border:1px solid var(--border);border-radius:8px;padding:9px 14px">
            <span style="font-size:18px">${teamFlag(opp)}</span>
            <span style="font-weight:600">${opp}</span>
            <span style="color:var(--muted);font-size:12px">${m.date||''}</span>
            <span style="margin-left:auto">${score}</span>
          </div>`;
        }).join('') || '<div style="color:var(--muted);font-size:13px">No matches found</div>'}
      </div>
    </div>

    ${teamProfileSections(name)}
  `;
}

// ── Extended profile: squad, 2026 stats, scorers, World Cup history ─────────
function teamProfileSections(name) {
  const tp = (DATA.team_profiles || {})[name];
  if (!tp) return '';
  const s = tp.stats;

  const statTile = (val, lbl) => `
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 12px;text-align:center">
      <div style="font-size:17px;font-weight:800;font-variant-numeric:tabular-nums">${val}</div>
      <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-top:2px">${lbl}</div>
    </div>`;

  const statsHtml = s ? `
    <div class="tp-section-label" style="margin-top:22px">2026 Tournament Stats</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:8px">
      ${statTile(`${s.w}-${s.d}-${s.l}`, `W-D-L · ${s.p} played`)}
      ${statTile(`${s.gf} / ${s.ga}`, 'Goals for / against')}
      ${statTile(s.shots, 'Shots')}
      ${statTile(s.sot, 'On target')}
      ${statTile(s.poss + '%', 'Avg possession')}
      ${statTile(s.pass_pct + '%', 'Pass accuracy')}
      ${statTile(s.corners, 'Corners')}
      ${statTile(s.fouls, 'Fouls')}
      ${statTile(`${s.yellows} 🟨 ${s.reds} 🟥`, 'Cards')}
      ${statTile(s.offsides, 'Offsides')}
      ${statTile(s.saves, 'Saves')}
    </div>` : '';

  const scorerList = (rows, valKey, icon) => rows.length ? rows.map((r, i) => `
    <div style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:var(--surface);
                border:1px solid var(--border);border-radius:8px;margin-bottom:5px">
      <span style="color:var(--muted);font-size:11px;width:14px">${i+1}</span>
      <span style="font-weight:600;font-size:13px">${r.player}</span>
      <span style="margin-left:auto;font-weight:800;color:var(--gold)">${icon} ${r[valKey]}${r.pens ? ` <span style="font-size:10px;color:var(--muted)">(${r.pens} pen)</span>` : ''}</span>
    </div>`).join('') : '<div style="color:var(--muted);font-size:12px">None yet</div>';

  const scorersHtml = (tp.top_scorers.length || tp.top_assists.length) ? `
    <div class="tp-section-label" style="margin-top:22px">Top Players — 2026</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
      <div>
        <div style="font-size:11px;color:var(--muted);font-weight:700;margin-bottom:6px">⚽ GOALS</div>
        ${scorerList(tp.top_scorers, 'goals', '⚽')}
      </div>
      <div>
        <div style="font-size:11px;color:var(--muted);font-weight:700;margin-bottom:6px">🅰️ ASSISTS</div>
        ${scorerList(tp.top_assists, 'assists', '🅰️')}
      </div>
    </div>` : '';

  const POS_LABEL = {GK: 'Goalkeepers', DF: 'Defenders', MF: 'Midfielders', FW: 'Forwards'};
  const byPos = {};
  (tp.players || []).forEach(p => (byPos[p.pos] = byPos[p.pos] || []).push(p));
  const squadHtml = tp.players && tp.players.length ? `
    <div class="tp-section-label" style="margin-top:22px">Squad
      <span style="font-weight:400;color:var(--muted);font-size:11px;margin-left:8px">
        ${tp.squad_size} players · avg age ${tp.avg_age ?? '—'} · avg height ${tp.avg_height ?? '—'} cm
        ${tp.coach ? ` · coach <b style="color:var(--text)">${tp.coach}</b>` : ''}
      </span>
    </div>
    ${['GK','DF','MF','FW'].filter(g => byPos[g]).map(g => `
      <div style="font-size:11px;color:var(--accent);font-weight:700;text-transform:uppercase;
                  letter-spacing:.05em;margin:12px 0 6px">${POS_LABEL[g]}</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px">
        ${byPos[g].sort((a,b)=>(+a.jersey||99)-(+b.jersey||99)).map(p => `
          <div style="display:flex;align-items:center;gap:9px;background:var(--surface);
                      border:1px solid var(--border);border-radius:8px;padding:7px 11px">
            <span style="width:24px;height:24px;border-radius:6px;background:var(--surface2);
                         display:inline-flex;align-items:center;justify-content:center;
                         font-size:11px;font-weight:700;color:var(--accent)">${p.jersey ?? '–'}</span>
            <div style="min-width:0">
              <div style="font-weight:600;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${p.name}</div>
              <div style="font-size:10.5px;color:var(--muted)">
                ${p.age ? p.age + ' yrs' : ''}${p.height_cm ? ` · ${p.height_cm} cm` : ''}${p.weight_kg ? ` · ${p.weight_kg} kg` : ''}
              </div>
            </div>
          </div>`).join('')}
      </div>`).join('')}` : '';

  const histHtml = `
    <div class="tp-section-label" style="margin-top:22px">World Cup History</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      ${statTile(tp.wc_appearances, 'WC appearances')}
      ${statTile(tp.wc_first, 'First appearance')}
      ${tp.wc_titles ? statTile('🏆'.repeat(Math.min(tp.wc_titles,5)), `${tp.wc_titles} World Cup title${tp.wc_titles>1?'s':''}`) : ''}
    </div>`;

  return statsHtml + scorersHtml + squadHtml + histHtml;
}

// ── Accuracy Tracker ───────────────────────────────────────────────────────
function renderAccuracy() {
  const acc = DATA.accuracy;
  const total = acc.total_played;
  const dirPct = acc.direction_pct;

  // Summary cards
  const dirColor = dirPct >= 70 ? 'var(--green)' : dirPct >= 50 ? 'var(--yellow)' : 'var(--red)';
  document.getElementById('acc-summary').innerHTML = `
    <div class="acc-stat">
      <div class="as-val" style="color:${dirColor}">${acc.direction_correct}/${total}</div>
      <div class="as-label">Correct Direction</div>
    </div>
    <div class="acc-stat">
      <div class="as-val" style="color:${dirColor}">${dirPct}%</div>
      <div class="as-label">Directional Accuracy</div>
    </div>
    <div class="acc-stat">
      <div class="as-val" style="color:var(--purple)">${acc.score_correct}/${total}</div>
      <div class="as-label">Exact Score Hits</div>
    </div>
    <div class="acc-stat">
      <div class="as-val" style="color:var(--gold)">${total}</div>
      <div class="as-label">Matches Played</div>
    </div>`;

  // ELO change chips
  const changes = acc.elo_changes || {};
  const chips = Object.entries(changes)
    .sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([team, delta]) => {
      const cls = delta > 0 ? 'up' : 'down';
      const sign = delta > 0 ? '+' : '';
      return `<span class="elo-chip ${cls}">${team} ${sign}${delta}</span>`;
    }).join('');
  document.getElementById('elo-change-list').innerHTML = chips || '<span style="color:var(--muted)">No results yet</span>';

  // Per-match breakdown
  if (!acc.matches || !acc.matches.length) {
    document.getElementById('acc-matches').innerHTML =
      '<div class="card"><p style="color:var(--muted);padding:20px">No matches played yet.</p></div>';
    return;
  }

  document.getElementById('acc-matches').innerHTML = acc.matches.map(m => {
    const hName = m.home_display || m.home;
    const aName = m.away_display || m.away;
    const dirOk = m.direction_correct;
    const scoreOk = m.score_correct;
    const icon = dirOk ? '✅' : '❌';
    const predW = m.predicted_winner === m.home ? hName : (m.predicted_winner === m.away ? aName : 'Draw');
    const actW  = m.actual_winner  === m.home ? hName : (m.actual_winner  === m.away ? aName : 'Draw');
    const borderC = dirOk ? 'var(--green)' : 'var(--red)';

    return `
      <div class="acc-match" style="border-left:4px solid ${borderC}">
        <div class="acc-icon">${icon}</div>
        <div>
          <div class="acc-teams">${['R32','R16','QF','SF','3rd','Final'].includes(m.group) ? {'R32':'R32','R16':'R16','QF':'QF','SF':'SF','3rd':'3rd Place','Final':'Final'}[m.group] : 'Group '+m.group} · ${hName} vs ${aName}
            <span style="color:var(--muted);font-size:12px;font-weight:400"> — ${m.date}</span>
          </div>
          <div class="acc-row">
            <div class="acc-box">
              <div class="ab-label">🤖 Model Predicted</div>
              <div class="ab-val" style="color:#93c5fd">${predW}</div>
              <div style="font-size:11px;color:var(--muted);margin-top:2px">
                ${hName} ${m.pre_p_home != null ? m.pre_p_home.toFixed(0) : '?'}% / Draw ${m.pre_p_draw != null ? m.pre_p_draw.toFixed(0) : '?'}% / ${aName} ${m.pre_p_away != null ? m.pre_p_away.toFixed(0) : '?'}%
              </div>
            </div>
            <div class="acc-box">
              <div class="ab-label">✅ Actual Result</div>
              <div class="ab-val" style="color:#86efac">${actW}</div>
              <div style="color:var(--muted);font-size:11px">Score: ${m.actual_score}</div>
            </div>
            <div class="acc-box">
              <div class="ab-label">Expected Goals</div>
              <div style="font-size:13px;font-weight:700;margin-top:4px">
                ${m.pre_xg_home != null ? m.pre_xg_home.toFixed(1) : '?'} – ${m.pre_xg_away != null ? m.pre_xg_away.toFixed(1) : '?'}
              </div>
              <div style="font-size:10px;color:var(--muted)">${hName} vs ${aName}</div>
            </div>
          </div>
          <div style="margin-top:8px;display:flex;gap:8px">
            <span class="correct-badge ${dirOk ? 'ok' : 'fail'}">${dirOk ? '✅ Direction correct' : '❌ Direction wrong'}</span>
            ${scoreOk ? '<span class="correct-badge ok">🎯 Exact score!</span>' : '<span class="correct-badge fail">Score missed</span>'}
          </div>
        </div>
      </div>`;
  }).join('');
}


// ── Online Learner status panel ─────────────────────────────────────────────
function renderOnlineLearner() {
  const ol = DATA.online_learning || {};
  const el = document.getElementById('ol-content');
  if (!ol || !ol.matches_processed) {
    el.innerHTML = '<p style="color:var(--muted);font-size:13px;padding:8px 0">No matches processed yet — learner will activate after first result.</p>';
    return;
  }
  const biases = ol.learned_biases || {};
  const fw = ol.factor_weights || {};
  const biasColor = v => v > 0.01 ? '#86efac' : v < -0.01 ? '#fca5a5' : '#94a3b8';

  const biasHTML = `
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 16px">
      ${Object.entries(biases).map(([k,v]) => `
        <div style="background:#1e293b;border-radius:8px;padding:10px 14px;min-width:130px">
          <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px">${k.replace(/_/g,' ')}</div>
          <div style="font-size:18px;font-weight:700;font-family:monospace;color:${biasColor(v)}">${v>=0?'+':''}${v.toFixed(4)}</div>
        </div>`).join('')}
    </div>`;

  const fwHTML = Object.keys(fw).length ? `
    <div style="font-size:12px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Factor Learned Weights</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      ${Object.entries(fw).filter(([,d])=>d.fires>0).map(([ft,d])=>{
        const wColor = d.learned_weight > 1.2 ? '#86efac' : d.learned_weight < 0.8 ? '#fca5a5' : '#94a3b8';
        const verdict = d.learned_weight > 1.2 ? '🔥 Boosted' : d.learned_weight < 0.8 ? '❌ Downweighted' : '✓ Neutral';
        const pct = Math.round(d.learned_weight * 100);
        return `<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:10px 14px;min-width:140px">
          <div style="font-size:11px;color:#94a3b8">${ft}</div>
          <div style="font-size:16px;font-weight:700;color:${wColor}">${pct}%</div>
          <div style="font-size:10px;color:#64748b">${d.fires} fires · ${d.accuracy_pct}% acc</div>
          <div style="font-size:10px;color:${wColor};margin-top:2px">${verdict}</div>
        </div>`;
      }).join('')}
    </div>` : '<p style="color:var(--muted);font-size:12px">No factor calibration data yet.</p>';

  el.innerHTML = `
    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px">
      <div><span style="color:#94a3b8;font-size:12px">Matches Processed</span><br><span style="font-size:22px;font-weight:700;color:#f1f5f9">${ol.matches_processed}</span></div>
      <div><span style="color:#94a3b8;font-size:12px">Direction Accuracy</span><br><span style="font-size:22px;font-weight:700;color:#86efac">${ol.direction_accuracy||'—'}</span></div>
      <div><span style="color:#94a3b8;font-size:12px">Exact Score</span><br><span style="font-size:22px;font-weight:700;color:#a78bfa">${ol.score_accuracy||'—'}</span></div>
      <div><span style="color:#94a3b8;font-size:12px">Last Updated</span><br><span style="font-size:13px;color:#94a3b8">${(ol.last_updated||'').slice(0,19).replace('T',' ') || '—'}</span></div>
    </div>
    <div style="font-size:12px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Systematic Bias Corrections Applied to Upcoming Predictions</div>
    ${biasHTML}
    ${fwHTML}`;
}


// ── Update overview with accuracy badge ────────────────────────────────────
function renderOverviewAccBadge() {
  const acc = DATA.accuracy;
  if (!acc || !acc.total_played) return;
  const el = document.getElementById('overview-top');
  const dirColor = acc.direction_pct >= 70 ? '#22c55e' : acc.direction_pct >= 50 ? '#eab308' : '#ef4444';
  const card = `
    <div class="ov-card">
      <div class="label">Model Accuracy</div>
      <div class="value" style="color:${dirColor}">${acc.direction_pct}%</div>
      <div class="sub">${acc.direction_correct}/${acc.total_played} correct · <a href="#" onclick="showTabAcc()" style="color:#60a5fa">See tracker →</a></div>
    </div>`;
  el.insertAdjacentHTML('afterbegin', card);
}
function showTabAcc() {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-accuracy').classList.add('active');
  document.getElementById('acc-tab-btn').classList.add('active');
}

// ══════════════════════ 🚩 CORNERS PREDICTION ══════════════════════════════

window.renderCornersMarkets = function() {
  const el = document.getElementById('corners-container');
  const corners = DATA.corners || [];
  const cl = DATA.corners_learning || {};
  if (corners.length === 0) {
    el.innerHTML = `<div class="flow-no-data"><h2>🚩 No corners data</h2></div>`;
    return;
  }

  const wf = cl.walk_forward || [];
  const lb = cl.learned_bias  || {};

  // ── Corners pick helper ───────────────────────────────────────────────────────
  // Pick the O/U line nearest to the model's expected value.
  // That line sits at ~50% and shows where we differ from a coin flip.
  function bestPick(exp, probTable) {
    if (exp == null || !probTable) return null;
    const lines = Object.keys(probTable).map(Number).sort((a,b)=>a-b);
    if (!lines.length) return null;
    // Scan lines within ±3 of expected value, skip trivially easy lines (>85%)
    const candidates = lines.filter(l => Math.abs(l - exp) <= 3);
    let best = null;
    for (const l of candidates) {
      const p = probTable[String(l)];
      if (!p) continue;
      if (p.over > 0.60 && p.over < 0.85 && (!best || p.over > best.prob))
        best = { line: l, side: 'OVER',  prob: p.over  };
      if (p.under > 0.60 && p.under < 0.85 && (!best || p.under > best.prob))
        best = { line: l, side: 'UNDER', prob: p.under };
    }
    return best;
  }

  function pickBadge(pick, compact) {
    if (!pick) return '<span style="color:var(--muted);font-size:11px">—</span>';
    const pct  = Math.round(pick.prob * 100);
    const col  = pct >= 65 ? '#22c55e' : pct >= 58 ? '#f59e0b' : '#94a3b8';
    const side = pick.side === 'OVER' ? 'O' : 'U';
    if (compact) {
      return `<span style="color:${col};font-weight:700;font-size:12px">${side} ${pick.line}</span>
              <span style="color:${col};font-size:10px"> ${pct}%</span>`;
    }
    return `<div style="font-weight:700;color:${col};font-size:13px">${pick.side} ${pick.line}</div>
            <div style="color:${col};font-size:11px">${pct}%</div>`;
  }

  function oeBadge(oe) {
    if (!oe || (!oe.odd && !oe.even)) return '<span style="color:var(--muted)">—</span>';
    const useOdd  = (oe.odd  || 0) >= (oe.even || 0);
    const prob    = useOdd ? oe.odd : oe.even;
    const label   = useOdd ? 'ODD' : 'EVEN';
    const pct     = Math.round(prob * 100);
    if (pct < 52) return '<span style="color:var(--muted);font-size:11px">PUSH</span>';
    const col = pct >= 60 ? '#22c55e' : pct >= 55 ? '#f59e0b' : '#94a3b8';
    return `<span style="color:${col};font-weight:700;font-size:12px">${label}</span>
            <span style="color:${col};font-size:10px"> ${pct}%</span>`;
  }

  // ── Accuracy summary + scoreboard table ──────────────────────────────────
  const n   = cl.n  || 0;
  const w2  = cl.within2 ?? 0;
  const mae = cl.mae ?? 0;
  const w2pct = n>0 ? Math.round(w2/n*100) : 0;
  const w2col = w2pct>=60?'#22c55e':w2pct>=45?'#f59e0b':'#ef4444';

  // Scoreboard table of all played matches
  const playedWf = (cl.walk_forward||[]);
  const scoreboard = playedWf.length ? `
  <div style="margin-bottom:16px">
    <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:6px">📋 All predictions vs results</div>
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:11px">
      <thead>
        <tr style="color:var(--muted);border-bottom:1px solid var(--border)">
          <th style="text-align:left;padding:4px 6px;font-weight:600">#</th>
          <th style="text-align:left;padding:4px 6px;font-weight:600">Match</th>
          <th style="text-align:center;padding:4px 6px;font-weight:600">Pred</th>
          <th style="text-align:center;padding:4px 6px;font-weight:600">Actual</th>
          <th style="text-align:center;padding:4px 6px;font-weight:600">Error</th>
          <th style="text-align:center;padding:4px 6px;font-weight:600">Result</th>
        </tr>
      </thead>
      <tbody>
        ${playedWf.map((w,i) => {
          const col = w.within2?'#22c55e':Math.abs(w.error)<=3?'#f59e0b':'#ef4444';
          const icon = w.within2?'✓':Math.abs(w.error)<=3?'~':'✗';
          const errSign = w.error>0?'+':'';
          return `<tr style="border-bottom:1px solid var(--border);opacity:${w.within2?1:0.85}">
            <td style="padding:3px 6px;color:var(--muted)">${i+1}</td>
            <td style="padding:3px 6px;white-space:nowrap"><b>${w.home}</b> <span style="color:var(--muted)">vs</span> ${w.away}</td>
            <td style="padding:3px 6px;text-align:center">${w.pred_total}</td>
            <td style="padding:3px 6px;text-align:center;font-weight:700">${w.actual_total}</td>
            <td style="padding:3px 6px;text-align:center;color:${col}">${errSign}${w.error}</td>
            <td style="padding:3px 6px;text-align:center;font-weight:700;color:${col}">${icon}</td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>
    </div>
  </div>` : '';

  const accHtml = n > 0 ? `
  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;align-items:stretch">
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 20px;text-align:center;min-width:100px">
      <div style="font-size:28px;font-weight:800;color:${w2col}">${w2}/${n}</div>
      <div style="font-size:11px;color:var(--muted);margin-top:2px">Within ±2</div>
      <div style="font-size:16px;font-weight:700;color:${w2col}">${w2pct}%</div>
    </div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 20px;text-align:center;min-width:100px">
      <div style="font-size:28px;font-weight:800;color:#94a3b8">${mae.toFixed(2)}</div>
      <div style="font-size:11px;color:var(--muted);margin-top:2px">Avg Error (MAE)</div>
    </div>
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 20px;text-align:center;min-width:100px">
      <div style="font-size:28px;font-weight:800;color:#ef4444">${n-w2}</div>
      <div style="font-size:11px;color:var(--muted);margin-top:2px">Misses (>±2)</div>
    </div>
  </div>
  ${scoreboard}` : '';

  // ── Rolling MAE chart ─────────────────────────────────────────────────────
  function learningChart() {
    if (wf.length < 2) return '';
    const W=560, H=110, PX=38, PY=12, PB=22;
    const maes   = wf.map(w => w.rolling_mae);
    const maxMAE = Math.max(...maes, 4);
    const pts = wf.map((w,i) => {
      const x = PX + (W-PX-8) * i / Math.max(wf.length-1, 1);
      const y = PY + (H-PY-PB) * (1 - w.rolling_mae / maxMAE);
      return [x, y, w];
    });
    const line = pts.map(([x,y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
    const yLbls = [1,2,3,4].filter(v=>v<=maxMAE+0.3).map(v => {
      const y = PY + (H-PY-PB) * (1 - v/maxMAE);
      return `<text x="${PX-3}" y="${y+3}" text-anchor="end" font-size="8" fill="var(--muted)">${v}</text>
              <line x1="${PX}" y1="${y}" x2="${W-4}" y2="${y}" stroke="var(--border)" stroke-width="0.5"/>`;
    }).join('');
    const xLbls = pts.map(([x,y,w]) =>
      `<text x="${x.toFixed(1)}" y="${H-5}" text-anchor="middle" font-size="7.5" fill="var(--muted)">${(w.away||'').split(' ').pop().slice(0,5)}</text>`
    ).join('');
    const dots = pts.map(([x,y,w]) => {
      const col = w.within2?'#22c55e':w.within3?'#f59e0b':'#ef4444';
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="${col}" stroke="var(--bg)" stroke-width="1.5">
                <title>${w.home} vs ${w.away}: pred ${w.pred_total} actual ${w.actual_total} err ${w.error>0?'+':''}${w.error}</title>
              </circle>`;
    }).join('');
    return `<div style="margin-bottom:14px">
      <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:4px">📉 Rolling MAE &nbsp;🟢≤±2 &nbsp;🟡≤±3 &nbsp;🔴&gt;±3</div>
      <svg width="${W}" height="${H}" style="display:block;overflow:visible">
        ${yLbls}
        <polyline points="${line}" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linejoin="round"/>
        ${xLbls}${dots}
      </svg></div>`;
  }

  // ── Learned bias pills ─────────────────────────────────────────────────────
  function biasSection() {
    const teams = Object.entries(lb).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).filter(([,v])=>Math.abs(v)>0.1);
    if (!teams.length) return '';
    const pills = teams.map(([team,bias]) => {
      const col = bias>0.3?'#22c55e':bias<-0.3?'#ef4444':'#94a3b8';
      const bg  = bias>0.3?'rgba(34,197,94,0.1)':bias<-0.3?'rgba(239,68,68,0.1)':'rgba(148,163,184,0.1)';
      return `<span style="display:inline-flex;align-items:center;gap:3px;background:${bg};color:${col};border-radius:99px;padding:2px 8px;font-size:11px;font-weight:600;white-space:nowrap">
        ${team} ${bias>0?'▲+':'▼'}${bias.toFixed(2)}
      </span>`;
    }).join(' ');
    return `<div style="margin-bottom:14px">
      <div style="font-size:11px;font-weight:600;color:var(--muted);margin-bottom:6px">🧠 Learned biases applied to upcoming predictions</div>
      <div style="display:flex;flex-wrap:wrap;gap:5px">${pills}</div>
    </div>`;
  }

  // ── Per-match card (played = result review, upcoming = picks) ──────────
  function matchCard(c) {
    const homeDisplay = c.home_display || c.home;
    const awayDisplay = c.away_display || c.away;
    const hBias = lb[c.home], aBias = lb[c.away];
    const bChip = (b) => b!=null && Math.abs(b)>=0.15
      ? `<sup style="font-size:9px;color:${b>0?'#22c55e':'#ef4444'}">${b>0?'▲+':'▼'}${Math.abs(b).toFixed(1)}</sup>` : '';

    // Derive picks
    const ftPick  = bestPick(c.total_exp, c.prob_total);
    const h1Pick  = bestPick(c.h1_exp,    c.prob_h1);
    const h2Pick  = bestPick(c.h2_exp,    c.prob_h2);
    const oeTotal = (c.odd_even||{}).total || {};

    const dateLabel  = c.date  ? `<span style="color:var(--muted);font-size:10px">${c.date}</span>` : '';
    const groupLabel = c.group ? `<span style="color:var(--muted);font-size:10px;margin-left:4px">${c.group.length===1?'Grp '+c.group:c.group}</span>` : '';

    // ── PLAYED: show result vs prediction ─────────────────────────────────
    if (c.played && c.actual_total != null) {
      const err    = c.total_exp - c.actual_total;
      const absErr = Math.abs(err);
      const col    = absErr<=1?'#22c55e':absErr<=2?'#f59e0b':'#ef4444';
      const sign   = err>0?'+':'';
      const checkFt = ftPick
        ? (ftPick.side==='OVER' ? c.actual_total > ftPick.line : c.actual_total <= ftPick.line)
        : null;
      const checkH1 = h1Pick && c.actual_total != null
        ? null  // we don't have 1H actual, so skip
        : null;
      const checkOE = oeTotal.odd != null
        ? ((c.actual_total % 2 !== 0) === (oeTotal.odd >= oeTotal.even))
        : null;

      const resultRow = (label, pick, hit) => {
        if (!pick) return '';
        const hitCol  = hit===true?'#22c55e':hit===false?'#ef4444':'#94a3b8';
        const hitIcon = hit===true?'✓':hit===false?'✗':'?';
        const pct = Math.round(pick.prob*100);
        const pcol = pct>=65?'#22c55e':pct>=58?'#f59e0b':'#94a3b8';
        return `<div style="display:flex;align-items:center;gap:8px;padding:3px 0">
          <span style="color:var(--muted);font-size:10px;width:28px">${label}</span>
          <span style="font-weight:700;color:${pcol};font-size:12px">${pick.side} ${pick.line}</span>
          <span style="color:${pcol};font-size:10px">${pct}%</span>
          <span style="color:${hitCol};font-weight:700;font-size:13px;margin-left:4px">${hitIcon}</span>
        </div>`;
      };
      const oeRow = (() => {
        if (!oeTotal.odd) return '';
        const useOdd = oeTotal.odd >= oeTotal.even;
        const prob = useOdd ? oeTotal.odd : oeTotal.even;
        const label = useOdd ? 'ODD' : 'EVEN';
        const pct = Math.round(prob*100);
        if (pct < 52) return '';
        const hit = checkOE;
        const hitCol = hit===true?'#22c55e':hit===false?'#ef4444':'#94a3b8';
        const hitIcon = hit===true?'✓':hit===false?'✗':'?';
        const pcol = pct>=65?'#22c55e':pct>=58?'#f59e0b':'#94a3b8';
        return `<div style="display:flex;align-items:center;gap:8px;padding:3px 0">
          <span style="color:var(--muted);font-size:10px;width:28px">O/E</span>
          <span style="font-weight:700;color:${pcol};font-size:12px">${label}</span>
          <span style="color:${pcol};font-size:10px">${pct}%</span>
          <span style="color:${hitCol};font-weight:700;font-size:13px;margin-left:4px">${hitIcon}</span>
        </div>`;
      })();

      // ── Model reasoning ───────────────────────────────────────────────
      const reasoning = c.reasoning || '';

      // ── Learning section ──────────────────────────────────────────────
      const learnHtml = (() => {
        const rows = [];
        const hBefore = c.wf_bias_home_before ?? 0;
        const aBefore = c.wf_bias_away_before ?? 0;
        const hAfter  = c.wf_bias_home_after;
        const aAfter  = c.wf_bias_away_after;
        const resH = c.wf_res_h, resA = c.wf_res_a;
        const biasRow = (team, before, after, res) => {
          if (after == null) return '';
          const delta = after - before;
          const col = delta > 0.05 ? '#22c55e' : delta < -0.05 ? '#ef4444' : '#94a3b8';
          const sign = delta >= 0 ? '+' : '';
          const resStr = res != null ? ` (actual−pred = ${res >= 0 ? '+' : ''}${res.toFixed(1)})` : '';
          return `<div style="font-size:11px;color:var(--muted)">
            <b style="color:var(--fg)">${team}</b> bias:
            <span style="color:#94a3b8">${before >= 0 ? '+' : ''}${before.toFixed(2)}</span> →
            <span style="color:${col};font-weight:700">${after >= 0 ? '+' : ''}${after.toFixed(2)}</span>
            <span style="color:${col}"> (${sign}${delta.toFixed(2)}${resStr})</span>
          </div>`;
        };
        rows.push(biasRow(c.home, hBefore, hAfter, resH));
        rows.push(biasRow(c.away, aBefore, aAfter, resA));
        const content = rows.filter(Boolean).join('');
        if (!content) return '';
        return `<div style="margin-top:8px;padding:8px 10px;background:rgba(59,130,246,0.06);border-left:2px solid #3b82f6;border-radius:0 6px 6px 0">
          <div style="font-size:10px;font-weight:700;color:#3b82f6;margin-bottom:4px">🧠 MODEL LEARNED FROM THIS MATCH</div>
          ${content}
        </div>`;
      })();

      const borderCol = absErr<=2?'#22c55e':absErr<=3?'#f59e0b':'#ef4444';
      return `<div class="card" style="margin-bottom:8px;padding:10px 14px;border-left:3px solid ${borderCol}">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <div style="flex:1;min-width:200px">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
              <span style="font-size:20px;font-weight:800;color:${borderCol}">${absErr<=2?'✓':absErr<=3?'~':'✗'}</span>
              <span style="font-weight:700;font-size:13px">${homeDisplay}${bChip(hBias)} vs ${awayDisplay}${bChip(aBias)}</span>
              <span>${dateLabel}${groupLabel}</span>
            </div>
            <div style="margin-top:4px;font-size:11px;color:var(--muted)">
              Predicted <b style="color:var(--fg)">${c.total_exp!=null?c.total_exp.toFixed(1):'—'}</b>
              &nbsp;·&nbsp; Actual <b style="color:${borderCol};font-size:13px">${c.actual_total}</b>
              <span style="color:var(--muted)">(${c.actual_home_corners??'?'}+${c.actual_away_corners??'?'})</span>
              &nbsp;·&nbsp; Error <b style="color:${borderCol}">${sign}${err.toFixed(1)}</b>
            </div>
          </div>
          <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
            ${resultRow('FT', ftPick, checkFt)}
            ${oeRow}
          </div>
        </div>
        ${reasoning ? `<div style="margin-top:6px;font-size:11px;color:var(--muted);font-style:italic;line-height:1.4">
          📊 ${reasoning}
        </div>` : ''}
        ${learnHtml}
      </div>`;
    }

    // ── UPCOMING: prediction picks — always show FT / 1H / 2H cards ─────────
    function pickCard(label, pick, sub) {
      if (pick) {
        const pct = Math.round(pick.prob * 100);
        const col = pct>=65?'#22c55e':pct>=58?'#f59e0b':'#94a3b8';
        const bg  = pct>=65?'rgba(34,197,94,0.08)':pct>=58?'rgba(245,158,11,0.08)':'rgba(148,163,184,0.06)';
        return `<div style="background:${bg};border:1px solid ${col}33;border-radius:8px;padding:8px 14px;min-width:130px">
          <div style="font-size:9px;font-weight:600;color:var(--muted);text-transform:uppercase;margin-bottom:2px">${label}</div>
          <div style="font-size:18px;font-weight:800;color:${col}">${pick.side} ${pick.line}</div>
          <div style="font-size:11px;color:${col}">${pct}% confidence</div>
          <div style="font-size:10px;color:var(--muted);margin-top:2px">${sub}</div>
        </div>`;
      }
      return `<div style="border:1px solid var(--border);border-radius:8px;padding:8px 14px;min-width:130px;opacity:0.55">
        <div style="font-size:9px;font-weight:600;color:var(--muted);text-transform:uppercase;margin-bottom:2px">${label}</div>
        <div style="font-size:16px;font-weight:700;color:var(--muted)">PUSH</div>
        <div style="font-size:10px;color:var(--muted);margin-top:2px">${sub}</div>
      </div>`;
    }
    const pickCards = [
      pickCard('Full Time', ftPick, `exp ${c.total_exp!=null?c.total_exp.toFixed(1):'?'}`),
      pickCard('1st Half',  h1Pick, `exp ${c.h1_exp!=null?c.h1_exp.toFixed(1):'?'}`),
      pickCard('2nd Half',  h2Pick, `exp ${c.h2_exp!=null?c.h2_exp.toFixed(1):'?'}`),
    ].join('');

    const oeCard = (() => {
      if (!oeTotal.odd) return '';
      const useOdd = oeTotal.odd >= oeTotal.even;
      const prob   = useOdd ? oeTotal.odd : oeTotal.even;
      const label  = useOdd ? 'ODD' : 'EVEN';
      const pct    = Math.round(prob * 100);
      if (pct < 60) return `<div style="border:1px solid var(--border);border-radius:8px;padding:8px 14px;min-width:130px;opacity:0.5">
        <div style="font-size:9px;font-weight:600;color:var(--muted);text-transform:uppercase;margin-bottom:2px">Odd / Even</div>
        <div style="font-size:15px;font-weight:700;color:var(--muted)">PUSH</div>
        <div style="font-size:10px;color:var(--muted)">too close (${pct}%)</div></div>`;
      const col = pct>=60?'#22c55e':pct>=55?'#f59e0b':'#94a3b8';
      const bg  = pct>=60?'rgba(34,197,94,0.08)':pct>=55?'rgba(245,158,11,0.08)':'rgba(148,163,184,0.06)';
      return `<div style="background:${bg};border:1px solid ${col}33;border-radius:8px;padding:8px 14px;min-width:130px">
        <div style="font-size:9px;font-weight:600;color:var(--muted);text-transform:uppercase;margin-bottom:2px">Odd / Even</div>
        <div style="font-size:18px;font-weight:800;color:${col}">${label}</div>
        <div style="font-size:11px;color:${col}">${pct}% confidence</div>
        <div style="font-size:10px;color:var(--muted);margin-top:2px">exp total ${c.total_exp!=null?c.total_exp.toFixed(1):'?'}</div>
      </div>`;
    })();

    const noPicks = false;

    return `<div class="card" style="margin-bottom:10px;padding:10px 14px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:6px;margin-bottom:8px">
        <div>
          <span style="font-weight:700;font-size:13px">${homeDisplay}${bChip(hBias)} vs ${awayDisplay}${bChip(aBias)}</span>
          <span style="margin-left:8px">${dateLabel}${groupLabel}</span>
          ${c.predicted_winner?`<span style="font-size:11px;color:#3b82f6;margin-left:8px">→ ${c.predicted_winner}</span>`:''}
        </div>
        <div style="font-size:11px;color:var(--muted)">
          Total <b style="color:var(--fg)">${c.total_exp!=null?c.total_exp.toFixed(1):'—'}</b>
          &nbsp;H <b>${c.home_exp!=null?c.home_exp.toFixed(1):'—'}</b>
          A <b>${c.away_exp!=null?c.away_exp.toFixed(1):'—'}</b>
          &nbsp;·&nbsp; 1H <b>${c.h1_exp!=null?c.h1_exp.toFixed(1):'—'}</b> / 2H <b>${c.h2_exp!=null?c.h2_exp.toFixed(1):'—'}</b>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">${pickCards}${oeCard}</div>
    </div>`;
  }

  // ── Group by round ────────────────────────────────────────────────────────
  const ROUND_ORDER = ['Group','R32','R16','QF','SF','3rd','Final'];
  const ROUND_LABEL = {Group:'Group Stage',R32:'Round of 32',R16:'Round of 16',QF:'Quarter-Finals',SF:'Semi-Finals','3rd':'Third-Place Playoff',Final:'Final'};
  const byRound = {};
  corners.forEach(c => { const r=c.round||c.group||'Group'; (byRound[r]=byRound[r]||[]).push(c); });

  let body = '';
  ROUND_ORDER.forEach(round => {
    const list = byRound[round]; if (!list) return;
    const played_   = list.filter(c => c.played);
    const upcoming_ = list.filter(c => !c.played);
    body += `<h3 style="font-size:14px;font-weight:700;margin:18px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--border)">${ROUND_LABEL[round]||round} — ${list.length} matches</h3>`;
    if (played_.length) {
      body += `<div style="font-size:11px;color:var(--muted);margin-bottom:6px">✅ Played — model picks vs actual result</div>`;
      body += played_.map(matchCard).join('');
    }
    if (upcoming_.length) {
      body += `<div style="font-size:11px;color:var(--muted);margin:10px 0 6px">🗓 Upcoming — prediction recommendations (learned biases applied)</div>`;
      body += upcoming_.map(matchCard).join('');
    }
  });

  el.innerHTML = `
  <div style="padding:4px 0 12px">
    <h2 style="font-size:16px;font-weight:700;margin-bottom:4px">🚩 Corners Model — Predictions & Picks</h2>
    <p style="font-size:12px;color:var(--muted);margin-bottom:12px">
      Walk-forward learning — biases update after each result, applied to all upcoming predictions.
      Confidence colour: <span style="color:#22c55e">■</span> ≥65% · <span style="color:#f59e0b">■</span> ≥58% · <span style="color:#94a3b8">■</span> ≥52% · model skips &lt;52%.
    </p>
    ${accHtml}
    ${learningChart()}
    ${biasSection()}
  </div>
  ${body}`;
}

// ── Scorers tab ──────────────────────────────────────────────────────────────
function renderScorers() {
  const el = document.getElementById('scorers-container');
  const S = DATA.scorers;
  if (!S || !S.golden_boot || !S.golden_boot.length) {
    el.innerHTML = '<div class="card"><p style="color:var(--muted);padding:20px">No scorer data yet.</p></div>';
    return;
  }
  const acc = DATA.accuracy || {};

  // Summary cards: match-result model + scorer-pick backtest
  const bt = S.backtest || {};
  const summary = `
    <div class="acc-summary">
      <div class="acc-stat"><div class="as-val" style="color:var(--green)">${acc.direction_correct}/${acc.total_played}</div>
        <div class="as-label">Match Result Model Correct (${acc.direction_pct}%)</div></div>
      <div class="acc-stat"><div class="as-val" style="color:${bt.hit_pct>=30?'var(--green)':'var(--yellow)'}">${bt.hits}/${bt.picks}</div>
        <div class="as-label">Scorer Picks Hit (${bt.hit_pct}%) — walk-forward backtest</div></div>
      <div class="acc-stat"><div class="as-val" style="color:var(--gold)">${S.total_goals}</div>
        <div class="as-label">Goals · ${S.matches_with_data} matches</div></div>
    </div>`;

  // Upcoming scorer picks
  let upcoming = '';
  if (S.upcoming && S.upcoming.length) {
    upcoming = `<div class="card" style="margin-bottom:20px">
      <div class="card-header"><span class="card-title">🎯 Likely Scorers — Upcoming Matches</span></div>` +
      S.upcoming.map(u => {
        const rnd = {R32:'Round of 32',R16:'Round of 16',QF:'Quarter-Final',SF:'Semi-Final','3rd':'3rd Place','Final':'Final'}[u.round]||u.round;
        return `<div style="padding:10px 4px;border-bottom:1px solid var(--border)">
          <div style="font-weight:700;margin-bottom:8px">${teamFlag(u.home)} ${u.home} vs ${u.away} ${teamFlag(u.away)}
            <span style="font-size:11px;color:var(--muted);margin-left:8px">M${u.match_no} · ${rnd} · ${u.date}</span></div>
          <div style="display:flex;gap:8px;flex-wrap:wrap">` +
          u.picks.map(p => {
            const col = p.prob>=55?'#22c55e':p.prob>=40?'#f59e0b':'#94a3b8';
            return `<div style="background:${col}14;border:1px solid ${col}44;border-radius:8px;padding:6px 12px">
              <div style="font-weight:700;font-size:13px">${p.player}</div>
              <div style="font-size:11px;color:var(--muted)">${teamFlag(p.team)} ${p.team} · ${p.goals}⚽ so far</div>
              <div style="font-size:14px;font-weight:800;color:${col}">${p.prob}% to score</div>
            </div>`;
          }).join('') + `</div></div>`;
      }).join('') + `</div>`;
  }

  // Golden boot table
  const boot = `<div class="card" style="margin-bottom:20px">
    <div class="card-header"><span class="card-title">👟 Golden Boot Race</span></div>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="color:var(--muted);text-align:left;font-size:11px">
        <th style="padding:6px 8px">#</th><th>Player</th><th>Team</th><th style="text-align:right">Goals</th><th style="text-align:right;padding-right:8px">Pens</th></tr></thead>
      <tbody>` +
      S.golden_boot.map((r, i) => `<tr style="border-top:1px solid var(--border)">
        <td style="padding:6px 8px;color:var(--muted)">${i+1}</td>
        <td style="font-weight:${i<3?800:400}">${i===0?'👑 ':''}${r.player}</td>
        <td>${teamFlag(r.team)} ${r.team}</td>
        <td style="text-align:right;font-weight:800;color:var(--gold)">${r.goals}</td>
        <td style="text-align:right;padding-right:8px;color:var(--muted)">${r.pens||''}</td></tr>`).join('') +
      `</tbody></table></div>`;

  // Per-team breakdown (collapsible)
  const teams = `<div class="card">
    <div class="card-header"><span class="card-title">🌍 Who Scored — By Team</span></div>` +
    S.by_team.map(t => `<details style="border-bottom:1px solid var(--border);padding:6px 4px">
      <summary style="cursor:pointer;font-weight:700;font-size:13px">${teamFlag(t.team)} ${t.team}
        <span style="color:var(--gold);margin-left:6px">${t.total} goals</span>
        <span style="color:var(--muted);font-size:11px;margin-left:6px">${t.players.length} scorers</span></summary>
      <div style="padding:8px 12px;display:flex;gap:8px;flex-wrap:wrap">` +
      t.players.map(p => `<span style="background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-size:12px">
        ${p.player} <b style="color:var(--gold)">${p.goals}</b>${p.pens?`<span style="font-size:10px;color:var(--muted)"> (${p.pens} pen)</span>`:''}</span>`).join('') +
      `</div></details>`).join('') + `</div>`;

  // Recent backtest picks detail
  const btRows = (bt.rows||[]).slice().reverse().slice(0, 20);
  const btDetail = btRows.length ? `<div class="card" style="margin-top:20px">
    <div class="card-header"><span class="card-title">🔬 Recent Scorer Picks — Did They Score?</span></div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;padding:4px 0">` +
    btRows.map(r => `<span style="border:1px solid ${r.hit?'#22c55e55':'var(--border)'};background:${r.hit?'#22c55e11':'transparent'};border-radius:6px;padding:4px 10px;font-size:11px">
      ${r.hit?'✅':'❌'} ${r.pick} <span style="color:var(--muted)">(${r.team}, M${r.match_no})</span></span>`).join('') +
    `</div></div>` : '';

  el.innerHTML = summary + upcoming + boot + teams + btDetail;
}

// ── Hero stats + About modal ────────────────────────────────────────────────
function renderHeroStats() {
  const el = document.getElementById('hero-stats');
  if (!el) return;
  const acc = DATA.accuracy || {};
  const played = acc.total_played || 0;
  const sc = DATA.scorers || {};
  const koSched = DATA.actual_ko_schedule || [];
  const next = koSched.find(m => !m.played && !m.home_tbd && !m.away_tbd);
  const stage = next ? ({R32:'Round of 32',R16:'Round of 16',QF:'Quarter-Finals',SF:'Semi-Finals','3rd':'3rd Place',Final:'The Final'}[next.round] || next.round) : 'Tournament complete';
  const chips = [
    { val: `${acc.direction_pct || 0}%`, lbl: `Model accuracy · ${acc.direction_correct||0}/${played}`,
      col: (acc.direction_pct||0) >= 70 ? 'var(--green)' : 'var(--yellow)' },
    { val: `${played} / 104`, lbl: 'Matches tracked', col: 'var(--accent)' },
    { val: `${sc.total_goals || '—'}`, lbl: 'Goals recorded', col: 'var(--gold)' },
    { val: stage, lbl: next ? `Next: ${next.home} vs ${next.away}` : 'Awaiting champion', col: '#fff' },
    { val: DATA.champion || '—', lbl: 'Projected champion', col: 'var(--gold)' },
  ];
  el.innerHTML = chips.map(c => `
    <div class="hero-chip">
      <div class="hc-val" style="color:${c.col}">${c.val}</div>
      <div class="hc-lbl">${c.lbl}</div>
    </div>`).join('');
}
window.openAbout  = () => document.getElementById('about-overlay').classList.add('open');
window.closeAbout = () => document.getElementById('about-overlay').classList.remove('open');
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeAbout(); });

// ── Init ───────────────────────────────────────────────────────────────────
renderHeroStats();
renderOverview();
renderDateMatches();
renderGroupMatches();
renderStandings();
renderKO();
renderTeams();
renderAccuracy();
renderScorers();
renderOnlineLearner();
renderCornersMarkets();

// UTC+7 clock
function updateUTC7Clock() {
  const now = new Date();
  const utc7 = new Date(now.getTime() + (7 * 60 * 60 * 1000));
  const hh = String(utc7.getUTCHours()).padStart(2,'0');
  const mm = String(utc7.getUTCMinutes()).padStart(2,'0');
  const ss = String(utc7.getUTCSeconds()).padStart(2,'0');
  const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const dayName = days[utc7.getUTCDay()];
  const dateStr = `${dayName}, ${months[utc7.getUTCMonth()]} ${utc7.getUTCDate()}, ${utc7.getUTCFullYear()}`;
  const timeEl = document.getElementById('utc7-time');
  const dateEl = document.getElementById('utc7-date');
  if (timeEl) timeEl.textContent = `${hh}:${mm}:${ss}`;
  if (dateEl) dateEl.textContent = dateStr;
}
updateUTC7Clock();
setInterval(updateUTC7Clock, 1000);

// ── Rebuild button ──────────────────────────────────────────────────────────
function triggerRebuild() {
  const btn = document.getElementById('rebuild-btn');
  if (!btn) return;
  btn.textContent = '⏳ Rebuilding…';
  btn.disabled = true;
  btn.style.opacity = '0.6';
  // Hit the /rebuild endpoint that server.py handles (uses cached ML pipeline)
  fetch('/api/rebuild')
    .then(r => r.text())
    .then(() => {
      btn.textContent = '✅ Done — reloading…';
      // Page auto-reloads via SSE when build finishes
      // Fallback: force reload after 35s if SSE didn't fire
      setTimeout(() => {
        btn.textContent = '⚡ Rebuild (~25s)';
        btn.disabled = false;
        btn.style.opacity = '1';
      }, 35000);
    })
    .catch(() => {
      // Not running via server.py — open instructions
      btn.textContent = '⚠️ Run server.py first';
      btn.style.borderColor = '#f59e0b';
      btn.style.color = '#fbbf24';
      setTimeout(() => {
        btn.textContent = '🔄 Rebuild Now';
        btn.disabled = false;
        btn.style.opacity = '1';
        btn.style.borderColor = '#3b82f6';
        btn.style.color = '#93c5fd';
      }, 4000);
    });
}
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

_MODEL_CACHE = Path("models/pipeline_cache.pkl")


def _load_cached_pipeline():
    """Load cached (outcome_model, goals_model, elo_ratings, team_stats) if fresh."""
    if not _MODEL_CACHE.exists():
        return None
    try:
        import pickle, hashlib
        # Cache is valid as long as schedule.py and results.csv haven't changed
        sched_mtime = Path("schedule.py").stat().st_mtime
        csv_mtime   = Path("data/results.csv").stat().st_mtime if Path("data/results.csv").exists() else 0
        sig = f"{sched_mtime:.0f}_{csv_mtime:.0f}"
        with open(_MODEL_CACHE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("sig") == sig:
            print("[pipeline] ✅ Using cached models (schedule + CSV unchanged)")
            return cached
    except Exception as e:
        print(f"[pipeline] Cache miss ({e})")
    return None


def _save_cached_pipeline(outcome_model, goals_model, elo_ratings, team_stats):
    import pickle
    sched_mtime = Path("schedule.py").stat().st_mtime
    csv_mtime   = Path("data/results.csv").stat().st_mtime if Path("data/results.csv").exists() else 0
    sig = f"{sched_mtime:.0f}_{csv_mtime:.0f}"
    Path("models").mkdir(exist_ok=True)
    with open(_MODEL_CACHE, "wb") as f:
        pickle.dump({
            "sig": sig,
            "outcome_model": outcome_model,
            "goals_model": goals_model,
            "elo_ratings": elo_ratings,
            "team_stats": team_stats,
        }, f)
    print("[pipeline] 💾 Pipeline cached for next rebuild")


def main():
    parser = argparse.ArgumentParser(description="Generate WC 2026 prediction dashboard")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-download of live scores and training CSV")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force full ML retrain even if cache is valid")
    args = parser.parse_args()

    np.random.seed(42)

    live_mode = os.environ.get("GD_LIVE") == "1"

    # ── Step 0: Pull latest live scores & training data from GitHub ───────────
    if live_mode:
        # Live rebuild: the score watcher already patched schedule.py from ESPN
        print("\n=== Live rebuild — skipping external score/CSV refresh ===")
    else:
        print("\n=== Fetching live WC 2026 scores ===")
        from live_updater import patch_schedule_with_live_scores, also_refresh_training_data
        also_refresh_training_data(force=args.force_refresh)
        n_updated = patch_schedule_with_live_scores(force=args.force_refresh)
        if n_updated:
            print(f"   🔄 {n_updated} score(s) updated — retraining on latest data")
        else:
            print("   ✓ Scores up to date")

    # ── Training pipeline (cached when nothing changed) ───────────────────────
    cached = None if args.force_retrain else _load_cached_pipeline()
    if cached:
        outcome_model = cached["outcome_model"]
        goals_model   = cached["goals_model"]
        elo_ratings   = cached["elo_ratings"]
        team_stats    = cached["team_stats"]
    else:
        print("\n=== Training pipeline ===")
        df = data.load()
        df_elo, elo_ratings, elo_last_played = elo_module.compute_elo(df)
        feat_df = feat_module.build_features(df_elo)
        outcome_model, goals_model = model_module.train_all(feat_df)
        team_stats = build_team_stats_lookup(feat_df)
        _save_cached_pipeline(outcome_model, goals_model, elo_ratings, team_stats)

    print("\n=== Building all prediction data ===")
    payload = build_all_data(goals_model, outcome_model, elo_ratings, team_stats)

    print("\n=== Generating dashboard HTML ===")

    html = HTML_TEMPLATE \
        .replace("{{DATA_JSON}}", json.dumps(payload, ensure_ascii=False))

    Path("output").mkdir(exist_ok=True)
    out = Path("output/dashboard.html")
    out.write_text(html, encoding="utf-8")

    size_kb = out.stat().st_size // 1024
    print(f"\n✅  Dashboard saved → {out}  ({size_kb} KB)")
    print(f"   Open in any browser: open {out}")


if __name__ == "__main__":
    main()
