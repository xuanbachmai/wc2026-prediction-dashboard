"""
fetch_scorers.py — Per-player goal data from ESPN match summaries.

For every played match, pulls keyEvents and extracts goals:
    {scorer, team, clock, kind: goal|pen|og, assist}

Cached in data/scorers.json keyed by "home vs away" (lowercase). Matches
already cached are not re-fetched. GD_LIVE=1 skips all network calls.

Run standalone:  python3 fetch_scorers.py
"""

from __future__ import annotations
import json, os, ssl, time, urllib.request
from pathlib import Path

from schedule import SCHEDULE

OUTPUT_PATH = Path("data/scorers.json")
ESPN_SUMM   = "https://site.api.espn.com/apis/site/v2/sports/soccer/FIFA.WORLD/summary"

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode    = ssl.CERT_NONE


def _norm(s: str) -> str:
    return s.strip().lower()


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSL, timeout=15) as r:
        return json.loads(r.read())


def _extract_goals(summary: dict) -> list[dict]:
    goals = []
    for ev in summary.get("keyEvents", []):
        ttext = (ev.get("type", {}) or {}).get("text", "") or ""
        tl = ttext.lower()
        # Count: "Goal", "Goal - Header", "Own Goal", "Penalty - Scored"
        is_goal = "goal" in tl or ("penalty" in tl and "scored" in tl)
        if not is_goal:
            continue
        # Exclude: shootout kicks, disallowed/overturned goals
        if any(w in tl for w in ("shootout", "disallowed", "cancelled", "missed")):
            continue
        kind = "og" if "own" in tl else ("pen" if "penalty" in tl else "goal")
        participants = ev.get("participants", []) or []
        scorer = ""
        assist = ""
        if participants:
            scorer = (participants[0].get("athlete", {}) or {}).get("displayName", "")
            if len(participants) > 1:
                assist = (participants[1].get("athlete", {}) or {}).get("displayName", "")
        team = (ev.get("team", {}) or {}).get("displayName", "")
        goals.append({
            "scorer": scorer,
            "assist": assist,
            "team":   team,
            "clock":  (ev.get("clock", {}) or {}).get("displayValue", ""),
            "kind":   kind,
        })
    return goals


def fetch_all_scorers() -> dict:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    existing: dict = {}
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
        except Exception:
            pass

    played = [m for m in SCHEDULE if m.get("home_score") is not None]
    todo = [m for m in played
            if f"{_norm(m['home'])} vs {_norm(m['away'])}" not in existing]

    if os.environ.get("GD_LIVE") == "1":
        print(f"[scorers] Live mode — {len(existing)} cached, no network")
        return existing

    if not todo:
        print(f"[scorers] All {len(played)} played matches cached")
        return existing

    from fetch_corner_stats import fetch_event_ids
    event_ids = fetch_event_ids()
    print(f"[scorers] Fetching goals for {len(todo)} matches...")

    for m in todo:
        key = f"{_norm(m['home'])} vs {_norm(m['away'])}"
        eid = event_ids.get(key) or event_ids.get(f"{_norm(m['away'])} vs {_norm(m['home'])}")
        if not eid:
            print(f"  ✗ {m['home']} vs {m['away']}: no event ID")
            continue
        try:
            goals = _extract_goals(_get(f"{ESPN_SUMM}?event={eid}"))
            existing[key] = {
                "match_no": m["match_no"], "date": m["date"],
                "home": m["home"], "away": m["away"],
                "goals": goals,
            }
            print(f"  ✓ {m['home']} vs {m['away']}: {len(goals)} goals")
        except Exception as e:
            print(f"  ✗ {m['home']} vs {m['away']}: {e}")
        time.sleep(0.2)

    OUTPUT_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"[scorers] Saved {len(existing)} matches → {OUTPUT_PATH}")
    return existing


if __name__ == "__main__":
    fetch_all_scorers()
