"""
fetch_team_stats.py — Full per-match team statistics from ESPN box scores.

Stores every statistic ESPN provides (shots, possession, passes, corners,
fouls, cards, saves, tackles, …) per team per played match, keyed
"home vs away" in data/match_stats.json. Only new matches are fetched.
GD_LIVE=1 uses cache only. Run standalone: python3 fetch_team_stats.py
"""

from __future__ import annotations
import json, os, ssl, time, urllib.request
from pathlib import Path

from schedule import SCHEDULE

OUTPUT_PATH = Path("data/match_stats.json")
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


def fetch_all_team_stats() -> dict:
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
        print(f"[teamstats] Live mode — {len(existing)} cached, no network")
        return existing

    if not todo:
        print(f"[teamstats] All {len(played)} played matches cached")
        return existing

    from fetch_corner_stats import fetch_event_ids
    event_ids = fetch_event_ids()
    print(f"[teamstats] Fetching box scores for {len(todo)} matches...")

    for m in todo:
        key = f"{_norm(m['home'])} vs {_norm(m['away'])}"
        eid = event_ids.get(key) or event_ids.get(f"{_norm(m['away'])} vs {_norm(m['home'])}")
        if not eid:
            print(f"  ✗ {m['home']} vs {m['away']}: no event ID")
            continue
        try:
            summ = _get(f"{ESPN_SUMM}?event={eid}")
            sides = {}
            for t in summ.get("boxscore", {}).get("teams", []):
                hoa = t.get("homeAway", "")
                sides[hoa] = {s["name"]: s.get("displayValue")
                              for s in t.get("statistics", [])}
            if sides:
                existing[key] = {
                    "match_no": m["match_no"], "date": m["date"],
                    "home": m["home"], "away": m["away"],
                    "home_stats": sides.get("home", {}),
                    "away_stats": sides.get("away", {}),
                }
                print(f"  ✓ {m['home']} vs {m['away']}")
        except Exception as e:
            print(f"  ✗ {m['home']} vs {m['away']}: {e}")
        time.sleep(0.2)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(existing, indent=1, ensure_ascii=False))
    print(f"[teamstats] Saved {len(existing)} matches → {OUTPUT_PATH}")
    return existing


if __name__ == "__main__":
    fetch_all_team_stats()
