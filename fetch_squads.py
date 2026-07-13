"""
fetch_squads.py — Full 26-man squads for all 48 teams from ESPN.

Per player: name, jersey, position group, age, date of birth, height (cm),
weight (kg). Per team: coach. Cached in data/squads.json; GD_LIVE=1 uses
cache only. Run standalone: python3 fetch_squads.py
"""

from __future__ import annotations
import json, os, re, ssl, time, urllib.request
from pathlib import Path

OUTPUT_PATH = Path("data/squads.json")
ESPN_TEAMS  = "https://site.api.espn.com/apis/site/v2/sports/soccer/FIFA.WORLD/teams?limit=60"
ESPN_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/soccer/FIFA.WORLD/teams/{id}/roster"

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode    = ssl.CERT_NONE

# ESPN display name → schedule.py team name
from update_results import NAME_MAP


def _norm_team(name: str) -> str:
    return NAME_MAP.get(name.lower(), name)


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSL, timeout=15) as r:
        return json.loads(r.read())


def _height_cm(display: str | None) -> int | None:
    """'6\\' 2"' → 188"""
    if not display:
        return None
    m = re.match(r"(\d+)'\s*(\d+)", display)
    if not m:
        return None
    return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54)


def _weight_kg(display: str | None) -> int | None:
    """'196 lbs' → 89"""
    if not display:
        return None
    m = re.match(r"(\d+)", display)
    return round(int(m.group(1)) * 0.4536) if m else None


POS_GROUP = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}


def fetch_all_squads() -> dict:
    existing: dict = {}
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
        except Exception:
            pass

    if os.environ.get("GD_LIVE") == "1":
        print(f"[squads] Live mode — {len(existing)} cached teams, no network")
        return existing

    data = _get(ESPN_TEAMS)
    teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    print(f"[squads] {len(teams)} teams on ESPN")

    result: dict = {}
    for entry in teams:
        t = entry["team"]
        name = _norm_team(t["displayName"])
        try:
            roster = _get(ESPN_ROSTER.format(id=t["id"]))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            if name in existing:
                result[name] = existing[name]
            continue
        players = []
        for a in roster.get("athletes", []):
            pos = (a.get("position") or {}).get("abbreviation", "")
            players.append({
                "name":      a.get("displayName", ""),
                "jersey":    a.get("jersey"),
                "pos":       POS_GROUP.get(pos, pos or "?"),
                "age":       a.get("age"),
                "dob":       (a.get("dateOfBirth") or "")[:10],
                "height_cm": _height_cm(a.get("displayHeight")),
                "weight_kg": _weight_kg(a.get("displayWeight")),
            })
        coaches = roster.get("coach") or []
        coach = coaches[-1] if isinstance(coaches, list) and coaches else None
        result[name] = {
            "espn_id": t["id"],
            "coach":   f"{coach['firstName']} {coach['lastName']}" if coach else None,
            "players": players,
        }
        print(f"  ✓ {name}: {len(players)} players · coach {result[name]['coach']}")
        time.sleep(0.15)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"[squads] Saved {len(result)} teams → {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    fetch_all_squads()
