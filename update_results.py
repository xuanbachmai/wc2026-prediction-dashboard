"""
update_results.py — Fetch finished match results from ESPN and write them
into schedule.py (including penalty shootouts).

Used two ways:
  - imported by serve.py (live watcher + hourly backfill)
  - run standalone: python3 update_results.py   (e.g. from CI / GitHub Actions)

Exit code 0 = ran fine; prints "CHANGED" if any result was written.
"""

from __future__ import annotations
import json, re, ssl, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

ESPN_SCORE    = "https://site.api.espn.com/apis/site/v2/sports/soccer/FIFA.WORLD/scoreboard"
SCHEDULE_PATH = Path(__file__).parent / "schedule.py"

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode    = ssl.CERT_NONE

# ESPN display name → schedule.py team name
NAME_MAP = {
    "türkiye": "Turkey", "turkey": "Turkey",
    "united states": "United States", "usa": "United States",
    "côte d'ivoire": "Ivory Coast", "ivory coast": "Ivory Coast",
    "curaçao": "Curacao", "curacao": "Curacao",
    "czechia": "Czech Republic", "czech republic": "Czech Republic",
    "korea republic": "South Korea", "south korea": "South Korea",
    "dr congo": "Congo DR", "congo dr": "Congo DR",
    "democratic republic of the congo": "Congo DR",
    "cape verde": "Cabo Verde", "cabo verde": "Cabo Verde",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "ir iran": "Iran", "iran": "Iran",
}


def norm(name: str) -> str:
    return NAME_MAP.get(name.lower(), name)


def espn_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSL, timeout=15) as r:
        return json.loads(r.read())


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def patch_schedule(home: str, away: str, hs: int, as_: int,
                   hp: int | None = None, ap: int | None = None) -> bool:
    """Write a result (and optional penalty shootout) into schedule.py.
    Matches within a single {...} entry so it can't bleed across matches."""
    text = SCHEDULE_PATH.read_text()
    entry_pat = (r'\{[^{}]*?"home":\s*"' + re.escape(home) +
                 r'"[^{}]*?"away":\s*"' + re.escape(away) + r'"[^{}]*?\}')
    m = re.search(entry_pat, text, flags=re.DOTALL)
    if not m:
        log(f"⚠️ patch_schedule: no entry found for {home} vs {away}")
        return False
    block = m.group(0)
    new_block = re.sub(r'("home_score":\s*)(?:None|\d+)', rf'\g<1>{hs}', block)
    new_block = re.sub(r'("away_score":\s*)(?:None|\d+)', rf'\g<1>{as_}', new_block)
    if hp is not None and ap is not None:
        if '"home_pens"' in new_block:
            new_block = re.sub(r'("home_pens":\s*)(?:None|\d+)', rf'\g<1>{hp}', new_block)
            new_block = re.sub(r'("away_pens":\s*)(?:None|\d+)', rf'\g<1>{ap}', new_block)
        else:
            new_block = re.sub(
                r'("away_score":\s*(?:None|\d+))',
                rf'\g<1>, "home_pens": {hp}, "away_pens": {ap}',
                new_block, count=1)
    if new_block == block:
        return False
    text = text[:m.start()] + new_block + text[m.end():]
    SCHEDULE_PATH.write_text(text)
    return True


def _parse_event(ev: dict):
    """Extract (home, away, hs, as_, hp, ap, state) from an ESPN event."""
    comp   = ev["competitions"][0]
    state  = comp["status"]["type"]["state"]
    teams  = comp["competitors"]
    t_home = next(t for t in teams if t["homeAway"] == "home")
    t_away = next(t for t in teams if t["homeAway"] == "away")
    home = norm(t_home["team"]["displayName"])
    away = norm(t_away["team"]["displayName"])
    hs   = int(t_home.get("score", 0)) if state != "pre" else None
    as_  = int(t_away.get("score", 0)) if state != "pre" else None
    hp = t_home.get("shootoutScore")
    ap = t_away.get("shootoutScore")
    hp = int(hp) if hp is not None else None
    ap = int(ap) if ap is not None else None
    return home, away, hs, as_, hp, ap, state


def resolve_ko_placeholders() -> bool:
    """Replace 'TBD (Wnn)' / 'TBD (Lnn)' slots with the actual winner/loser of
    match nn once it has been played (penalty shootouts decide drawn KO games)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib
    import schedule as sched_mod
    importlib.reload(sched_mod)

    outcome: dict[str, str] = {}
    for m in sched_mod.SCHEDULE:
        if m["home_score"] is None or m["away_score"] is None:
            continue
        hs, as_ = m["home_score"], m["away_score"]
        hp, ap  = m.get("home_pens"), m.get("away_pens")
        if hs > as_:
            w, l = m["home"], m["away"]
        elif as_ > hs:
            w, l = m["away"], m["home"]
        elif hp is not None and ap is not None:
            w, l = (m["home"], m["away"]) if hp > ap else (m["away"], m["home"])
        else:
            continue    # drawn group match — no winner to propagate
        outcome[f"W{m['match_no']}"] = w
        outcome[f"L{m['match_no']}"] = l

    text = SCHEDULE_PATH.read_text()
    changed = False
    for token, team in outcome.items():
        placeholder = f'"TBD ({token})"'
        if placeholder in text:
            text = text.replace(placeholder, f'"{team}"')
            log(f"🔁 Resolved TBD ({token}) → {team}")
            changed = True
    if changed:
        SCHEDULE_PATH.write_text(text)
    return changed


def backfill_results() -> bool:
    """Scan schedule.py for past/today matches missing a score and fetch them
    from ESPN by date. Also resolves KO TBD slots from played results.
    Returns True if anything was written."""
    changed = resolve_ko_placeholders()   # resolve first so new pairings backfill below

    sys.path.insert(0, str(Path(__file__).parent))
    import importlib
    import schedule as sched_mod
    importlib.reload(sched_mod)

    today = datetime.now().strftime("%Y-%m-%d")
    missing = [m for m in sched_mod.SCHEDULE
               if m["home_score"] is None
               and m["date"] <= today
               and not m["home"].startswith("TBD")
               and not m["away"].startswith("TBD")]
    if not missing:
        log("Backfill: nothing missing")
        return changed

    log(f"🔎 Backfill: {len(missing)} matches missing results")
    for date in sorted({m["date"] for m in missing}):
        espn_date = date.replace("-", "")
        try:
            data = espn_get(f"{ESPN_SCORE}?dates={espn_date}")
        except Exception as e:
            log(f"Backfill fetch error for {date}: {e}")
            continue
        for ev in data.get("events", []):
            try:
                home, away, hs, as_, hp, ap, state = _parse_event(ev)
            except Exception:
                continue
            if state != "post":
                continue
            if not any(m["home"] == home and m["away"] == away for m in missing):
                continue
            if patch_schedule(home, away, hs, as_, hp, ap):
                pens = f" (pens {hp}–{ap})" if hp is not None else ""
                log(f"✅ Backfilled: {home} {hs}–{as_} {away}{pens}  ({date})")
                changed = True
        time.sleep(0.3)
    if changed:
        resolve_ko_placeholders()   # new results may unlock next-round slots
    return changed


def verify_results() -> bool:
    """Re-check ALL recorded results against ESPN and correct any drift
    (the live watcher can record a score just before a late goal)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib
    import schedule as sched_mod
    importlib.reload(sched_mod)

    played = [m for m in sched_mod.SCHEDULE if m["home_score"] is not None]

    # ESPN buckets by US-Eastern date, which can differ from the schedule
    # date — fetch every date ±1 day and match events against ALL played.
    from datetime import timedelta
    dates: set[str] = set()
    for m in played:
        d = datetime.strptime(m["date"], "%Y-%m-%d")
        for off in (-1, 0):
            dates.add((d + timedelta(days=off)).strftime("%Y-%m-%d"))

    changed = False
    seen_events: set[tuple] = set()
    for date in sorted(dates):
        try:
            data = espn_get(f"{ESPN_SCORE}?dates={date.replace('-', '')}")
        except Exception as e:
            log(f"Verify fetch error {date}: {e}")
            continue
        for ev in data.get("events", []):
            try:
                home, away, hs, as_, hp, ap, state = _parse_event(ev)
            except Exception:
                continue
            if state != "post" or (home, away) in seen_events:
                continue
            seen_events.add((home, away))
            m = next((x for x in played
                      if x["home"] == home and x["away"] == away), None)
            if m is None:
                continue
            if (m["home_score"] != hs or m["away_score"] != as_
                    or m.get("home_pens") != hp or m.get("away_pens") != ap):
                old = f"{m['home_score']}–{m['away_score']}"
                if patch_schedule(home, away, hs, as_, hp, ap):
                    pens = f" (pens {hp}–{ap})" if hp is not None else ""
                    log(f"🛠 Corrected: {home} vs {away}  {old} → {hs}–{as_}{pens}")
                    changed = True
        time.sleep(0.3)
    if changed:
        resolve_ko_placeholders()
    else:
        log("Verify: all recorded results match ESPN")
    return changed


if __name__ == "__main__":
    changed = backfill_results()
    changed = verify_results() or changed
    if changed:
        print("CHANGED")
