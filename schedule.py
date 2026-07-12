"""
schedule.py — Official FIFA World Cup 2026 match schedule.

Source: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums

Fields per match
────────────────
  match_no   : official FIFA match number
  date       : ISO date string  "2026-06-11"
  group      : "A"–"L" for group stage, "R32"/"R16"/"QF"/"SF"/"3rd"/"Final"
  matchday   : 1 / 2 / 3 for group stage
  home       : team name (data/results.csv spelling)
  away       : team name (data/results.csv spelling)
  stadium    : venue name
  city       : host city
  home_score : int or None (None = not yet played)
  away_score : int or None
"""

from __future__ import annotations

# ── Schedule ──────────────────────────────────────────────────────────────────

SCHEDULE: list[dict] = [

    # ══════════════════════ MATCHDAY 1 ══════════════════════

    # Thursday 11 June
    {"match_no": 1,  "date": "2026-06-11", "group": "A", "matchday": 1,
     "home": "Mexico",       "away": "South Africa",  "stadium": "Mexico City Stadium",     "city": "Mexico City",
     "home_score": 2, "away_score": 0},

    {"match_no": 2,  "date": "2026-06-11", "group": "A", "matchday": 1,
     "home": "South Korea",  "away": "Czech Republic","stadium": "Estadio Akron Guadalajara","city": "Guadalajara",
     "home_score": 2, "away_score": 1},

    # Friday 12 June
    {"match_no": 3,  "date": "2026-06-12", "group": "B", "matchday": 1,
     "home": "Canada",       "away": "Bosnia and Herzegovina","stadium": "BMO Field Toronto","city": "Toronto",
     "home_score": 1, "away_score": 1},

    {"match_no": 4,  "date": "2026-06-12", "group": "D", "matchday": 1,
     "home": "United States","away": "Paraguay",      "stadium": "SoFi Stadium",            "city": "Los Angeles",
     "home_score": 4, "away_score": 1},

    # Saturday 13 June
    {"match_no": 5,  "date": "2026-06-13", "group": "C", "matchday": 1,
     "home": "Haiti",        "away": "Scotland",      "stadium": "Gillette Stadium",        "city": "Boston",
     "home_score": 0, "away_score": 1},   # Scotland 1–0 Haiti (McGinn 28')

    {"match_no": 6,  "date": "2026-06-13", "group": "D", "matchday": 1,
     "home": "Australia",    "away": "Turkey",        "stadium": "BC Place",                "city": "Vancouver",
     "home_score": 2, "away_score": 0},   # Australia 2–0 Türkiye (Irankunda 27', Metcalfe 75')

    {"match_no": 7,  "date": "2026-06-13", "group": "C", "matchday": 1,
     "home": "Brazil",       "away": "Morocco",       "stadium": "MetLife Stadium",         "city": "New York/New Jersey",
     "home_score": 1, "away_score": 1},   # Brazil 1–1 Morocco

    {"match_no": 8,  "date": "2026-06-13", "group": "B", "matchday": 1,
     "home": "Qatar",        "away": "Switzerland",   "stadium": "Levi's Stadium",          "city": "San Francisco Bay Area",
     "home_score": 1, "away_score": 1},   # Qatar 1–1 Switzerland (Embolo pen 17')

    # Sunday 14 June
    {"match_no": 9,  "date": "2026-06-14", "group": "E", "matchday": 1,
     "home": "Ivory Coast",  "away": "Ecuador",       "stadium": "Lincoln Financial Field", "city": "Philadelphia",
     "home_score": 1, "away_score": 0},   # Ivory Coast 1–0 Ecuador

    {"match_no": 10, "date": "2026-06-14", "group": "E", "matchday": 1,
     "home": "Germany",      "away": "Curacao",       "stadium": "NRG Stadium",             "city": "Houston",
     "home_score": 7, "away_score": 1},   # Germany 7–1 Curaçao (Nmecha, Schlotterbeck, Havertz x2, Musiala, Brown, Undav / Comenencia)

    {"match_no": 11, "date": "2026-06-14", "group": "F", "matchday": 1,
     "home": "Netherlands",  "away": "Japan",         "stadium": "AT&T Stadium",            "city": "Dallas",
     "home_score": 2, "away_score": 2},   # Netherlands 2–2 Japan

    {"match_no": 12, "date": "2026-06-14", "group": "F", "matchday": 1,
     "home": "Sweden",       "away": "Tunisia",       "stadium": "Estadio BBVA Monterrey",  "city": "Monterrey",
     "home_score": 5, "away_score": 1},   # Sweden 5–1 Tunisia

    # Monday 15 June
    {"match_no": 13, "date": "2026-06-15", "group": "H", "matchday": 1,
     "home": "Saudi Arabia", "away": "Uruguay",       "stadium": "Hard Rock Stadium",       "city": "Miami",
     "home_score": 1, "away_score": 1},  # Saudi Arabia 1–1 Uruguay (FT)

    {"match_no": 14, "date": "2026-06-15", "group": "H", "matchday": 1,
     "home": "Spain",        "away": "Cabo Verde",    "stadium": "Mercedes-Benz Stadium",   "city": "Atlanta",
     "home_score": 0, "away_score": 0},  # Spain 0–0 Cabo Verde (FT)

    {"match_no": 15, "date": "2026-06-15", "group": "G", "matchday": 1,
     "home": "Iran",         "away": "New Zealand",   "stadium": "SoFi Stadium",            "city": "Los Angeles",
     "home_score": 2, "away_score": 2},  # Iran 2–2 New Zealand (FT)

    {"match_no": 16, "date": "2026-06-15", "group": "G", "matchday": 1,
     "home": "Belgium",      "away": "Egypt",         "stadium": "Lumen Field",             "city": "Seattle",
     "home_score": 1, "away_score": 1},  # Belgium 1–1 Egypt (FT)

    # Tuesday 16 June
    {"match_no": 17, "date": "2026-06-16", "group": "I", "matchday": 1,
     "home": "France",       "away": "Senegal",       "stadium": "MetLife Stadium",         "city": "New York/New Jersey",
     "home_score": 3, "away_score": 1},  # France 3–1 Senegal

    {"match_no": 18, "date": "2026-06-16", "group": "I", "matchday": 1,
     "home": "Iraq",         "away": "Norway",        "stadium": "Gillette Stadium",        "city": "Boston",
     "home_score": 1, "away_score": 4},  # Iraq 1–4 Norway

    {"match_no": 19, "date": "2026-06-16", "group": "J", "matchday": 1,
     "home": "Argentina",    "away": "Algeria",       "stadium": "Children's Mercy Park",   "city": "Kansas City",
     "home_score": 3, "away_score": 0},  # Argentina 3–0 Algeria

    {"match_no": 20, "date": "2026-06-17", "group": "J", "matchday": 1,
     "home": "Austria",      "away": "Jordan",        "stadium": "Levi's Stadium",          "city": "San Francisco Bay Area",
     "home_score": 3, "away_score": 1},  # Austria 3–1 Jordan

    # Wednesday 17 June
    {"match_no": 21, "date": "2026-06-17", "group": "L", "matchday": 1,
     "home": "Ghana",        "away": "Panama",        "stadium": "BMO Field Toronto",       "city": "Toronto",
     "home_score": 1, "away_score": 0},

    {"match_no": 22, "date": "2026-06-17", "group": "L", "matchday": 1,
     "home": "England",      "away": "Croatia",       "stadium": "AT&T Stadium",            "city": "Dallas",
     "home_score": 4, "away_score": 2},

    {"match_no": 23, "date": "2026-06-17", "group": "K", "matchday": 1,
     "home": "Portugal",     "away": "Congo DR",      "stadium": "NRG Stadium",             "city": "Houston",
     "home_score": 1, "away_score": 1},

    {"match_no": 24, "date": "2026-06-17", "group": "K", "matchday": 1,
     "home": "Uzbekistan",   "away": "Colombia",      "stadium": "Mexico City Stadium",     "city": "Mexico City",
     "home_score": 1, "away_score": 3},

    # ══════════════════════ MATCHDAY 2 ══════════════════════

    # Thursday 18 June
    {"match_no": 25, "date": "2026-06-18", "group": "A", "matchday": 2,
     "home": "Czech Republic","away": "South Africa", "stadium": "Mercedes-Benz Stadium",   "city": "Atlanta",
     "home_score": 1, "away_score": 1},

    {"match_no": 26, "date": "2026-06-18", "group": "B", "matchday": 2,
     "home": "Switzerland",  "away": "Bosnia and Herzegovina","stadium": "SoFi Stadium",    "city": "Los Angeles",
     "home_score": 4, "away_score": 1},

    {"match_no": 27, "date": "2026-06-18", "group": "B", "matchday": 2,
     "home": "Canada",       "away": "Qatar",         "stadium": "BC Place",                "city": "Vancouver",
     "home_score": 6, "away_score": 0},

    {"match_no": 28, "date": "2026-06-18", "group": "A", "matchday": 2,
     "home": "Mexico",       "away": "South Korea",   "stadium": "Estadio Akron Guadalajara","city": "Guadalajara",
     "home_score": 1, "away_score": 0},

    # Friday 19 June
    {"match_no": 29, "date": "2026-06-19", "group": "C", "matchday": 2,
     "home": "Brazil",       "away": "Haiti",         "stadium": "Lincoln Financial Field", "city": "Philadelphia",
     "home_score": 3, "away_score": 0},

    {"match_no": 30, "date": "2026-06-19", "group": "C", "matchday": 2,
     "home": "Scotland",     "away": "Morocco",       "stadium": "Gillette Stadium",        "city": "Boston",
     "home_score": 0, "away_score": 1},

    {"match_no": 31, "date": "2026-06-19", "group": "D", "matchday": 2,
     "home": "Turkey",       "away": "Paraguay",      "stadium": "Levi's Stadium",          "city": "San Francisco Bay Area",
     "home_score": 0, "away_score": 1},

    {"match_no": 32, "date": "2026-06-19", "group": "D", "matchday": 2,
     "home": "United States","away": "Australia",     "stadium": "Lumen Field",             "city": "Seattle",
     "home_score": 2, "away_score": 0},

    # Saturday 20 June
    {"match_no": 33, "date": "2026-06-20", "group": "E", "matchday": 2,
     "home": "Germany",      "away": "Ivory Coast",   "stadium": "BMO Field Toronto",       "city": "Toronto",
     "home_score": 2, "away_score": 1},

    {"match_no": 34, "date": "2026-06-20", "group": "E", "matchday": 2,
     "home": "Ecuador",      "away": "Curacao",       "stadium": "Children's Mercy Park",   "city": "Kansas City",
     "home_score": 0, "away_score": 0},

    {"match_no": 35, "date": "2026-06-20", "group": "F", "matchday": 2,
     "home": "Netherlands",  "away": "Sweden",        "stadium": "NRG Stadium",             "city": "Houston",
     "home_score": 5, "away_score": 1},

    {"match_no": 36, "date": "2026-06-20", "group": "F", "matchday": 2,
     "home": "Tunisia",      "away": "Japan",         "stadium": "Estadio BBVA Monterrey",  "city": "Monterrey",
     "home_score": 0, "away_score": 4},

    # Sunday 21 June
    {"match_no": 37, "date": "2026-06-21", "group": "H", "matchday": 2,
     "home": "Uruguay",      "away": "Cabo Verde",    "stadium": "Hard Rock Stadium",       "city": "Miami",
     "home_score": 2, "away_score": 2},

    {"match_no": 38, "date": "2026-06-21", "group": "H", "matchday": 2,
     "home": "Spain",        "away": "Saudi Arabia",  "stadium": "Mercedes-Benz Stadium",   "city": "Atlanta",
     "home_score": 4, "away_score": 0},

    {"match_no": 39, "date": "2026-06-21", "group": "G", "matchday": 2,
     "home": "Belgium",      "away": "Iran",          "stadium": "SoFi Stadium",            "city": "Los Angeles",
     "home_score": 0, "away_score": 0},

    {"match_no": 40, "date": "2026-06-21", "group": "G", "matchday": 2,
     "home": "New Zealand",  "away": "Egypt",         "stadium": "BC Place",                "city": "Vancouver",
     "home_score": 1, "away_score": 3},

    # Monday 22 June
    {"match_no": 41, "date": "2026-06-22", "group": "I", "matchday": 2,
     "home": "Norway",       "away": "Senegal",       "stadium": "MetLife Stadium",         "city": "New York/New Jersey",
     "home_score": 3, "away_score": 2},

    {"match_no": 42, "date": "2026-06-22", "group": "I", "matchday": 2,
     "home": "France",       "away": "Iraq",          "stadium": "Lincoln Financial Field", "city": "Philadelphia",
     "home_score": 3, "away_score": 0},

    {"match_no": 43, "date": "2026-06-22", "group": "J", "matchday": 2,
     "home": "Argentina",    "away": "Austria",       "stadium": "AT&T Stadium",            "city": "Dallas",
     "home_score": 2, "away_score": 0},

    {"match_no": 44, "date": "2026-06-22", "group": "J", "matchday": 2,
     "home": "Jordan",       "away": "Algeria",       "stadium": "Levi's Stadium",          "city": "San Francisco Bay Area",
     "home_score": 1, "away_score": 2},

    # Tuesday 23 June
    {"match_no": 45, "date": "2026-06-23", "group": "L", "matchday": 2,
     "home": "England",      "away": "Ghana",         "stadium": "Gillette Stadium",        "city": "Boston",
     "home_score": 0, "away_score": 0},

    {"match_no": 46, "date": "2026-06-23", "group": "L", "matchday": 2,
     "home": "Panama",       "away": "Croatia",       "stadium": "BMO Field Toronto",       "city": "Toronto",
     "home_score": 0, "away_score": 1},

    {"match_no": 47, "date": "2026-06-23", "group": "K", "matchday": 2,
     "home": "Portugal",     "away": "Uzbekistan",    "stadium": "NRG Stadium",             "city": "Houston",
     "home_score": 5, "away_score": 0},

    {"match_no": 48, "date": "2026-06-23", "group": "K", "matchday": 2,
     "home": "Colombia",     "away": "Congo DR",      "stadium": "Estadio Akron Guadalajara","city": "Guadalajara",
     "home_score": 1, "away_score": 0},

    # ══════════════════════ MATCHDAY 3 (simultaneous) ══════════════════════

    # Wednesday 24 June — Group A & B & C (all 3 simultaneous per group)
    {"match_no": 49, "date": "2026-06-24", "group": "C", "matchday": 3,
     "home": "Scotland",     "away": "Brazil",        "stadium": "Hard Rock Stadium",       "city": "Miami",
     "home_score": 0, "away_score": 3},

    {"match_no": 50, "date": "2026-06-24", "group": "C", "matchday": 3,
     "home": "Morocco",      "away": "Haiti",         "stadium": "Mercedes-Benz Stadium",   "city": "Atlanta",
     "home_score": 4, "away_score": 2},

    {"match_no": 51, "date": "2026-06-24", "group": "B", "matchday": 3,
     "home": "Switzerland",  "away": "Canada",        "stadium": "BC Place",                "city": "Vancouver",
     "home_score": 2, "away_score": 1},

    {"match_no": 52, "date": "2026-06-24", "group": "B", "matchday": 3,
     "home": "Bosnia and Herzegovina","away": "Qatar","stadium": "Lumen Field",             "city": "Seattle",
     "home_score": 3, "away_score": 1},

    {"match_no": 53, "date": "2026-06-24", "group": "A", "matchday": 3,
     "home": "Czech Republic","away": "Mexico",       "stadium": "Mexico City Stadium",     "city": "Mexico City",
     "home_score": 0, "away_score": 3},

    {"match_no": 54, "date": "2026-06-24", "group": "A", "matchday": 3,
     "home": "South Africa", "away": "South Korea",   "stadium": "Estadio BBVA Monterrey",  "city": "Monterrey",
     "home_score": 1, "away_score": 0},

    # Thursday 25 June — Group D, E & F
    {"match_no": 55, "date": "2026-06-25", "group": "E", "matchday": 3,
     "home": "Curacao",      "away": "Ivory Coast",   "stadium": "Lincoln Financial Field", "city": "Philadelphia",
     "home_score": 0, "away_score": 2},

    {"match_no": 56, "date": "2026-06-25", "group": "E", "matchday": 3,
     "home": "Ecuador",      "away": "Germany",       "stadium": "MetLife Stadium",         "city": "New York/New Jersey",
     "home_score": 2, "away_score": 1},

    {"match_no": 57, "date": "2026-06-25", "group": "F", "matchday": 3,
     "home": "Japan",        "away": "Sweden",        "stadium": "AT&T Stadium",            "city": "Dallas",
     "home_score": 1, "away_score": 1},

    {"match_no": 58, "date": "2026-06-25", "group": "F", "matchday": 3,
     "home": "Tunisia",      "away": "Netherlands",   "stadium": "Children's Mercy Park",   "city": "Kansas City",
     "home_score": 1, "away_score": 3},

    {"match_no": 59, "date": "2026-06-25", "group": "D", "matchday": 3,
     "home": "Turkey",       "away": "United States", "stadium": "SoFi Stadium",            "city": "Los Angeles",
     "home_score": 3, "away_score": 2},

    {"match_no": 60, "date": "2026-06-25", "group": "D", "matchday": 3,
     "home": "Paraguay",     "away": "Australia",     "stadium": "Levi's Stadium",          "city": "San Francisco Bay Area",
     "home_score": 0, "away_score": 0},

    # Friday 26 June — Group G, H & I
    {"match_no": 61, "date": "2026-06-26", "group": "I", "matchday": 3,
     "home": "Norway",       "away": "France",        "stadium": "Gillette Stadium",        "city": "Boston",
     "home_score": 1, "away_score": 4},

    {"match_no": 62, "date": "2026-06-26", "group": "I", "matchday": 3,
     "home": "Senegal",      "away": "Iraq",          "stadium": "BMO Field Toronto",       "city": "Toronto",
     "home_score": 5, "away_score": 0},

    {"match_no": 63, "date": "2026-06-26", "group": "G", "matchday": 3,
     "home": "Egypt",        "away": "Iran",          "stadium": "Lumen Field",             "city": "Seattle",
     "home_score": 1, "away_score": 1},

    {"match_no": 64, "date": "2026-06-26", "group": "G", "matchday": 3,
     "home": "New Zealand",  "away": "Belgium",       "stadium": "BC Place",                "city": "Vancouver",
     "home_score": 1, "away_score": 5},

    {"match_no": 65, "date": "2026-06-26", "group": "H", "matchday": 3,
     "home": "Cabo Verde",   "away": "Saudi Arabia",  "stadium": "NRG Stadium",             "city": "Houston",
     "home_score": 0, "away_score": 0},

    {"match_no": 66, "date": "2026-06-26", "group": "H", "matchday": 3,
     "home": "Uruguay",      "away": "Spain",         "stadium": "Estadio Akron Guadalajara","city": "Guadalajara",
     "home_score": 0, "away_score": 1},

    # Saturday 27 June — Group J, K & L
    {"match_no": 67, "date": "2026-06-27", "group": "L", "matchday": 3,
     "home": "Panama",       "away": "England",       "stadium": "MetLife Stadium",         "city": "New York/New Jersey",
     "home_score": 0, "away_score": 2},

    {"match_no": 68, "date": "2026-06-27", "group": "L", "matchday": 3,
     "home": "Croatia",      "away": "Ghana",         "stadium": "Lincoln Financial Field", "city": "Philadelphia",
     "home_score": 2, "away_score": 1},

    {"match_no": 69, "date": "2026-06-27", "group": "J", "matchday": 3,
     "home": "Algeria",      "away": "Austria",       "stadium": "Children's Mercy Park",   "city": "Kansas City",
     "home_score": 3, "away_score": 3},

    {"match_no": 70, "date": "2026-06-27", "group": "J", "matchday": 3,
     "home": "Jordan",       "away": "Argentina",     "stadium": "AT&T Stadium",            "city": "Dallas",
     "home_score": 1, "away_score": 3},

    {"match_no": 71, "date": "2026-06-27", "group": "K", "matchday": 3,
     "home": "Colombia",     "away": "Portugal",      "stadium": "Hard Rock Stadium",       "city": "Miami",
     "home_score": 0, "away_score": 0},

    {"match_no": 72, "date": "2026-06-27", "group": "K", "matchday": 3,
     "home": "Congo DR",     "away": "Uzbekistan",    "stadium": "Mercedes-Benz Stadium",   "city": "Atlanta",
     "home_score": 3, "away_score": 1},

    # ══════════════════════ ROUND OF 32 ══════════════════════

    {"match_no": 73, "date": "2026-06-28", "group": "R32", "matchday": None,
     "home": "South Africa", "away": "Canada",     "stadium": "SoFi Stadium",           "city": "Los Angeles",
     "home_score": 0, "away_score": 1},

    {"match_no": 74, "date": "2026-06-29", "group": "R32", "matchday": None,
     "home": "Germany",      "away": "Paraguay",  "stadium": "Gillette Stadium",        "city": "Boston",
     "home_score": 1, "away_score": 1, "home_pens": 3, "away_pens": 4},

    {"match_no": 75, "date": "2026-06-29", "group": "R32", "matchday": None,
     "home": "Netherlands",  "away": "Morocco",   "stadium": "Estadio BBVA Monterrey",  "city": "Monterrey",
     "home_score": 1, "away_score": 1, "home_pens": 2, "away_pens": 3},

    {"match_no": 76, "date": "2026-06-29", "group": "R32", "matchday": None,
     "home": "Brazil",       "away": "Japan",     "stadium": "NRG Stadium",             "city": "Houston",
     "home_score": 2, "away_score": 1},

    {"match_no": 77, "date": "2026-06-30", "group": "R32", "matchday": None,
     "home": "France",        "away": "Sweden",    "stadium": "MetLife Stadium",         "city": "New York/New Jersey",
     "home_score": 3, "away_score": 0},

    {"match_no": 78, "date": "2026-06-30", "group": "R32", "matchday": None,
     "home": "Ivory Coast",  "away": "Norway",   "stadium": "AT&T Stadium",            "city": "Dallas",
     "home_score": 1, "away_score": 2},

    {"match_no": 79, "date": "2026-06-30", "group": "R32", "matchday": None,
     "home": "Mexico",        "away": "Ecuador",  "stadium": "Mexico City Stadium",     "city": "Mexico City",
     "home_score": 2, "away_score": 0},

    {"match_no": 80, "date": "2026-07-01", "group": "R32", "matchday": None,
     "home": "England",       "away": "Congo DR",  "stadium": "Mercedes-Benz Stadium",   "city": "Atlanta",
     "home_score": 2, "away_score": 1},

    {"match_no": 81, "date": "2026-07-01", "group": "R32", "matchday": None,
     "home": "United States",           "away": "Bosnia and Herzegovina", "stadium": "Levi's Stadium",          "city": "San Francisco Bay Area",
     "home_score": 2, "away_score": 0},

    {"match_no": 82, "date": "2026-07-01", "group": "R32", "matchday": None,
     "home": "Belgium",       "away": "Senegal",   "stadium": "Lumen Field",             "city": "Seattle",
     "home_score": 3, "away_score": 2},

    {"match_no": 83, "date": "2026-07-02", "group": "R32", "matchday": None,
     "home": "Portugal",      "away": "Croatia",   "stadium": "BMO Field Toronto",       "city": "Toronto",
     "home_score": 2, "away_score": 1},

    {"match_no": 84, "date": "2026-07-02", "group": "R32", "matchday": None,
     "home": "Spain",         "away": "Austria",   "stadium": "SoFi Stadium",            "city": "Los Angeles",
     "home_score": 3, "away_score": 0},

    {"match_no": 85, "date": "2026-07-02", "group": "R32", "matchday": None,
     "home": "Switzerland",   "away": "Algeria",   "stadium": "BC Place",                "city": "Vancouver",
     "home_score": 2, "away_score": 0},

    {"match_no": 86, "date": "2026-07-03", "group": "R32", "matchday": None,
     "home": "Argentina",     "away": "Cabo Verde", "stadium": "Hard Rock Stadium",      "city": "Miami",
     "home_score": 3, "away_score": 2},

    {"match_no": 87, "date": "2026-07-03", "group": "R32", "matchday": None,
     "home": "Colombia",      "away": "Ghana",     "stadium": "Children's Mercy Park",   "city": "Kansas City",
     "home_score": 1, "away_score": 0},

    {"match_no": 88, "date": "2026-07-03", "group": "R32", "matchday": None,
     "home": "Australia",     "away": "Egypt",     "stadium": "AT&T Stadium",            "city": "Dallas",
     "home_score": 1, "away_score": 1, "home_pens": 2, "away_pens": 4},

    # ══════════════════════ ROUND OF 16 ══════════════════════
    {"match_no": 89, "date": "2026-07-04", "group": "R16", "matchday": None,
     "home": "Paraguay",    "away": "France",    "stadium": "Lincoln Financial Field", "city": "Philadelphia", "home_score": 0, "away_score": 1},
    {"match_no": 90, "date": "2026-07-04", "group": "R16", "matchday": None,
     "home": "Canada",      "away": "Morocco",   "stadium": "NRG Stadium",             "city": "Houston",      "home_score": 0, "away_score": 3},
    {"match_no": 91, "date": "2026-07-05", "group": "R16", "matchday": None,
     "home": "Brazil",      "away": "Norway",    "stadium": "MetLife Stadium",         "city": "New York/New Jersey", "home_score": 1, "away_score": 2},
    {"match_no": 92, "date": "2026-07-05", "group": "R16", "matchday": None,
     "home": "Mexico",      "away": "England",   "stadium": "Mexico City Stadium",     "city": "Mexico City",  "home_score": 2, "away_score": 3},
    {"match_no": 93, "date": "2026-07-06", "group": "R16", "matchday": None,
     "home": "Portugal",    "away": "Spain",     "stadium": "AT&T Stadium",            "city": "Dallas",       "home_score": 0, "away_score": 1},
    {"match_no": 94, "date": "2026-07-06", "group": "R16", "matchday": None,
     "home": "United States",         "away": "Belgium",   "stadium": "Lumen Field",             "city": "Seattle",      "home_score": 1, "away_score": 4},
    {"match_no": 95, "date": "2026-07-07", "group": "R16", "matchday": None,
     "home": "Argentina",   "away": "Egypt",     "stadium": "Mercedes-Benz Stadium",   "city": "Atlanta",      "home_score": 3, "away_score": 2},
    {"match_no": 96, "date": "2026-07-07", "group": "R16", "matchday": None,
     "home": "Switzerland", "away": "Colombia",  "stadium": "BC Place",                "city": "Vancouver",    "home_score": 0, "away_score": 0, "home_pens": 4, "away_pens": 3},

    # ══════════════════════ QUARTER-FINALS ══════════════════════
    {"match_no": 97,  "date": "2026-07-09", "group": "QF", "matchday": None,
     "home": "France",      "away": "Morocco",   "stadium": "Gillette Stadium",        "city": "Boston",       "home_score": 2, "away_score": 0},
    {"match_no": 98,  "date": "2026-07-10", "group": "QF", "matchday": None,
     "home": "Spain",       "away": "Belgium",   "stadium": "SoFi Stadium",            "city": "Los Angeles",  "home_score": 2, "away_score": 1},
    {"match_no": 99,  "date": "2026-07-11", "group": "QF", "matchday": None,
     "home": "Norway",      "away": "England",   "stadium": "Hard Rock Stadium",       "city": "Miami",        "home_score": 1, "away_score": 2},
    {"match_no": 100, "date": "2026-07-11", "group": "QF", "matchday": None,
     "home": "Argentina",  "away": "Switzerland", "stadium": "Children's Mercy Park",   "city": "Kansas City",  "home_score": 3, "away_score": 1},

    # ══════════════════════ SEMI-FINALS ══════════════════════
    {"match_no": 101, "date": "2026-07-14", "group": "SF", "matchday": None,
     "home": "France", "away": "Spain", "stadium": "AT&T Stadium",            "city": "Dallas",       "home_score": None, "away_score": None},
    {"match_no": 102, "date": "2026-07-15", "group": "SF", "matchday": None,
     "home": "England", "away": "Argentina","stadium": "Mercedes-Benz Stadium",   "city": "Atlanta",      "home_score": None, "away_score": None},

    # ══════════════════════ BRONZE FINAL ══════════════════════
    {"match_no": 103, "date": "2026-07-18", "group": "3rd", "matchday": None,
     "home": "TBD (L101)","away": "TBD (L102)","stadium": "Hard Rock Stadium",       "city": "Miami",        "home_score": None, "away_score": None},

    # ══════════════════════ FINAL ══════════════════════
    {"match_no": 104, "date": "2026-07-19", "group": "Final", "matchday": None,
     "home": "TBD (W101)","away": "TBD (W102)","stadium": "MetLife Stadium",         "city": "New York/New Jersey", "home_score": None, "away_score": None},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_group_matches() -> list[dict]:
    """Return only group stage matches (A–L)."""
    return [m for m in SCHEDULE if len(m["group"]) == 1]

def get_matches_by_date(date_str: str) -> list[dict]:
    return [m for m in SCHEDULE if m["date"] == date_str]

# ── Kickoff times (UTC ISO strings) ──────────────────────────────────────────
# All times sourced from official FIFA/ESPN schedule. All US matches are EDT (UTC-4).
# To display in Vietnam (UTC+7): add 7h to UTC time.
KICKOFF_TIMES: dict[int, str] = {
    # MD1
    1:  "2026-06-11T18:00Z",  # Mexico vs SA         — 2PM ET
    2:  "2026-06-11T21:00Z",  # Korea vs Czech        — 5PM ET
    3:  "2026-06-12T19:00Z",  # Canada vs Bosnia      — 3PM ET
    4:  "2026-06-13T01:00Z",  # USA vs Paraguay       — 9PM ET
    5:  "2026-06-14T01:00Z",  # Haiti vs Scotland     — 9PM ET
    6:  "2026-06-14T04:00Z",  # Australia vs Turkey   — 12AM ET
    7:  "2026-06-13T22:00Z",  # Brazil vs Morocco     — 6PM ET
    8:  "2026-06-13T19:00Z",  # Qatar vs Switzerland  — 3PM ET
    9:  "2026-06-14T23:00Z",  # Ivory Coast vs Ecuador— 7PM ET
    10: "2026-06-14T17:00Z",  # Germany vs Curacao    — 1PM ET
    11: "2026-06-14T20:00Z",  # Netherlands vs Japan  — 4PM ET
    12: "2026-06-15T02:00Z",  # Sweden vs Tunisia     — 10PM ET
    13: "2026-06-15T22:00Z",  # Saudi Arabia vs Uruguay— 6PM ET
    14: "2026-06-15T16:00Z",  # Spain vs Cabo Verde   — 12PM ET
    15: "2026-06-16T01:00Z",  # Iran vs New Zealand   — 9PM ET
    16: "2026-06-15T19:00Z",  # Belgium vs Egypt      — 3PM ET
    17: "2026-06-16T19:00Z",  # France vs Senegal     — 3PM ET
    18: "2026-06-16T22:00Z",  # Iraq vs Norway        — 6PM ET
    19: "2026-06-17T01:00Z",  # Argentina vs Algeria  — 9PM ET
    20: "2026-06-17T04:00Z",  # Austria vs Jordan     — 12AM ET
    21: "2026-06-17T23:00Z",  # Ghana vs Panama       — 7PM ET
    22: "2026-06-17T20:00Z",  # England vs Croatia    — 4PM ET
    23: "2026-06-17T17:00Z",  # Portugal vs Congo DR  — 1PM ET
    24: "2026-06-18T02:00Z",  # Uzbekistan vs Colombia— 10PM ET
    # MD2
    25: "2026-06-18T16:00Z",  # Czech vs SA           — 12PM ET
    26: "2026-06-18T19:00Z",  # Swiss vs Bosnia       — 3PM ET
    27: "2026-06-18T22:00Z",  # Canada vs Qatar       — 6PM ET
    28: "2026-06-19T01:00Z",  # Mexico vs Korea       — 9PM ET
    29: "2026-06-20T00:30Z",  # Brazil vs Haiti       — 8:30PM ET
    30: "2026-06-19T22:00Z",  # Scotland vs Morocco   — 6PM ET
    31: "2026-06-20T03:00Z",  # Turkey vs Paraguay    — 11PM ET
    32: "2026-06-19T19:00Z",  # USA vs Australia      — 3PM ET
    33: "2026-06-20T20:00Z",  # Germany vs Ivory Coast— 4PM ET
    34: "2026-06-21T00:00Z",  # Ecuador vs Curacao    — 8PM ET
    35: "2026-06-20T17:00Z",  # Netherlands vs Sweden — 1PM ET
    36: "2026-06-21T04:00Z",  # Tunisia vs Japan      — 12AM ET
    37: "2026-06-21T22:00Z",  # Uruguay vs Cabo Verde — 6PM ET
    38: "2026-06-21T16:00Z",  # Spain vs Saudi Arabia — 12PM ET
    39: "2026-06-21T19:00Z",  # Belgium vs Iran       — 3PM ET
    40: "2026-06-22T01:00Z",  # New Zealand vs Egypt  — 9PM ET
    41: "2026-06-23T00:00Z",  # Norway vs Senegal     — 8PM ET
    42: "2026-06-22T21:00Z",  # France vs Iraq        — 5PM ET
    43: "2026-06-22T17:00Z",  # Argentina vs Austria  — 1PM ET
    44: "2026-06-23T03:00Z",  # Jordan vs Algeria     — 11PM ET
    45: "2026-06-23T20:00Z",  # England vs Ghana      — 4PM ET
    46: "2026-06-23T23:00Z",  # Panama vs Croatia     — 7PM ET
    47: "2026-06-23T17:00Z",  # Portugal vs Uzbekistan— 1PM ET
    48: "2026-06-24T02:00Z",  # Colombia vs Congo DR  — 10PM ET
    # MD3
    49: "2026-06-24T22:00Z",  # Scotland vs Brazil    — 6PM ET
    50: "2026-06-24T22:00Z",  # Morocco vs Haiti      — 6PM ET
    51: "2026-06-24T19:00Z",  # Swiss vs Canada       — 3PM ET
    52: "2026-06-24T19:00Z",  # Bosnia vs Qatar       — 3PM ET
    53: "2026-06-25T01:00Z",  # Czech vs Mexico       — 9PM ET
    54: "2026-06-25T01:00Z",  # SA vs Korea           — 9PM ET
    55: "2026-06-25T20:00Z",  # Curacao vs IC         — 4PM ET
    56: "2026-06-25T20:00Z",  # Ecuador vs Germany    — 4PM ET
    57: "2026-06-25T23:00Z",  # Japan vs Sweden       — 7PM ET
    58: "2026-06-25T23:00Z",  # Tunisia vs Netherlands— 7PM ET
    59: "2026-06-26T02:00Z",  # Turkey vs USA         — 10PM ET
    60: "2026-06-26T02:00Z",  # Paraguay vs Australia — 10PM ET
    61: "2026-06-26T19:00Z",  # Norway vs France      — 3PM ET
    62: "2026-06-26T19:00Z",  # Senegal vs Iraq       — 3PM ET
    63: "2026-06-27T03:00Z",  # Egypt vs Iran         — 11PM ET
    64: "2026-06-27T03:00Z",  # NZ vs Belgium         — 11PM ET
    65: "2026-06-27T00:00Z",  # Cabo Verde vs Saudi   — 8PM ET
    66: "2026-06-27T00:00Z",  # Uruguay vs Spain      — 8PM ET
    67: "2026-06-27T21:00Z",  # Panama vs England     — 5PM ET
    68: "2026-06-27T21:00Z",  # Croatia vs Ghana      — 5PM ET
    69: "2026-06-28T02:00Z",  # Algeria vs Austria    — 10PM ET
    70: "2026-06-28T02:00Z",  # Jordan vs Argentina   — 10PM ET
    71: "2026-06-27T23:30Z",  # Colombia vs Portugal  — 7:30PM ET
    72: "2026-06-27T23:30Z",  # Congo DR vs Uzbekistan— 7:30PM ET
}

def kickoff_vn(match_no: int) -> str | None:
    """Return kickoff time in Vietnam (UTC+7) as 'HH:MM' string, or None."""
    utc_str = KICKOFF_TIMES.get(match_no)
    if not utc_str:
        return None
    from datetime import datetime, timezone, timedelta
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    vn = dt + timedelta(hours=7)
    return vn.strftime("%H:%M")

def get_all_dates() -> list[str]:
    return sorted(set(m["date"] for m in SCHEDULE))

def is_played(m: dict) -> bool:
    """Returns True only if the match has a score AND enough time has passed
    since kickoff that we can be confident it is finished (not live)."""
    if m["home_score"] is None:
        return False
    # If we have a kickoff time, check at least 110 minutes have elapsed
    kickoff = KICKOFF_TIMES.get(m["match_no"])
    if kickoff:
        try:
            from datetime import datetime, timezone
            ko = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - ko).total_seconds() / 60
            if elapsed < 110:
                return False   # Still live — treat as upcoming
        except Exception:
            pass
    return True

def result_str(m: dict) -> str:
    if is_played(m):
        s = f"{m['home_score']}–{m['away_score']}"
        hp, ap = m.get("home_pens"), m.get("away_pens")
        if hp is not None and ap is not None:
            s += f" (pens {hp}–{ap})"
        return s
    return ""
