# TerraVision AI — Forest Intelligence from Space

TerraVision watches forests from orbit. It spots new night-time light inside forests (camps, roads,
settlements, burning), checks every detection against independent satellites, lets people review and
share what it finds, looks up fire and weather danger for any place on Earth, and forecasts where
thunderstorms, heavy rain and cyclones may hit your forest regions.

Everything runs on **real data from live services**. There is no sample or demo data anywhere.

*EcoSentinel — YESIST12 Innovation Challenge 2026*

> **Read this first.** TerraVision is a research and awareness tool. Night-light change *suggests* a
> problem, it does not prove one, and weather outputs are model indications, not official warnings.
> Verify alerts on the ground, and follow your national weather service (IMD, NHC, JTWC) for any
> decision that affects safety. See [Limits and honesty](#12-limits-and-honesty).

---

## Contents

1. [What you can do with it](#1-what-you-can-do-with-it)
2. [What you need](#2-what-you-need)
3. [Install and first run](#3-install-and-first-run)
4. [Add your API keys](#4-add-your-api-keys)
5. [Your first scan, step by step](#5-your-first-scan-step-by-step)
6. [The app, section by section](#6-the-app-section-by-section)
7. [How the detection and evidence work](#7-how-the-detection-and-evidence-work)
8. [Train the classifier on real data](#8-train-the-classifier-on-real-data)
9. [Region boundaries](#9-region-boundaries)
10. [Automatic scans and weather alerts](#10-automatic-scans-and-weather-alerts)
11. [Hosting it online](#11-hosting-it-online)
12. [Limits and honesty](#12-limits-and-honesty)
13. [Troubleshooting](#13-troubleshooting)
14. [Testing](#14-testing)
15. [Project map](#15-project-map)
16. [Data sources and credits](#16-data-sources-and-credits)

---

## 1. What you can do with it

| Section in the app | What it does |
|---|---|
| **Monitor** | Scan a forest region. Clusters of new night light appear on satellite imagery, each classified (illegal encroachment, wildfire, agricultural burn, anomaly) and judged against independent evidence. |
| **Dashboard** | Charts across your scans: what changed, where, how reliable it is, how the classifier's confidence shifted. |
| **Alerts** | Browse every alert, filter it, record a review (verified, false positive, assigned, noted), see its exact location, then download or share it by e-mail, Telegram or WhatsApp. |
| **Search a place** | Type a place name or coordinates. The map jumps there and says whether NASA thermal satellites see fire nearby, whether night lights look unusual, or "No anomaly detected". |
| **Weather risk** | Forecast thunderstorms, heavy rain, damaging wind, extreme heat and tropical cyclones. Risky zones are flagged on the map, and alerts can be sent. |

Behind the app there are scripts for scheduled scans, scheduled weather alerts, training the classifier
on real data, and checking that your API keys work.

---

## 2. What you need

- **Python 3.10 or newer** (developed on 3.13).
- **A Google account with Earth Engine access.** Earth Engine is free for research and non-commercial use
  but needs a one-time sign-up at https://earthengine.google.com/ and a Google Cloud project. Note the
  project name: it is your `GEE_PROJECT_ID`.
- **A free NASA FIRMS key** for real fire detections (see step 4).
- An internet connection. The first scan downloads satellite data and takes a few minutes.
- Optional: an e-mail account (SMTP) or a Telegram bot if you want alerts delivered to people.

---

## 3. Install and first run

**Windows (PowerShell)** — the folder is the one containing `app.py`:

```powershell
cd C:\path\to\tvai_without
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
earthengine authenticate
copy .env.example .env
```

If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**macOS / Linux:**

```bash
cd /path/to/tvai_without
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
earthengine authenticate
cp .env.example .env
```

`earthengine authenticate` opens a browser; sign in with the Google account that has Earth Engine
access. You do this once per computer.

Then open `.env` in a text editor and fill in your keys (next section), and start the app:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. Stop it with `Ctrl+C` in the terminal.

---

## 4. Add your API keys

Keys live in the file named **`.env`** in the project folder (not `.env.example`, which is only a
template the app never reads). Each line is `NAME=value`: no quotes, no spaces around `=`. A line
starting with `#` is a comment and is ignored, so remove the `#` from any line you fill in.

### What to fill in

| Service | Lines in `.env` | Needed? | What it gives you | Where to get it |
|---|---|---|---|---|
| Google Earth Engine | `GEE_PROJECT_ID=your-project` | **Yes** | Night lights, protected areas, land cover, forest loss, vegetation | earthengine.google.com |
| NASA FIRMS | `FIRMS_MAP_KEY=your-key` | **Strongly recommended** | Real fire hotspots within hours of each satellite pass, used by place search | firms.modaps.eosdis.nasa.gov/api/map_key (free) |
| Geocoder | `GEOCODER_URL=...` and `GEOCODER_API_KEY=...` | Recommended once several people use search | Reliable place search | e.g. locationiq.com (free tier) |
| Open-Meteo | `OPENMETEO_API_KEY=...` | Only for commercial use | Weather forecast. Works with no key for non-commercial use | open-meteo.com/en/pricing |
| E-mail | `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD` (+ `SMTP_PORT`, `SMTP_FROM`) | Only to send e-mail | Alert e-mails | Your mail provider |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Only to send Telegram | Alert messages to a group | @BotFather on Telegram |
| GDACS cyclones | nothing | No key | Active cyclones and forecast tracks | gdacs.org |

A typical starting `.env`:

```
GEE_PROJECT_ID=my-earth-engine-project
FIRMS_MAP_KEY=your-nasa-key
GEOCODER_URL=https://us1.locationiq.com/v1/search
GEOCODER_API_KEY=your-locationiq-key
```

**The geocoder needs both lines.** The key is only ever sent to the address in `GEOCODER_URL`. If you
set a key without an address, searches use the free OpenStreetMap service and your key is not attached.
For LocationIQ, `GEOCODER_URL` is `https://us1.locationiq.com/v1/search` (use the region host shown in
your account if it differs). Do not put the key inside the URL line.

**Gmail:** use an *App Password* (Google Account → Security → 2-Step Verification → App passwords), with
`SMTP_HOST=smtp.gmail.com` and `SMTP_PORT=587`.

### Check that everything works

```bash
python check_apis.py
```

It makes one tiny request per service and prints `OK`, `FAILED` with a plain reason, or `SKIPPED`
(no key set, which is fine for optional services). It never prints a key. The app's sidebar also has a
**Data sources** panel: green means a key is set, yellow means a free service is used, red means a key
is missing or incomplete.

### Keep your keys safe

- Keys are read only from the environment or Streamlit secrets, never logged, and removed from error
  messages before they reach the screen.
- `.env` is already in `.gitignore`. **Never upload it to GitHub or include it in a zip you share.**
- If a key ever appears somewhere it should not, regenerate it in the provider's dashboard.

---

## 5. Your first scan, step by step

1. Start the app (`streamlit run app.py`).
2. In the sidebar, open **Monitor** and choose a **Forest Region**.
3. Leave the settings at their defaults for the first run: threshold 5, minimum cluster 3,
   "Correct for regional brightening" on, "Cross-check with other satellites" on.
4. Press **Run Satellite Scan**. The first run downloads a five-year baseline and the latest monthly
   night-light data (a few minutes). Later scans of the same month use the cache and are much faster.
5. Read the map. Coloured circles are alerts; white-ringed ones have strong supporting evidence.
   Click a circle to see its coordinates, radiance, confidence and the evidence behind it.
6. Open **Alerts** to review them: tick rows, set a status (in review, verified, false positive),
   assign someone, add a note. Decisions are saved and survive new scans.
7. Open **Dashboard** for the overall picture across your scans.

---

## 6. The app, section by section

### Monitor
The search box (the dashed rectangle) is the area the satellite data is downloaded for. The coloured
polygons are the region's real boundary (see [section 9](#9-region-boundaries)). Sidebar controls:

- **Alert threshold** — how much brighter than normal a pixel must be (higher = fewer, surer alerts).
- **Min cluster size** — ignores lone pixels, which are usually sensor noise.
- **Correct for regional brightening** — removes the district-wide brightening caused by village
  electrification so only patches that stand out are flagged.
- **Cross-check with other satellites** — adds the evidence layers (slower, needs Earth Engine).
- **Hide urban lights** and **Forest & protected areas only** — hide alerts that are towns or lie
  outside the region's real boundary.
- **Force fresh satellite download** — ignore the cache.

The **Scan history & health** panel shows past runs and which cross-check layers worked.

### Dashboard
Overview, evidence and tracking, distributions, a map, and classifier insight (how the evidence moved
the model's confidence). Filters let you narrow by type, severity, verdict and status.

### Alerts
Filter, sort and search every alert; select rows to review them; zoom to any alert on satellite imagery
with its exact 500 m pixel outlined; press **Show before / after satellite imagery** to see Sentinel-2
imagery a year apart. Export as CSV, JSON or a printable HTML report; e-mail it, post it to Telegram, or
share by WhatsApp. The **Training labels** tab exports reviewed alerts for model training.

### Search a place
Type a place ("Sariska Tiger Reserve") or coordinates (`27.2046, 77.4977`, `27.2046 N 77.4977 E`, or
degrees-minutes-seconds). Choose a radius and fire window. You get one of four answers:

| Answer | Meaning |
|---|---|
| **Fire detected** | Thermal hotspots found; they are drawn on the map with distance, temperature and time. |
| **Night-light anomaly** | No fire, but night lights rose clearly above the 2019–2023 normal (new buildings, a camp, industry). |
| **No anomaly detected** | Both checks ran and found nothing unusual. |
| **Could not check** | A data source failed. This is **not** an all-clear. |

If only one of the two checks could run, the result says so.

### Weather risk
Four tabs:

- **Risk map** — pick a forest region (or search a place), a forecast window (24–120 h) and a detail
  level. Cells are coloured Low, Watch, Warning or Severe; a table lists flagged zones with peak times
  and can be downloaded.
- **All forest regions** — one coarse check of every region you monitor.
- **Cyclones** — active storms with observed and forecast tracks, gale-force wind fields, and which of
  your regions are exposed.
- **Send alert** — e-mail and/or Telegram the current watch to your team.

How hazards are scored (all from the hourly forecast): heavy rain follows India Meteorological Department
24 h classes (64.5, 115.6, 204.5 mm); gusts of 60, 90 and 118 km/h mark Watch, Warning and Severe;
heat thresholds are 40, 43 and 46 °C; storm energy alone only raises a Watch, a thunderstorm must also
appear in the forecast for a Warning. Cyclone risk comes from distance to the forecast track and
whether a zone lies inside the forecast gale-force wind field.

A zone whose forecast could not be fetched is shown as **unknown**, never as safe, and a cyclone feed
outage is reported instead of "no cyclones".

---

## 7. How the detection and evidence work

1. **Detect.** The latest two monthly night-light composites (NASA VIIRS, 500 m pixels) are compared with
   a five-year median (2019–2023). Pixels with fewer than two cloud-free observations are ignored. The
   regional brightening trend is removed, then connected bright pixels become clusters.
2. **Classify.** A Random Forest looks at each cluster's brightness and shape (size, compactness,
   convexity and so on) and labels it: *natural wildfire*, *agricultural burn*, *illegal encroachment*
   or *anomaly*.
3. **Cross-check.** Independent satellites are asked about every cluster: is it inside a protected area
   (WDPA)? Is it a town (ESA WorldCover built-up)? Did vegetation drop (Sentinel-2 NDVI)? Was forest lost
   (Hansen)? Is there fire (NASA FIRMS)? Has it appeared in several scans?
4. **Score.** The evidence adjusts the model's confidence by transparent rules (listed in `evidence.py`)
   and gives a verdict: **Strong evidence**, **Possible** or **Likely false positive**, with the reasons.
   The model's own confidence is always kept alongside.
5. **Zone.** Each alert is placed *official*, *protected*, *forest* or *outside* relative to the
   region's real boundary. Outside alerts lose confidence and are hidden by default.
6. **Track.** Each scan is compared with the previous one: alerts are *new*, *recurring* or
   *resolved*.

**Calibration.** Once you have reviewed alerts, run `python calibrate.py`. It reports whether "80 %"
really means about 80 %, and suggests whether the evidence weights should change. Review alerts across
*all* confidence levels, not just the top ones, or the numbers will look better than reality.

---

## 8. Train the classifier on real data

The classifier ships trained on generated examples, which is only a starting point. The sidebar says which
kind of model is active. To train on real labelled data:

```bash
python training_data.py --regions all --windows 6     # needs Earth Engine; resumable; takes a while
python train_real.py                                  # trains, validates, saves if good enough
python train_real.py --db terravision.db              # also learn from alerts you reviewed
```

How it earns trust:

- **Labels come from satellites the classifier never sees** (fire → wildfire or agricultural burn by
  surrounding land cover; built-up → anomaly; forest loss or vegetation drop inside forest →
  encroachment). Ambiguous clusters are dropped, not guessed.
- **Spatial validation:** clusters from the same region and 1-degree map cell never appear in both
  training and test folds, so the score reflects places the model has not seen.
- It compares a Random Forest with gradient boosting, calibrates the probabilities, and scores the old
  model on the same real labels.
- **Quality gate:** the new model is saved only if it clears a minimum score and beats the current model.
  `--force` overrides this. The previous synthetic model is backed up once.
- `model_card.json` records what the model was trained on and how well it did.

These labels are weak supervision, not ground truth. Reviewing real alerts in the app is what improves
them; your verdicts are merged in at double weight.

---

## 9. Region boundaries

Each region's real footprint is, in order of trust:

1. **An official boundary file you provide.** Put a GeoJSON named after the region in `boundaries/`, for
   example `boundaries/aravalli_hills_india.geojson` (from your forest department, Survey of India, FSI,
   Protected Planet or OpenStreetMap). The file name is the region name lower-cased with punctuation
   replaced by underscores (see `regions.region_slug`). It is authoritative.
2. **Designated protected areas from WDPA,** limited to the region's countries and search box, fetched
   automatically and cached in `data_cache/boundaries/` for 90 days.

Alerts are tested against these polygons locally, with holes and multipolygons respected. Add or edit
regions in `regions.py` (search box and country codes).

---

## 10. Automatic scans and weather alerts

### Forest scan with a digest

```bash
python scheduled_scan.py --regions "Aravalli Hills, India"                 # one region
python scheduled_scan.py --regions all --min-confidence 65 --telegram      # all regions
python scheduled_scan.py --regions "Borneo Rainforest" --dry-run           # print, send nothing
```

Separate several regions with `;` because names contain commas. A digest is sent only when something
**new** and credible turns up. Recipients come from `--to` or `DIGEST_RECIPIENTS` in `.env`.

### Weather and cyclone watch

```bash
python scheduled_weather.py --regions all --min-level 2 --dry-run
python scheduled_weather.py --regions all --min-level 2 --telegram
```

Levels: 1 Watch, 2 Warning, 3 Severe. It alerts only when a region reaches the level **and** the
situation changed since the last alert (a Severe situation is repeated after 24 h). The record of what
was sent is kept in `data_cache/weather_state.json`.

### Running them on a schedule

- **cron (always-on machine):**
  `0 6 * * MON cd /path/to/tvai_without && .venv/bin/python scheduled_scan.py --regions all --telegram`
  `15 */6 * * * cd /path/to/tvai_without && .venv/bin/python scheduled_weather.py --regions all --telegram`
- **Windows Task Scheduler:** create a task that runs `.venv\Scripts\python.exe scheduled_scan.py ...`
  from the project folder.
- **GitHub Actions:** `.github/workflows/scheduled_scan.yml` (weekly) and `scheduled_weather.yml`
  (every 6 hours). Add the keys as repository secrets (never commit `.env`):
  `GEE_PROJECT_ID`, `GEE_SERVICE_ACCOUNT_JSON`, `SMTP_*`, `DIGEST_RECIPIENTS`, `TELEGRAM_BOT_TOKEN`,
  `TELEGRAM_CHAT_ID`. The weather job needs no Earth Engine access.

A headless Earth Engine login needs a **service account**: create one in Google Cloud, grant it Earth
Engine access, download its JSON key, and paste the whole JSON as one line in
`GEE_SERVICE_ACCOUNT_JSON`.

GitHub runners are wiped after every run, so the database is kept in the Actions cache, which expires
if unused for 7 days. For a permanent archive, run on an always-on machine and point `TERRAVISION_DB`
at a persistent disk.

---

## 11. Hosting it online

To share the app with a team, deploy it to Streamlit Community Cloud or any server:

1. Push the project to GitHub (without `.env`, `terravision.db` or `data_cache/`; `.gitignore` covers
   them).
2. On Streamlit Cloud, create an app from the repository with `app.py` as the entry point.
3. In the app's **Secrets** settings, add the same names as `.env`, in TOML form:

```toml
GEE_PROJECT_ID = "my-earth-engine-project"
GEE_SERVICE_ACCOUNT_JSON = '{"type": "service_account", ...}'
FIRMS_MAP_KEY = "your-nasa-key"
GEOCODER_URL = "https://us1.locationiq.com/v1/search"
GEOCODER_API_KEY = "your-locationiq-key"
```

Hosted apps lose local files when they restart, so set `TERRAVISION_DB` to a persistent location if
you need reviews and scan history to last. Use the paid Open-Meteo plan and a geocoder key for shared
or commercial deployments.

---

## 12. Limits and honesty

- **Resolution.** VIIRS pixels are about 500 m. A pin marks the centre of a glowing area, not a building.
  Established tools built on Sentinel imagery see smaller changes; TerraVision complements them.
- **It suggests, it does not prove.** Every alert needs a ground check or imagery review.
- **Labels are weak.** The real-data classifier learns from other satellites' signals, which carry their
  own errors. Accuracy claims are earned only after you compare alerts with field findings.
- **Fire detection misses things.** Satellites miss small, brief or cloud-covered fires. "No anomaly
  detected" means nothing was detected, not that a place is certainly safe.
- **Weather is a model forecast,** not an official warning. Cyclone tracks come from public advisories
  republished by GDACS.
- **Services change.** Earth Engine dataset IDs, free-tier limits and API formats can change. Layers
  degrade gracefully and say so; a failed check is reported as unknown.
- **Free tiers have limits.** Open-Meteo's free tier is non-commercial (about 10,000 calls a day);
  OpenStreetMap's free search is for light use; FIRMS allows a generous but finite request rate.

---

## 13. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `earthengine: command not found` | The virtual environment is not active. Activate it, then run `earthengine authenticate` again. |
| "Could not complete the scan" with an authentication or permission error | Run `earthengine authenticate`, check `GEE_PROJECT_ID`, and confirm the project has Earth Engine enabled. |
| Sidebar shows 🔴 for a service | The key is missing or incomplete. For the geocoder, both `GEOCODER_URL` and `GEOCODER_API_KEY` are needed. |
| My new key has no effect | You edited `.env.example` instead of `.env`, left a `#` at the start of the line, added quotes or spaces, or did not restart the app. Stop it with Ctrl+C and run `streamlit run app.py` again. |
| `check_apis.py` says FAILED for FIRMS: "Invalid MAP_KEY" | Copy the key again from the e-mail NASA sent, with no spaces. |
| Place search says it could not be reached | Check your internet connection. Typing coordinates still works without any service. |
| Browser shows `Invalid key` at the geocoder address | You opened the address without `?key=…&q=…`. That is expected; the app adds them. |
| `InconsistentVersionWarning` from scikit-learn | The saved classifier was built with another scikit-learn version. The app retrains it once automatically (from real data if you built it, otherwise synthetic). |
| Earth Engine "asset not found" for forest loss | Hansen datasets are renamed each year. The app tries the newest first; update `HANSEN_IDS` in `enrichment.py` if all fail. |
| `pip install rasterio` fails on Windows | Upgrade pip (`python -m pip install -U pip`) and use a supported Python version; rasterio ships Windows wheels for current Python releases. |
| Map is empty after a scan | Your filters may hide everything. Untick **Hide urban lights** and **Forest & protected areas only**, and tick **Unclassified anomaly**. |
| Weather page says "forecast service returned nothing" | The service refused or timed out (rate limit, network). Wait a minute and retry. The page reports missing zones as unknown. |
| Cyclone tab shows a note about the feed | GDACS was unreachable or changed format. The rest of the page still works. If it persists, report the exact message. |
| Something else | Look at `logs/terravision.log` for the full error. |

---

## 14. Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite (about 100 tests) covers detection and clustering, evidence scoring, boundaries and zones,
tracking, the database, calibration, model training, place search, fire checks, weather scoring,
cyclone parsing, alert de-duplication, key handling and redaction. Network calls are replaced with
recorded-style responses, so the tests run offline. They do **not** call the live services: run
`python check_apis.py` for that.

---

## 15. Project map

**The app:** `app.py` (navigation, Monitor) · `dashboard.py` · `alerts_view.py` · `search_view.py` ·
`weather_view.py` · `locator.py` (exact-location map) · `theme.py` (colours and styling).

**Detection:** `pipeline.py` (data download, scan orchestration) · `detect.py` (clusters and classifier) ·
`model.py` (classifier and synthetic training) · `regional.py` (brightening correction) ·
`enrichment.py` (cross-check layers) · `evidence.py` (scoring and zones) · `tracking.py` (new / recurring /
resolved) · `boundaries.py` · `regions.py` · `imagery.py` (before/after imagery).

**Search and weather:** `geocode.py` · `firms.py` · `placecheck.py` · `weather.py` · `cyclones.py` ·
`riskzones.py` · `weather_alerts.py`.

**Data and operations:** `store.py` (database: scans, reviews, history) · `apis.py` (keys and status) ·
`check_apis.py` · `reliability.py` (retries and logging) · `alert_tools.py` (exports, e-mail, Telegram) ·
`calibrate.py` · `modelcard.py`.

**Training:** `training_data.py` · `train_real.py`.

**Scheduled jobs:** `scheduled_scan.py` · `scheduled_weather.py` · `.github/workflows/`.

**Other:** `gee_export.py` (optional manual raster export) · `test_connection.py` (checks Earth Engine
login) · `dynamics.py` (building blocks for planned interface features such as staged scan progress,
sparklines and scan replay; not yet connected to the app) · `tests/`.

**Files the app creates:** `terravision.db` (scans and reviews) · `data_cache/` (downloaded satellite
data, boundaries) · `logs/terravision.log` · `training/` (real training data) ·
`terravision_rf.joblib` and `model_card.json` (the classifier) · `boundaries/` (your official boundary
files). Back up `terravision.db` and `boundaries/`; the rest can be rebuilt.

---

## 16. Data sources and credits

NASA VIIRS Day/Night Band monthly composites (NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG) · NASA FIRMS active
fire · World Database on Protected Areas (UNEP-WCMC and IUCN) · ESA WorldCover · Copernicus Sentinel-2 ·
Hansen / UMD Global Forest Change · Google Earth Engine · Open-Meteo weather forecasts (CC BY 4.0,
attribution required for commercial use) · GDACS tropical cyclone data (NOAA and JTWC advisories) ·
OpenStreetMap contributors (ODbL) via Nominatim or LocationIQ · Esri World Imagery tiles for maps.
