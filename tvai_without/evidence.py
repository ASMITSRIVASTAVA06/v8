"""
TerraVision AI — evidence scoring.

Night-light change alone only *suggests* a problem. This module combines the
classifier's confidence with independent evidence gathered for each alert
(protected-area membership, built-up land cover, vegetation change, forest loss,
active fire, persistence over time) into one transparent, adjustable score.

Every rule is listed in RULES below so the logic can be reviewed and tuned; the
original model confidence is always kept as `model_confidence`.
Pure Python — no Earth Engine or Streamlit needed.
"""

from __future__ import annotations

from typing import Dict, List

URBAN_STRONG = 0.30    # >= 30 % built-up inside the pixel: almost surely a town
URBAN_WEAK = 0.15
NDVI_DROP_STRONG = -0.15
NDVI_DROP_WEAK = -0.08
LOSS_STRONG = 0.10     # share of the pixel with tree loss in the last two years
LOSS_WEAK = 0.02
FOREST_CONTEXT = 0.30  # share of tree cover within ~1 km that makes unprotected land still "forest"

ZONE_OFFICIAL = "official"      # inside the region's official boundary file
ZONE_PROTECTED = "protected"    # inside a designated protected area (WDPA)
ZONE_FOREST = "forest"          # unprotected, but surrounded by tree cover
ZONE_OUTSIDE = "outside"        # outside the boundary / not forest: probably not this region's forest

RULES = [
    ("Built-up land ≥ 30 % of the pixel", -35),
    ("Built-up land 15–30 %", -15),
    ("Inside a protected area", +8),
    ("Outside every protected-area polygon", -10),
    ("Vegetation (NDVI) dropped ≥ 0.15", +12),
    ("Vegetation (NDVI) dropped 0.08–0.15", +6),
    ("Forest loss ≥ 10 % of the pixel", +12),
    ("Forest loss 2–10 % of the pixel", +6),
    ("Active fire detected nearby (encroachment alerts)", -15),
    ("Active fire detected nearby (fire / burn alerts)", +8),
    ("Seen in ≥ 3 satellite periods (encroachment)", +8),
    ("Seen in 2 satellite periods (encroachment)", +4),
]

VERDICT_STRONG = "Strong evidence"
VERDICT_POSSIBLE = "Possible"
VERDICT_UNLIKELY = "Likely false positive"
VERDICTS = [VERDICT_STRONG, VERDICT_POSSIBLE, VERDICT_UNLIKELY]


def zone_of(alert: Dict):
    """
    Where the alert sits relative to the region's real footprint, or None if nothing is known.
    An official boundary, when one was loaded, is authoritative; otherwise protected-area
    membership, otherwise tree cover around the pixel.
    """
    official = alert.get("in_official_boundary")
    if official is True:
        return ZONE_OFFICIAL
    if official is False:
        return ZONE_OUTSIDE
    in_pa = alert.get("in_protected_area")
    if in_pa is True:
        return ZONE_PROTECTED
    tree = alert.get("tree_context")
    if tree is not None and tree >= FOREST_CONTEXT:
        return ZONE_FOREST
    if in_pa is False:
        return ZONE_OUTSIDE
    return None


def score_evidence(alert: Dict) -> Dict:
    """
    Return a dict of fields to merge into the alert:
    confidence (adjusted), model_confidence, evidence (list of short reasons),
    corroboration (count of independent confirming signals), urban_light, verdict, zone.
    """
    cat = alert.get("category")
    model_conf = int(alert.get("model_confidence", alert.get("confidence", 0)))
    delta = 0
    reasons: List[str] = []
    positives = 0
    negatives = 0

    builtup = alert.get("builtup_fraction")
    urban = False
    if builtup is not None:
        if builtup >= URBAN_STRONG:
            delta -= 35; negatives += 1; urban = True
            reasons.append(f"Urban light: {builtup:.0%} built-up land")
        elif builtup >= URBAN_WEAK:
            delta -= 15; negatives += 1
            reasons.append(f"Partly built-up ({builtup:.0%})")

    zone = zone_of(alert)
    if zone == ZONE_OFFICIAL:
        delta += 8; positives += 1
        reasons.append(f"Inside {alert.get('boundary_name') or 'the official forest boundary'}")
    elif zone == ZONE_PROTECTED:
        delta += 8; positives += 1
        reasons.append(f"Inside {alert.get('protected_area') or 'a protected area'}")
    elif zone == ZONE_FOREST:
        reasons.append("Forested land outside designated areas")
    elif zone == ZONE_OUTSIDE:
        delta -= 10
        reasons.append("Outside the forest boundary" if alert.get("in_official_boundary") is False
                       else "Outside protected-area polygons")

    ndvi = alert.get("ndvi_change")
    if ndvi is not None:
        if ndvi <= NDVI_DROP_STRONG:
            delta += 12; positives += 1; reasons.append(f"Vegetation loss (NDVI {ndvi:+.2f})")
        elif ndvi <= NDVI_DROP_WEAK:
            delta += 6; positives += 1; reasons.append(f"Vegetation decline (NDVI {ndvi:+.2f})")

    loss = alert.get("tree_loss_fraction")
    if loss is not None:
        if loss >= LOSS_STRONG:
            delta += 12; positives += 1; reasons.append(f"Forest loss: {loss:.0%} of pixel")
        elif loss >= LOSS_WEAK:
            delta += 6; positives += 1; reasons.append(f"Some forest loss: {loss:.0%} of pixel")

    fire = alert.get("fire_detected")
    if fire:
        if cat == "illegal_encroachment":
            delta -= 15; negatives += 1
            reasons.append("Active fire nearby: may be a burn, not a settlement")
        elif cat in ("natural_wildfire", "agricultural_burn"):
            delta += 8; positives += 1
            reasons.append("Active fire confirmed by thermal satellite")

    seen = int(alert.get("times_seen", 1) or 1)
    if cat == "illegal_encroachment":
        if seen >= 3:
            delta += 8; positives += 1; reasons.append(f"Persistent: seen in {seen} periods")
        elif seen == 2:
            delta += 4; positives += 1; reasons.append("Seen again in a second period")

    adjusted = max(5, min(99, model_conf + delta))

    if urban or (negatives >= 2 and positives == 0):
        verdict = VERDICT_UNLIKELY
    elif positives >= 2 and negatives == 0:
        verdict = VERDICT_STRONG
    else:
        verdict = VERDICT_POSSIBLE

    return {
        "model_confidence": model_conf,
        "confidence": adjusted,
        "evidence": reasons,
        "corroboration": positives,
        "urban_light": urban,
        "verdict": verdict,
        "zone": zone,
    }


def apply_evidence(alerts: List[Dict]) -> List[Dict]:
    """Score every alert in place and return the list."""
    for a in alerts:
        a.update(score_evidence(a))
    return alerts
