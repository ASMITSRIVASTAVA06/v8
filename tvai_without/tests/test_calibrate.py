import numpy as np

import calibrate as c


def make(n, seed, truth_prob, conf_range=(40, 95), urban_effect=False):
    """Synthetic reviewed alerts. truth_prob(conf, urban) is the real chance an alert is right."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        model_conf = int(rng.integers(*conf_range))
        urban = bool(rng.random() < 0.3)
        right = rng.random() < truth_prob(model_conf, urban)
        conf = max(5, model_conf - 35) if (urban and urban_effect) else model_conf
        out.append({"alert_id": f"TV-{i}", "category": "illegal_encroachment", "status": "verified" if right else "false positive",
                    "confidence": conf, "model_confidence": model_conf, "urban_light": urban,
                    "builtup_fraction": 0.6 if urban else 0.0, "in_protected_area": True,
                    "ndvi_change": -0.1, "tree_loss_fraction": 0.0, "fire_detected": False, "times_seen": 1,
                    "verdict": "Likely false positive" if urban else "Possible"})
    return out


def test_label_rules():
    assert c.is_correct({"status": "verified", "category": "a"}) == 1
    assert c.is_correct({"status": "false positive", "category": "a"}) == 0
    assert c.is_correct({"status": "verified", "category": "a", "true_category": "b"}) == 0   # analyst changed the type
    assert c.is_correct({"status": "under review", "category": "a"}) is None
    assert c.is_correct({"status": "new", "category": "a"}) is None


def test_calibrated_scores_have_small_error_and_overconfident_ones_large():
    good = c.analyse(make(1500, 1, lambda m, u: m / 100))
    bad = c.analyse(make(1500, 2, lambda m, u: max(0.02, m / 100 - 0.35)))
    assert good["ece_model"] < 0.06
    assert bad["ece_model"] > 0.25
    assert (bad["table_model"]["gap"] < 0).all()          # claimed more than reality in every band


def test_evidence_that_works_improves_scores_and_the_fit_sees_it():
    # Truth is driven by "urban" (urban light is rarely a real encroachment); model confidence is uninformative.
    r = c.analyse(make(900, 3, lambda m, u: 0.08 if u else 0.8, urban_effect=True))
    assert r["brier_adjusted"] < r["brier_model"]
    assert r["fit"]["ok"] and r["fit"]["auc_with_evidence"] > r["fit"]["auc_model_only"] + 0.1
    assert r["fit"]["coefficients"]["urban"] < 0


def test_small_or_biased_samples_are_flagged():
    small = c.analyse(make(20, 4, lambda m, u: 0.5))
    assert any("Only 20" in w for w in small["warnings"]) and not small["fit"]["ok"]
    biased = c.analyse(make(200, 5, lambda m, u: 0.9, conf_range=(85, 95)))
    assert any("biased" in w for w in biased["warnings"])


def test_report_renders_for_normal_and_empty_input():
    text = c.format_report(c.analyse(make(300, 6, lambda m, u: m / 100)))
    assert "Brier" in text and "Reliability" in text
    assert "Nothing to analyse" in c.format_report(c.analyse([]))
    assert "Nothing to analyse" in c.format_report(c.analyse([{"status": "new", "category": "x"}]))
