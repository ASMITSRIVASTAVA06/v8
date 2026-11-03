from evidence import (VERDICT_POSSIBLE, VERDICT_STRONG, VERDICT_UNLIKELY, apply_evidence, score_evidence)


def base(**kw):
    return {"category": "illegal_encroachment", "confidence": 70, **kw}


def test_no_context_changes_nothing():
    r = score_evidence(base())
    assert r["confidence"] == 70 and r["model_confidence"] == 70
    assert r["verdict"] == VERDICT_POSSIBLE and r["corroboration"] == 0


def test_urban_light_is_penalised_and_flagged():
    r = score_evidence(base(builtup_fraction=0.6, in_protected_area=False))
    assert r["urban_light"] is True
    assert r["verdict"] == VERDICT_UNLIKELY
    assert r["confidence"] == 70 - 35 - 10


def test_corroborated_encroachment_is_strong():
    r = score_evidence(base(in_protected_area=True, protected_area="Sariska", ndvi_change=-0.2,
                            tree_loss_fraction=0.2, times_seen=3))
    assert r["verdict"] == VERDICT_STRONG
    assert r["corroboration"] == 4
    assert r["confidence"] == 99                      # clamped
    assert any("Sariska" in e for e in r["evidence"])


def test_fire_nearby_cuts_encroachment_but_boosts_fire_alerts():
    enc = score_evidence(base(fire_detected=True))
    fire = score_evidence({"category": "natural_wildfire", "confidence": 70, "fire_detected": True})
    assert enc["confidence"] < 70 < fire["confidence"]


def test_model_confidence_is_preserved_on_rescoring():
    a = base(builtup_fraction=0.2)
    apply_evidence([a]); first = a["confidence"]
    apply_evidence([a])
    assert a["model_confidence"] == 70 and a["confidence"] == first


def test_confidence_stays_in_bounds():
    low = score_evidence({"category": "illegal_encroachment", "confidence": 10, "builtup_fraction": 0.9})
    assert low["confidence"] == 5
