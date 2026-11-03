from datetime import date

import numpy as np

import modelcard
import train_real
from training_data import label_cluster, training_windows

CLASSES = train_real.CLASS_NAMES


def test_label_rules():
    assert label_cluster({"builtup_fraction": 0.5, "fire_detected": False}) == ("anomaly", 1.0)
    assert label_cluster({"builtup_fraction": 0.0, "fire_detected": True, "tree_context": 0.7, "crop_context": 0.0})[0] == "natural_wildfire"
    assert label_cluster({"builtup_fraction": 0.0, "fire_detected": True, "tree_context": 0.1, "crop_context": 0.6})[0] == "agricultural_burn"
    assert label_cluster({"builtup_fraction": 0.0, "fire_detected": True, "tree_context": 0.4, "crop_context": 0.4})[0] is None
    enc = {"builtup_fraction": 0.02, "fire_detected": False, "tree_context": 0.7, "tree_loss_fraction": 0.1,
           "in_protected_area": True}
    assert label_cluster(enc)[0] == "illegal_encroachment"
    assert label_cluster({**enc, "tree_loss_fraction": 0.0, "ndvi_change": 0.0})[0] is None   # nothing cleared
    assert label_cluster({**enc, "tree_context": 0.0, "in_protected_area": False})[0] is None
    assert label_cluster({"fire_detected": False})[0] is None                                   # missing signal: dropped


def test_windows_start_after_baseline_and_do_not_overlap():
    w = training_windows(date(2026, 10, 24), 50)
    assert w[0][0] >= date(2024, 1, 1)
    assert all(a[1] == b[0] for a, b in zip(w, w[1:]))
    assert w[-1][1] == date(2026, 10, 1)                       # current month excluded


def fake_dataset(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    X = rng.normal(size=(n, len(train_real.FEATURE_NAMES))) + y[:, None] * 1.5       # separable by class
    groups = np.array([f"r|{i % 12}|0" for i in range(n)])
    return {"X": X.astype(np.float32), "y": y.astype(np.int32), "w": np.ones(n), "groups": groups}


def test_check_enough():
    d = fake_dataset()
    assert train_real.check_enough(d) is None
    small = {k: v[:50] for k, v in d.items()}
    assert "need" in train_real.check_enough(small)
    one_place = {**d, "groups": np.array(["a|0|0"] * len(d["y"]))}
    assert "places" in train_real.check_enough(one_place)


def test_spatial_folds_never_split_a_place():
    d = fake_dataset()
    for tr, te in train_real.spatial_folds(d["groups"], d["y"]):
        assert not set(d["groups"][tr]) & set(d["groups"][te])


def test_train_saves_model_and_card(tmp_path, monkeypatch):
    monkeypatch.setattr(train_real, "SYNTHETIC_BACKUP", tmp_path / "syn.joblib")
    card = tmp_path / "card.json"
    monkeypatch.setattr(modelcard, "CARD_PATH", card)
    monkeypatch.setattr(modelcard, "write", lambda c, path=card, _w=modelcard.write: _w(c, path))
    d = fake_dataset()
    model, report = train_real.train_from_data(d, model_path=tmp_path / "m.joblib", log=lambda *_: None)
    assert model is not None and report["accepted"]
    proba = model.predict_proba(d["X"][:5])
    assert proba.shape == (5, 4) and np.allclose(proba.sum(axis=1), 1)
    assert list(model.classes_) == [0, 1, 2, 3]
    saved = modelcard.read(card)
    assert saved["provenance"] == "real" and saved["metrics"]["macro_f1"] > 0.55


def test_gate_rejects_noise_and_force_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(train_real, "SYNTHETIC_BACKUP", tmp_path / "syn.joblib")
    monkeypatch.setattr(modelcard, "CARD_PATH", tmp_path / "card.json")
    d = fake_dataset()
    d["y"] = np.random.default_rng(1).integers(0, 4, len(d["y"])).astype(np.int32)   # labels unrelated to features
    model, report = train_real.train_from_data(d, model_path=tmp_path / "m.joblib", log=lambda *_: None)
    assert model is None and "gate" in report["reason"]
    assert not (tmp_path / "m.joblib").exists()
    model, _ = train_real.train_from_data(d, force=True, model_path=tmp_path / "m.joblib", log=lambda *_: None)
    assert model is not None


def test_analyst_rows_use_true_category_and_verified():
    shape = dict(aspect_ratio=1, compactness=.5, convexity=.5, edge_ratio=.5, spread_rate=.5, interior_uniformity=.5)
    base = dict(lat=1.0, lon=2.0, region="R", delta_mean=5, delta_max=9, delta_std=1, cluster_size=8, bbox_area=12, shape=shape)
    rows = train_real.analyst_rows([
        {**base, "category": "illegal_encroachment", "status": "verified"},
        {**base, "category": "illegal_encroachment", "status": "false positive"},
        {**base, "category": "illegal_encroachment", "true_category": "agricultural_burn"},
    ])
    assert list(rows["y"]) == [CLASSES.index("illegal_encroachment"), CLASSES.index("agricultural_burn")]
    assert (rows["w"] == train_real.ANALYST_WEIGHT).all()


def test_detect_class_mapping_and_extract_clusters():
    import detect

    class M:
        classes_ = np.array([2, 0, 1, 3])
    assert detect._class_names(M()) == [CLASSES[2], CLASSES[0], CLASSES[1], CLASSES[3]]
    delta = np.zeros((40, 40)); delta[10:15, 10:15] = 9.0
    alerts, feats, n = detect.extract_clusters(delta, lambda r, c: (c, r), detect.DetectionParams(regional_correction=False),
                                               classify=False)
    assert n == 25 and len(alerts) == 1 and feats[0].cluster_size == 25
    delta[10:15, 10:15] = np.nan                                   # masked (cloudy) pixels never become alerts
    assert detect.extract_clusters(delta, lambda r, c: (c, r), detect.DetectionParams(regional_correction=False),
                                   classify=False)[0] == []
