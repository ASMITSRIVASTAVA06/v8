import csv
import io

import alert_tools as t


def alert(i, **kw):
    a = {"alert_id": f"TV-{i}", "region": "R", "detected_at": "2026-02-01 10:00", "lat": 26.0 + i / 100,
         "lon": 75.0, "category": "illegal_encroachment", "severity": "high", "confidence": 70,
         "cluster_size": 5, "delta_mean": 6.0, "delta_max": 9.0, "delta_std": 1.2, "bbox_area": 12,
         "shape": {k: 0.5 for k in t.SHAPE_FEATURES}, "scores": {"illegal_encroachment": 0.7}}
    a.update(kw)
    return a


def test_dataframe_has_new_columns_and_defaults():
    df = t.to_dataframe([alert(1), alert(2, verdict="Possible", evidence=["a", "b"], lifecycle="new")])
    assert {"verdict", "lifecycle", "status", "evidence_text", "in_protected_area"} <= set(df.columns)
    assert df.loc[df.alert_id == "TV-2", "evidence_text"].iloc[0] == "a; b"
    assert (df["status"] == "new").all()


def test_exports_roundtrip():
    al = [alert(i) for i in range(5)]
    df = t.to_dataframe(al)
    rows = list(csv.DictReader(io.StringIO(t.to_csv_bytes(df).decode())))
    assert len(rows) == 5 and "verdict" in rows[0]
    assert b"<table>" in t.build_html_report(df)
    assert "TV-0" in t.build_summary_text(df)


def test_training_csv_uses_only_decided_alerts_with_full_features():
    al = [
        alert(1, status="verified"),                                              # label = own category
        alert(2, status="false positive", true_category="anomaly"),               # corrected label
        alert(3, status="false positive"),                                        # no true type -> skipped
        alert(4, status="new"),                                                   # undecided -> ignored
        alert(5, status="verified", delta_std=None),                              # missing feature -> skipped
        alert(6, status="verified", category="unclassified"),                     # not a model class -> skipped
    ]
    data, n, skipped = t.build_training_csv(al)
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    assert n == 2 and skipped == 3
    assert [r["label"] for r in rows] == ["illegal_encroachment", "anomaly"]
    assert list(rows[0].keys()) == t.MODEL_FEATURES + ["label"]


def test_recipient_parsing_and_links():
    ok, bad = t.parse_recipients("a@b.com; nope, c@d.org")
    assert ok == ["a@b.com", "c@d.org"] and bad == ["nope"]
    assert t.mailto_link(ok, "S", "body").startswith("mailto:a@b.com,c@d.org")


def test_footprint_is_about_500m():
    (s, w), (n, e) = t.pixel_footprint(26.0, 75.0)
    assert abs((n - s) * 111_320 - 500) < 1
