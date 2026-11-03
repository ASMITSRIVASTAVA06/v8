from tracking import apply_tracking, haversine_matrix, match_alerts


def alert(lat, lon, cat="illegal_encroachment", **kw):
    return {"lat": lat, "lon": lon, "category": cat, "severity": "high", "confidence": 70, **kw}


def test_haversine_one_degree_latitude_is_about_111_km():
    d = haversine_matrix([10.0], [20.0], [11.0], [20.0])
    assert abs(d[0, 0] - 111.2) < 0.5


def test_matching_is_one_to_one_and_prefers_closest():
    prev = [alert(26.0000, 75.0000, alert_id="TV-A")]
    cur = [alert(26.0030, 75.0000), alert(26.0010, 75.0000)]   # ~330 m and ~110 m away
    m = match_alerts(cur, prev)
    assert m == {1: 0}   # the closer one wins; the other stays unmatched


def test_far_alerts_do_not_match():
    assert match_alerts([alert(26.0, 75.0)], [alert(26.1, 75.0)]) == {}


def test_lifecycle_new_recurring_and_resolved():
    prev = [alert(26.0, 75.0, alert_id="TV-OLD", times_seen=1, first_seen="2026-01-01 00:00"),
            alert(27.0, 76.0, alert_id="TV-GONE", times_seen=2)]
    cur = [alert(26.0005, 75.0), alert(28.0, 77.0)]
    out, resolved = apply_tracking("R", cur, prev, "2026-02-01 00:00", "2026-02", "2026-01")
    assert out[0]["alert_id"] == "TV-OLD"
    assert out[0]["lifecycle"] == "recurring" and out[0]["times_seen"] == 2
    assert out[0]["first_seen"] == "2026-01-01 00:00"
    assert out[1]["lifecycle"] == "new" and out[1]["times_seen"] == 1
    assert [r["alert_id"] for r in resolved] == ["TV-GONE"]


def test_rescan_of_same_period_is_not_new_evidence():
    prev = [alert(26.0, 75.0, alert_id="TV-OLD", times_seen=2)]
    out, _ = apply_tracking("R", [alert(26.0, 75.0)], prev, "t", "2026-02", "2026-02")
    assert out[0]["times_seen"] == 2


def test_not_comparable_reports_nothing_resolved():
    prev = [alert(26.0, 75.0, alert_id="TV-OLD")]
    _, resolved = apply_tracking("R", [], prev, "t", "2026-02", "2026-01", comparable=False)
    assert resolved == []


def test_ids_are_unique():
    cur = [alert(26.0, 75.0), alert(26.0, 75.0)]
    out, _ = apply_tracking("R", cur, [], "t", "p", None)
    assert len({a["alert_id"] for a in out}) == 2
