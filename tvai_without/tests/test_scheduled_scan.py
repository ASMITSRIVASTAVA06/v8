import scheduled_scan as ss


def alert(i, **kw):
    a = {"alert_id": f"TV-{i}", "region": "R", "detected_at": "2026-02-01 10:00", "lat": 26.0 + i / 100,
         "lon": 75.0, "category": "illegal_encroachment", "severity": "high", "confidence": 75,
         "cluster_size": 5, "delta_mean": 6.0, "delta_max": 9.0, "lifecycle": "new",
         "verdict": "Possible", "shape": {}, "scores": {}}
    a.update(kw)
    return a


def test_only_new_credible_unreviewed_alerts_are_selected():
    alerts = [
        alert(1),                                       # kept
        alert(2, lifecycle="recurring"),                # not new
        alert(3, confidence=40),                        # below the bar
        alert(4, verdict="Likely false positive"),      # judged false positive
        alert(5, urban_light=True),                     # urban light
        alert(6, in_protected_area=False),              # outside the forest
        alert(7, status="verified"),                    # already handled
        alert(8, category="natural_wildfire"),          # wrong type for this digest
        alert(9, in_protected_area=True, verdict="Strong evidence"),   # kept
    ]
    ids = [a["alert_id"] for a in ss.select_digest_alerts(alerts, min_confidence=60)]
    assert ids == ["TV-1", "TV-9"]


def test_recurring_included_on_request():
    alerts = [alert(1, lifecycle="recurring")]
    assert ss.select_digest_alerts(alerts, only_new=False)


def test_digest_is_none_when_nothing_qualifies():
    scan = {"region": "R", "alerts": [alert(1, confidence=10)], "resolved": []}
    assert ss.build_digest([scan]) is None


def test_digest_text_mentions_region_and_alerts():
    scan = {"region": "Aravalli Hills, India", "period": "2026-08", "resolved": [{"alert_id": "x"}],
            "alerts": [alert(1, region="Aravalli Hills, India"), alert(2, region="Aravalli Hills, India")]}
    d = ss.build_digest([scan])
    assert d["count"] == 2 and "Aravalli Hills, India" in d["subject"]
    assert "TV-1" in d["body"] and "1 resolved" in d["body"]


def test_dry_run_sends_nothing():
    scan = {"region": "R", "alerts": [alert(1)], "resolved": []}
    d = ss.build_digest([scan])
    out = ss.deliver(d, ["a@b.com"], True, dry_run=True)
    assert out[0].startswith("[dry run]")


def test_region_parsing():
    assert "Borneo Rainforest" in ss.parse_regions("all")
    assert ss.parse_regions("Aravalli Hills, India") == ["Aravalli Hills, India"]
    assert ss.parse_regions("Borneo Rainforest;Black Forest, Germany") == ["Borneo Rainforest", "Black Forest, Germany"]
