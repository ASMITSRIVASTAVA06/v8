import numpy as np

import regional


def scene(seed=0, brighten_right=2.5, darken_left=0.0):
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 0.3, (240, 240))
    d[:, 120:] += brighten_right               # electrification: right half glows more everywhere
    d[:, :120] -= darken_left
    d[100:105, 170:175] += 8                    # small encroachment-like patch (25 px)
    d[40:52, 160:172] += 12                     # larger burn-like patch (144 px)
    return d


def test_regional_brightening_no_longer_floods_the_map():
    d = scene()
    raw_flagged = int((d > 2.0).sum())
    fixed = regional.correct_delta(d)
    flagged = int((fixed > 2.0).sum())
    assert raw_flagged > 10_000                  # without correction half the raster is "suspicious"
    assert 150 <= flagged <= 260                 # with it, essentially just the two real patches


def test_real_clusters_keep_their_strength():
    fixed = regional.correct_delta(scene())
    assert fixed[102, 172] > 7.0                 # patch was +8 on top of +2.5 background -> ~8 after
    assert fixed[46, 166] > 10.5


def test_darkening_never_creates_alerts():
    d = scene(brighten_right=0.0, darken_left=2.0)
    fixed = regional.correct_delta(d)
    assert np.allclose(fixed[:, :100].mean(), d[:, :100].mean(), atol=0.15)   # left stays dark, not lifted to 0
    assert (fixed[:, :100] > 2.0).sum() == 0


def test_handles_nans_and_small_rasters():
    d = scene()
    d[:60, :60] = np.nan                         # no-data corner (e.g. ocean / missing tiles)
    fixed = regional.correct_delta(d)
    assert np.isnan(fixed[:60, :60]).all() and np.isfinite(fixed[100:, 100:]).all()
    tiny = np.full((5, 7), 1.0)
    assert regional.correct_delta(tiny).shape == (5, 7)
    assert np.allclose(regional.correct_delta(np.full((8, 8), np.nan)), np.full((8, 8), np.nan), equal_nan=True)


def test_non_multiple_of_block_shapes_work():
    d = np.random.default_rng(1).normal(0, 0.2, (203, 177))
    assert regional.regional_trend(d).shape == d.shape
