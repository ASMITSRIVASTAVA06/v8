import reliability as r


def test_transient_error_is_retried_with_growing_delays():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("HTTP 503 Service Unavailable")
        return "ok"

    assert r.call_with_retry(flaky, sleep=waits.append, base_delay=2) == "ok"
    assert len(calls) == 3 and waits == [2, 4]


def test_permanent_error_is_not_retried():
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("Image asset 'X' not found")

    try:
        r.call_with_retry(broken, sleep=lambda s: None)
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    assert len(calls) == 1


def test_gives_up_after_the_last_attempt():
    calls = []

    def always():
        calls.append(1)
        raise TimeoutError("timed out")

    try:
        r.call_with_retry(always, attempts=3, sleep=lambda s: None)
    except TimeoutError:
        pass
    assert len(calls) == 3


def test_numbers_inside_unrelated_messages_do_not_trigger_retries():
    assert not r.is_transient(ValueError("bad scale 500"))
    assert r.is_transient(Exception("Too many concurrent aggregations"))
    assert r.is_transient(Exception("status code: 429"))
