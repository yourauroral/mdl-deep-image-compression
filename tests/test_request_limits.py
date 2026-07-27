from mdlic.request_limits import SlidingWindowRateLimiter


def test_sliding_window_limiter_releases_expired_requests():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=10)

    assert limiter.admit("client", now=0) == (True, 0)
    assert limiter.admit("client", now=1) == (True, 0)
    assert limiter.admit("client", now=2) == (False, 8)
    assert limiter.admit("client", now=10) == (True, 0)


def test_sliding_window_limiter_bounds_client_keys():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=10, max_keys=1)

    assert limiter.admit("first", now=0)[0] is True
    assert limiter.admit("second", now=1) == (False, 10)
    assert limiter.admit("second", now=11)[0] is True


def test_zero_limit_disables_limiter():
    limiter = SlidingWindowRateLimiter(limit=0, window_seconds=10)
    assert limiter.admit("client", now=0) == (True, 0)
