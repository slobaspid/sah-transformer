from sahformer.search import SearchConfig, time_to_sims

def test_time_to_sims_scales_and_clamps():
    c = SearchConfig(sims_per_second=16.0, min_sims=1, max_sims=256)
    assert time_to_sims(0.0, c) == 1                 # clamped up to the floor
    assert time_to_sims(0.2, c) in (3, 4)            # ~0.2s obvious move -> a few sims
    assert time_to_sims(8.0, c) == 128               # hard position -> real calculation
    assert time_to_sims(10_000.0, c) == 256          # clamped to the human-realism cap
    # monotonic non-decreasing
    xs = [time_to_sims(s, c) for s in (0.1, 1, 3, 8, 20)]
    assert xs == sorted(xs)
