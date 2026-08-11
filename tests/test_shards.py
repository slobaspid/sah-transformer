import numpy as np
from sahformer.shards import elo_bin, records_to_arrays, balance_indices
from sahformer.records import PositionRecord
from sahformer.encoding import TEMPORAL_DIM

def _rec(elo):
    return PositionRecord(
        board=np.zeros((8, 8, 12), np.int8),
        history=np.zeros((7, 8, 8, 12), np.int8),
        stm=0, elo_self=elo, elo_opp=elo,
        temporal=np.zeros(TEMPORAL_DIM, np.float32),
        move_from=0, move_to=1, promo=0, result=1, think_time=1.0,
    )

def test_elo_bin_edges():
    assert elo_bin(650) == 0
    assert elo_bin(700) == 1
    assert elo_bin(1550) == 9
    assert elo_bin(3000) == 21

def test_records_to_arrays_shapes():
    arr = records_to_arrays([_rec(1500), _rec(1500)])
    assert arr["board"].shape == (2, 8, 8, 12)
    assert arr["temporal"].shape == (2, TEMPORAL_DIM)
    assert arr["think_time"].shape == (2,)
    assert arr["think_time"].dtype == np.float32

def test_balance_caps_to_smallest_bin():
    elos = np.array([1500, 1500, 1500, 800])  # bin9 x3, bin1 x1
    idx = balance_indices(elos, seed=0)
    kept = elos[idx]
    # each present bin capped to min_count = 1
    assert (kept == 1500).sum() == 1
    assert (kept == 800).sum() == 1
