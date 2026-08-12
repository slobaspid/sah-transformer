from dataclasses import dataclass

@dataclass
class SearchConfig:
    c_puct: float = 1.5           # exploration constant in PUCT
    sims_per_second: float = 16.0  # predicted seconds -> simulation budget
    min_sims: int = 1             # instant moves fall back to (near) raw policy
    max_sims: int = 256           # human-realism cap: never calculate like an engine
    temperature: float = 0.0      # 0 = most-visited move; >0 = sample by visit counts
    elo: int = 1500               # skill to imitate during search

def time_to_sims(seconds: float, cfg: SearchConfig) -> int:
    """Turn a predicted human think-time (seconds) into a simulation budget."""
    n = round(float(seconds) * cfg.sims_per_second)
    return int(min(max(n, cfg.min_sims), cfg.max_sims))
