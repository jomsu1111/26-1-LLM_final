import hashlib
from typing import Dict


def split_for_id(example_id: str, seed: int = 42, train_frac: float = 0.7, val_frac: float = 0.15) -> str:
    key = f"{seed}:{example_id}".encode("utf-8")
    value = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "validation"
    return "test"


def add_split(row: Dict, seed: int = 42, train_frac: float = 0.7, val_frac: float = 0.15) -> Dict:
    enriched = dict(row)
    enriched["split"] = split_for_id(str(row.get("example_id")), seed=seed, train_frac=train_frac, val_frac=val_frac)
    return enriched

