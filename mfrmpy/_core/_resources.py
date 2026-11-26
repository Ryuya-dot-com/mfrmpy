"""English reporting notes included in the installed package."""
from functools import lru_cache
from importlib.resources import files
import json


@lru_cache(maxsize=1)
def _notes():
    return json.loads(files(__package__).joinpath("data", "en.json").read_text(encoding="utf-8"))["metric_notes"]


def metric_note(name: str) -> str:
    return _notes()[name]
