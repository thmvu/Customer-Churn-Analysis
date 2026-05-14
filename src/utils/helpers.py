from pathlib import Path


def ensure_directories(paths: list[str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
