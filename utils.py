from pathlib import Path


def check_path_exists(file_path):
    p = Path(file_path)
    return p.exists()

