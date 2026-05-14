from pathlib import Path

from src.utils.config import load_project_config
from src.utils.helpers import ensure_directories


def test_ensure_directories_creates_paths(tmp_path):
    target = tmp_path / "reports" / "figures"
    ensure_directories([str(target)])
    assert target.exists()


def test_load_project_config_has_expected_defaults():
    config = load_project_config("configs/does_not_exist.yaml")
    assert config["project"]["target_column"] == "Churn"
    assert "raw_data" in config["paths"]
