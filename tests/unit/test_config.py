import pytest

from common.config import apply_env_overrides, load_settings


def test_dataset_size_presets_and_numbers():
    base = {"simulation": {"n_transactions": 100}}
    assert apply_env_overrides(base, {"DATASET_SIZE": "medium"})["simulation"]["n_transactions"] == 1_000_000
    assert apply_env_overrides(base, {"DATASET_SIZE": "large"})["simulation"]["n_transactions"] == 10_000_000
    assert apply_env_overrides(base, {"DATASET_SIZE": "25000"})["simulation"]["n_transactions"] == 25_000


def test_invalid_dataset_size_rejected():
    with pytest.raises(ValueError):
        apply_env_overrides({"simulation": {"n_transactions": 1}}, {"DATASET_SIZE": "-5"})


def test_nested_override_parses_yaml_scalars():
    cfg = apply_env_overrides({"training": {"xgb": {"max_depth": 5}}}, {"FINGUARD__TRAINING__XGB__MAX_DEPTH": "7"})
    assert cfg["training"]["xgb"]["max_depth"] == 7


def test_load_settings_defaults_are_sane():
    s = load_settings(env={})
    assert 0 < s["training"]["train_fraction"] < 1
    assert s["kafka"]["topics"]["dead_letter"].endswith("dlq")
    assert s.path("models_dir").is_absolute()
