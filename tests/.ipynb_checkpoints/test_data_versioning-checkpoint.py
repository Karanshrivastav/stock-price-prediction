# tests/test_data_versioning.py
import pathlib
def test_data_versions_present():
    assert pathlib.Path("data_versions/v0").exists()
    assert pathlib.Path("data_versions/v1").exists()
