"""Tests for temporal versioning integration in Store."""

import pytest

from cortex.core.config import CortexConfig
from cortex.db.store import Store
from cortex.ontology.resolver import find_ontology


@pytest.fixture
def store(tmp_path):
    config = CortexConfig(data_dir=tmp_path)
    s = Store(config)
    s.initialize(find_ontology())
    return s


class TestStoreTemporalIntegration:
    def test_temporal_initialized_after_store_init(self, store):
        assert store.temporal is not None

    def test_update_creates_version_snapshot(self, store):
        obj_id = store.create(obj_type="fix", title="v1", content="original")
        store.update(obj_id, title="v2")
        assert store.temporal.version_count(obj_id) == 1

    def test_multiple_updates_create_versions(self, store):
        obj_id = store.create(obj_type="fix", title="v1", content="original")
        store.update(obj_id, title="v2")
        store.update(obj_id, title="v3")
        store.update(obj_id, title="v4")
        assert store.temporal.version_count(obj_id) == 3

    def test_version_contains_old_data(self, store):
        obj_id = store.create(obj_type="fix", title="original", content="test")
        store.update(obj_id, title="updated")
        version = store.temporal.get_version(obj_id, 1)
        assert version is not None
        assert version["title"] == "original"

    def test_temporal_none_when_not_initialized(self, tmp_path):
        config = CortexConfig(data_dir=tmp_path / "uninit")
        s = Store(config)
        # Not calling s.initialize()
        assert s.temporal is None
