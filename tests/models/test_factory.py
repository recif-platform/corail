"""Tests for ModelFactory — registry-based model resolution."""

import pytest

from corail.models.base import Model
from corail.models.factory import ModelFactory
from corail.models.stub import StubModel


class TestModelFactoryRegistry:
    def test_stub_returns_stub_model(self) -> None:
        instance = ModelFactory.create("stub")
        assert isinstance(instance, StubModel)
        assert isinstance(instance, Model)

    def test_stub_uses_default_model_id(self) -> None:
        instance = ModelFactory.create("stub")
        assert instance.model_id == "stub-echo"

    def test_stub_custom_model_id(self) -> None:
        instance = ModelFactory.create("stub", model_id="custom-stub")
        assert instance.model_id == "custom-stub"

    def test_all_types_registered(self) -> None:
        available = ModelFactory.available()
        for expected in ("stub", "ollama", "openai", "anthropic", "vertex-ai", "bedrock", "google-ai"):
            assert expected in available

    def test_available_returns_sorted_list(self) -> None:
        available = ModelFactory.available()
        assert available == sorted(available)

    def test_unknown_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown model type: nope"):
            ModelFactory.create("nope")

    def test_error_message_lists_available(self) -> None:
        with pytest.raises(ValueError, match="Available:"):
            ModelFactory.create("nonexistent")

    def test_create_returns_new_instance_each_call(self) -> None:
        a = ModelFactory.create("stub")
        b = ModelFactory.create("stub")
        assert a is not b
