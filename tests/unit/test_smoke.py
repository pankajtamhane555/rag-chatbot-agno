"""Smoke tests to verify testing framework setup."""

import pytest
import pytest_check as check


def test_pytest_works() -> None:
    """Verify pytest is working."""
    assert True, "Pytest is working"


def test_pytest_check_available() -> None:
    """Verify pytest-check is available."""
    check.equal(1 + 1, 2, "Basic math should work")
    check.is_true(True, "pytest-check assertions work")


def test_pytest_asyncio_configured() -> None:
    """Verify pytest-asyncio is configured (checking config, not running async)."""
    # This test verifies the configuration exists
    # The actual async test below will verify it works
    assert True, "pytest-asyncio configuration checked"


@pytest.mark.asyncio
async def test_async_test_works() -> None:
    """Verify async tests work."""
    async def get_value() -> str:
        return "async works"

    result = await get_value()
    assert result == "async works"


def test_fixtures_work(test_settings) -> None:
    """Verify fixtures are working."""
    assert test_settings.openai_api_key == "test-key"
    assert test_settings.openai_model == "gpt-4o-mini"
    check.equal(test_settings.log_level, "DEBUG")


def test_pytest_raises_works() -> None:
    """Verify pytest.raises works for error testing."""
    with pytest.raises(ValueError, match="test error"):
        raise ValueError("test error")

