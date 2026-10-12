import pytest
from pydantic import ValidationError

from agentic_generation.config import Settings


@pytest.fixture(autouse=True)
def isolate_database_environment(monkeypatch):
    for key in (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)


def test_environment_overrides_dotenv_and_special_password_is_preserved(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "POSTGRES_PORT=5433\nPOSTGRES_PASSWORD='local@pass%word:/?'\nUNRELATED_SETTING=value\n"
    )
    monkeypatch.setenv("POSTGRES_PORT", "5544")
    settings = Settings(_env_file=dotenv)
    assert settings.postgres_port == 5544
    assert settings.database_url.password == "local@pass%word:/?"
    assert "local@pass" not in repr(settings)
    assert "local@pass" not in str(settings.database_url)


def test_missing_password_fails_without_a_silent_default():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_settings_do_not_print_credentials():
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, postgres_password="do-not-print-me", postgres_port=0)
    assert "do-not-print-me" not in str(error.value)
