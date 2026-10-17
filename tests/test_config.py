import pytest
from pydantic import ValidationError

from agentic_generation.config import GenerationSettings, Settings


@pytest.fixture(autouse=True)
def isolate_database_environment(monkeypatch):
    for key in (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "OPENAI_API_KEY",
        "OPENAI_USER_MODEL",
        "OPENAI_ASSISTANT_MODEL",
        "OPENAI_MAX_OUTPUT_TOKENS",
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


def test_generation_settings_have_separate_required_values_and_token_default():
    settings = GenerationSettings(
        _env_file=None,
        api_key="generation-secret",
        user_model="user-model",
        assistant_model="assistant-model",
    )

    assert settings.user_model == "user-model"
    assert settings.assistant_model == "assistant-model"
    assert settings.max_output_tokens == 2048
    assert "generation-secret" not in repr(settings)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"api_key": " ", "user_model": "user-model", "assistant_model": "assistant-model"},
        {"api_key": "key", "user_model": " ", "assistant_model": "assistant-model"},
        {
            "api_key": "key",
            "user_model": "user-model",
            "assistant_model": "assistant-model",
            "max_output_tokens": 0,
        },
    ],
)
def test_generation_settings_reject_missing_or_invalid_values(overrides):
    with pytest.raises(ValidationError):
        GenerationSettings(_env_file=None, **overrides)
