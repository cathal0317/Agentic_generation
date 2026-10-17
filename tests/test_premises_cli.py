import json
from types import SimpleNamespace

import pytest

from agentic_generation import premises as planner
from agentic_generation.config import GenerationSettings


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.parse_calls = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True


@pytest.fixture
def setup(tmp_path, monkeypatch):
    spec = planner.ConversationSpec(scenario="Yesterday, she stumbled. She did not fall.")
    path = tmp_path / "scenario.json"
    path.write_text(spec.model_dump_json(), encoding="utf-8")
    settings = GenerationSettings(
        _env_file=None,
        api_key="test-secret",
        user_model="user-model",
        assistant_model="assistant-model",
        max_output_tokens=2048,
    )
    monkeypatch.setattr(planner, "GenerationSettings", lambda: settings)
    batch = planner.PremiseBatch(
        premises=[
            planner.Premise(
                concern="Still shaken by yesterday's stumble.",
                opening="I am still shaken by yesterday.",
                source_quotes=["Yesterday, she stumbled."],
            )
        ]
    )
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=batch))
    client = FakeClient(responses)
    monkeypatch.setattr(planner, "OpenAI", lambda **kwargs: client)
    return path, client, batch


def test_command_outputs_reviewable_json_and_closes_client(setup, capsys):
    path, client, batch = setup

    assert planner.main([str(path)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["spec"] == json.loads(path.read_text())
    assert output["model"] == "user-model"
    assert output["max_output_tokens"] == 2048
    assert output["prompt_version"] == planner.PROMPT_VERSION
    assert output["review_status"] == "unreviewed"
    assert output["premises"] == [{"id": "p1", **batch.premises[0].model_dump()}]
    assert len(client.responses.parse_calls) == 1
    assert client.closed


def test_command_reports_provider_failure_without_partial_json_or_details(setup, capsys):
    path, client, _ = setup
    client.responses.error = RuntimeError("raw response with test-secret")

    assert planner.main([str(path)]) == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "Premise generation failed" in captured.err
    assert "raw response" not in captured.err
    assert "test-secret" not in captured.err
    assert client.closed


def test_command_rejects_missing_generation_settings(tmp_path, monkeypatch, capsys):
    path = tmp_path / "scenario.json"
    path.write_text('{"scenario": "A valid scenario."}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_USER_MODEL",
        "OPENAI_ASSISTANT_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    assert planner.main([str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid OpenAI settings" in captured.err
