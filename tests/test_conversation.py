import pytest

from agentic_generation.conversation import (
    OPENING_USER_REQUEST,
    generate_conversation,
)
from agentic_generation.prompts import build_assistant_system_prompt
from agentic_generation.spec import ConversationSpec


def test_generates_alternating_exchanges_with_agent_specific_histories():
    calls = []
    user_outputs = iter(["User zero", "User two"])
    assistant_outputs = iter(["Assistant one", "Assistant three"])

    def user_generator(system_prompt, messages):
        calls.append(("user", system_prompt, messages))
        return next(user_outputs)

    def assistant_generator(system_prompt, messages):
        calls.append(("assistant", system_prompt, messages))
        return next(assistant_outputs)

    turns = generate_conversation(
        ConversationSpec(scenario="A test scenario.", max_exchanges=2),
        user_generator,
        assistant_generator,
    )

    assert [call[0] for call in calls] == ["user", "assistant", "user", "assistant"]
    assert [(turn.turn_index, turn.role, turn.text) for turn in turns] == [
        (0, "user", "User zero"),
        (1, "assistant", "Assistant one"),
        (2, "user", "User two"),
        (3, "assistant", "Assistant three"),
    ]
    assert calls[0][2] == [{"role": "user", "content": OPENING_USER_REQUEST}]
    assert calls[1][2] == [{"role": "user", "content": "User zero"}]
    assert calls[2][2] == [
        {"role": "user", "content": OPENING_USER_REQUEST},
        {"role": "assistant", "content": "User zero"},
        {"role": "user", "content": "Assistant one"},
    ]
    assert calls[3][2] == [
        {"role": "user", "content": "User zero"},
        {"role": "assistant", "content": "Assistant one"},
        {"role": "user", "content": "User two"},
    ]


def test_private_details_stay_out_of_assistant_request_before_disclosure():
    private_scenario = "PRIVATE_SCENARIO_DETAIL"
    private_personality = "PRIVATE_PERSONALITY_DETAIL"
    assistant_requests = []

    def user_generator(system_prompt, messages):
        assert private_scenario in system_prompt
        assert private_personality in system_prompt
        return "I would like to talk."

    def assistant_generator(system_prompt, messages):
        assistant_requests.append((system_prompt, messages))
        return "What would you like to discuss?"

    generate_conversation(
        ConversationSpec(
            scenario=private_scenario,
            user_personality={"private_detail": private_personality},
            max_exchanges=1,
        ),
        user_generator,
        assistant_generator,
    )

    system_prompt, messages = assistant_requests[0]
    request_text = system_prompt + repr(messages)
    assert private_scenario not in request_text
    assert private_personality not in request_text
    assert OPENING_USER_REQUEST not in request_text
    assert system_prompt == build_assistant_system_prompt()


def test_user_disclosure_reaches_assistant():
    disclosed_detail = "I nearly fell outside the shop yesterday."
    assistant_messages = []

    def user_generator(system_prompt, messages):
        return disclosed_detail

    def assistant_generator(system_prompt, messages):
        assistant_messages.extend(messages)
        return "That sounds frightening. Are you hurt?"

    generate_conversation(
        ConversationSpec(scenario="The user had a frightening experience.", max_exchanges=1),
        user_generator,
        assistant_generator,
    )

    assert assistant_messages == [{"role": "user", "content": disclosed_detail}]


@pytest.mark.parametrize("invalid_output", [None, 7, "", " \n "])
def test_invalid_user_output_stops_generation(invalid_output):
    calls = []

    def user_generator(system_prompt, messages):
        calls.append("user")
        return invalid_output

    def assistant_generator(system_prompt, messages):
        calls.append("assistant")
        return "This must not run."

    error_type = TypeError if invalid_output is None or invalid_output == 7 else ValueError
    with pytest.raises(error_type):
        generate_conversation(
            ConversationSpec(scenario="A test scenario.", max_exchanges=2),
            user_generator,
            assistant_generator,
        )

    assert calls == ["user"]


def test_failed_assistant_call_stops_generation_and_propagates_error():
    calls = []

    def user_generator(system_prompt, messages):
        calls.append("user")
        return "First user turn"

    def assistant_generator(system_prompt, messages):
        calls.append("assistant")
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        generate_conversation(
            ConversationSpec(scenario="A test scenario.", max_exchanges=3),
            user_generator,
            assistant_generator,
        )

    assert calls == ["user", "assistant"]


def test_invalid_assistant_output_stops_before_the_next_user_turn():
    calls = []

    def user_generator(system_prompt, messages):
        calls.append("user")
        return "First user turn"

    def assistant_generator(system_prompt, messages):
        calls.append("assistant")
        return "   "

    with pytest.raises(ValueError, match="assistant generator returned a blank output"):
        generate_conversation(
            ConversationSpec(scenario="A test scenario.", max_exchanges=2),
            user_generator,
            assistant_generator,
        )

    assert calls == ["user", "assistant"]
