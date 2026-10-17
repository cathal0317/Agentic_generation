import pytest
from pydantic import ValidationError

from agentic_generation.controller import DialogueAssessment
from agentic_generation.conversation import (
    OPENING_USER_REQUEST,
    generate_conversation,
)
from agentic_generation.prompts import build_assistant_system_prompt
from agentic_generation.spec import ConversationSpec


def continue_review(transcript):
    return DialogueAssessment(decision="continue", reason="The dialogue is progressing.")


@pytest.mark.parametrize(
    "data",
    [
        {"decision": "unknown", "reason": "Unsupported decision."},
        {"decision": "continue", "reason": "   "},
        {"decision": "end", "reason": "Valid reason.", "extra": "not allowed"},
    ],
)
def test_dialogue_assessment_rejects_invalid_control_data(data):
    with pytest.raises(ValidationError):
        DialogueAssessment.model_validate(data)


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

    result = generate_conversation(
        ConversationSpec(scenario="A test scenario.", max_exchanges=2),
        user_generator,
        assistant_generator,
        continue_review,
    )
    turns = result.turns

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
    assert all("The dialogue is progressing." not in repr(call[2]) for call in calls)
    assert result.termination_reason == "max_exchanges"


def test_private_details_stay_out_of_assistant_request_before_disclosure():
    private_scenario = "PRIVATE_SCENARIO_DETAIL"
    private_personality = "PRIVATE_PERSONALITY_DETAIL"
    assistant_requests = []
    reviewer_transcripts = []

    def user_generator(system_prompt, messages):
        assert private_scenario in system_prompt
        assert private_personality in system_prompt
        return "I would like to talk."

    def assistant_generator(system_prompt, messages):
        assistant_requests.append((system_prompt, messages))
        return "What would you like to discuss?"

    def reviewer(transcript):
        reviewer_transcripts.append(transcript)
        return DialogueAssessment(decision="continue", reason="The user introduced a concern.")

    generate_conversation(
        ConversationSpec(
            scenario=private_scenario,
            user_personality={"private_detail": private_personality},
            max_exchanges=1,
        ),
        user_generator,
        assistant_generator,
        reviewer,
    )

    system_prompt, messages = assistant_requests[0]
    request_text = system_prompt + repr(messages)
    assert private_scenario not in request_text
    assert private_personality not in request_text
    assert OPENING_USER_REQUEST not in request_text
    assert system_prompt == build_assistant_system_prompt()
    assert private_scenario not in repr(reviewer_transcripts)
    assert private_personality not in repr(reviewer_transcripts)
    assert reviewer_transcripts == [
        [
            {"turn_index": 0, "role": "user", "text": "I would like to talk."},
            {
                "turn_index": 1,
                "role": "assistant",
                "text": "What would you like to discuss?",
            },
        ]
    ]


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
        continue_review,
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
            continue_review,
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
            continue_review,
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
            continue_review,
        )

    assert calls == ["user", "assistant"]


def test_natural_end_stops_all_subsequent_generation():
    calls = []

    def user_generator(system_prompt, messages):
        calls.append("user")
        return "That answers my question, thank you."

    def assistant_generator(system_prompt, messages):
        calls.append("assistant")
        return "You’re welcome."

    def reviewer(transcript):
        calls.append("reviewer")
        return DialogueAssessment(decision="end", reason="The user closed the conversation.")

    result = generate_conversation(
        ConversationSpec(scenario="A test scenario.", max_exchanges=5),
        user_generator,
        assistant_generator,
        reviewer,
    )

    assert calls == ["user", "assistant", "reviewer"]
    assert result.termination_reason == "natural_end"
    assert len(result.turns) == 2
    assert result.control_trace[0].after_turn_index == 1


def test_two_consecutive_stalls_stop_the_loop():
    calls = []

    def user_generator(system_prompt, messages):
        calls.append("user")
        return "Nothing else has changed."

    def assistant_generator(system_prompt, messages):
        calls.append("assistant")
        return "I understand."

    def reviewer(transcript):
        calls.append("reviewer")
        return DialogueAssessment(decision="stalled", reason="The exchange repeats itself.")

    result = generate_conversation(
        ConversationSpec(scenario="A test scenario.", max_exchanges=5),
        user_generator,
        assistant_generator,
        reviewer,
    )

    assert calls == ["user", "assistant", "reviewer"] * 2
    assert result.termination_reason == "stalled"
    assert len(result.turns) == 4
    assert [record.after_turn_index for record in result.control_trace] == [1, 3]


def test_progress_resets_the_consecutive_stall_counter():
    decisions = iter(["stalled", "continue", "stalled", "stalled"])
    generation_calls = []

    def user_generator(system_prompt, messages):
        generation_calls.append("user")
        return "User turn"

    def assistant_generator(system_prompt, messages):
        generation_calls.append("assistant")
        return "Assistant turn"

    def reviewer(transcript):
        decision = next(decisions)
        return DialogueAssessment(decision=decision, reason=f"The decision is {decision}.")

    result = generate_conversation(
        ConversationSpec(scenario="A test scenario.", max_exchanges=6),
        user_generator,
        assistant_generator,
        reviewer,
    )

    assert result.termination_reason == "stalled"
    assert len(result.turns) == 8
    assert generation_calls == ["user", "assistant"] * 4
    assert [record.decision for record in result.control_trace] == [
        "stalled",
        "continue",
        "stalled",
        "stalled",
    ]


def test_max_exchanges_remains_a_hard_cap():
    review_calls = []

    def user_generator(system_prompt, messages):
        return "User turn"

    def assistant_generator(system_prompt, messages):
        return "Assistant turn"

    def reviewer(transcript):
        review_calls.append(transcript)
        return DialogueAssessment(decision="continue", reason="The dialogue is progressing.")

    result = generate_conversation(
        ConversationSpec(scenario="A test scenario.", max_exchanges=2),
        user_generator,
        assistant_generator,
        reviewer,
    )

    assert result.termination_reason == "max_exchanges"
    assert len(result.turns) == 4
    assert len(review_calls) == 2
    assert len(result.control_trace) == 2
