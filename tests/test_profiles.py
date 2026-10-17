import pytest

from agentic_generation.profiles import GenerationProfile, resolve_profile


@pytest.mark.parametrize(
    ("profile_id", "expected_goal"),
    [
        (
            "share_concern",
            "Seek understanding of an existing concern. Explain relevant feelings when "
            "appropriate; practical help is optional.",
        ),
        (
            "weigh_options",
            "Explore an existing uncertainty and what matters to you before deciding. "
            "Reaching a decision is optional.",
        ),
        (
            "seek_practical_help",
            "Seek a concrete next step for an existing concern when useful. Do not invent "
            "a new task merely to keep talking.",
        ),
    ],
)
def test_resolve_profile_returns_versioned_profile(profile_id, expected_goal):
    profile = resolve_profile(profile_id)

    assert profile == GenerationProfile(
        id=profile_id,
        version="1",
        goal=expected_goal,
    )


def test_resolve_profile_preserves_profile_free_baseline():
    assert resolve_profile(None) is None


def test_resolve_profile_rejects_unknown_id():
    with pytest.raises(ValueError, match="Unknown generation profile"):
        resolve_profile("unknown")
