"""Fixed appraisal contrasts for the contrastive-generation pilot."""

CATALOG_VERSION = "appraisal-catalog-v1"

# This is a working classification for the pilot, not an exhaustive or mutually
# exclusive taxonomy of emotion.
CATALOG = {
    "wanted_vs_unwanted": {
        "appraisal_a": "wanted",
        "appraisal_b": "unwanted",
        "description": ("Whether an event matches what the speaker wanted or goes against it."),
    },
    "opportunity_vs_burden": {
        "appraisal_a": "opportunity",
        "appraisal_b": "burden",
        "description": (
            "Whether added responsibility is a welcome opportunity or unwanted pressure."
        ),
    },
    "relief_vs_disruption": {
        "appraisal_a": "relief",
        "appraisal_b": "disruption",
        "description": ("Whether a change reduces pressure or disrupts the speaker's plans."),
    },
    "anticipation_vs_anxiety": {
        "appraisal_a": "anticipation",
        "appraisal_b": "anxiety",
        "description": (
            "Whether an upcoming event is approached with positive expectation or worry."
        ),
    },
    "pride_vs_embarrassment": {
        "appraisal_a": "pride",
        "appraisal_b": "embarrassment",
        "description": (
            "Whether attention or recognition feels affirming or uncomfortably exposing."
        ),
    },
    "welcome_vs_intrusion": {
        "appraisal_a": "welcome",
        "appraisal_b": "intrusion",
        "description": ("Whether someone's arrival or involvement feels welcome or invasive."),
    },
    "success_vs_disappointment": {
        "appraisal_a": "success",
        "appraisal_b": "disappointment",
        "description": (
            "Whether the speaker regards a result as meeting their hopes or falling short. "
            "Success here is the speaker's appraisal, not an objective fact."
        ),
    },
    "amusement_vs_offence": {
        "appraisal_a": "amusement",
        "appraisal_b": "offence",
        "description": ("Whether a remark or social interaction feels entertaining or offensive."),
    },
}
