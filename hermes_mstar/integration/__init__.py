"""Hermes MSTAR Integration Layer.

Performance optimization modules inspired by Claude Computer Use best practices
and M* Paper architecture.
"""

from hermes_mstar.integration.skill_effort_classifier import (
    EffortLevel,
    SkillEffortClassifier,
)
from hermes_mstar.integration.in_conversation_advisor import (
    InConversationAdvisor,
    AdvisorContext,
    AdvisorRecommendation,
    ADVISOR_DECISION_TYPES,
)

__all__ = [
    # Module 2: Skill Effort Classifier
    "EffortLevel",
    "SkillEffortClassifier",
    # Module 3: In-Conversation Advisor
    "InConversationAdvisor",
    "AdvisorContext",
    "AdvisorRecommendation",
    "ADVISOR_DECISION_TYPES",
]
