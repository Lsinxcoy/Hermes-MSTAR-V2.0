"""Skill Effort Classifier — M* Paper + Claude CBU Thinking Effort Integration.

Automatically classifies skill execution complexity and assigns appropriate
LLM config (max_tokens, reasoning_effort) before dispatch.

Inspired by:
  - Claude CBU "Thinking Effort"分级 (low/medium/high/max per model)
  - M* Paper 5-Phase effort mapping (P1-2=MEDIUM, P3-4=HIGH, P5=MEDIUM)

Design principles:
  - Zero LLM calls for classification (rule-based, deterministic)
  - Falls back to MEDIUM for unknown skills
  - Integrates with FitnessTracker for adaptive reclassification
"""

from __future__ import annotations

import os
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Effort Levels — mirrors Claude CBU "thinking effort" tiers
# ---------------------------------------------------------------------------

class EffortLevel(Enum):
    LOW = "low"       # Simple lookup, recall, basic retrieval
    MEDIUM = "medium" # Standard single-skill execution
    HIGH = "high"     # Multi-step, mutation, complex reasoning
    MAX = "max"       # Novel domains, deep research, complex design


# ---------------------------------------------------------------------------
# Effort → LLM Config mapping (provider-agnostic)
# ---------------------------------------------------------------------------

DEFAULT_EFFORT_TOKENS: dict[EffortLevel, int] = {
    EffortLevel.LOW: 4096,
    EffortLevel.MEDIUM: 16384,
    EffortLevel.HIGH: 32000,
    EffortLevel.MAX: 64000,
}

DEFAULT_EFFORT_REASONING: dict[EffortLevel, str] = {
    EffortLevel.LOW: "minimal",
    EffortLevel.MEDIUM: "medium",
    EffortLevel.HIGH: "high",
    EffortLevel.MAX: "xhigh",
}

# ---------------------------------------------------------------------------
# Keyword classifiers
# ---------------------------------------------------------------------------

_SIMPLE_KEYWORDS = frozenset({
    "lookup", "remember", "retrieve", "search",
    "list", "show", "get", "find", "count",
    "check", "view", "read", "info",
})

_COMPLEX_KEYWORDS = frozenset({
    # M* Paper complex operations
    "mutate", "evolve", "reflect", "analyze", "research",
    "compile", "transform", "optimize", "design", "architect",
    "debug", "refactor", "crossover", "fitness", "selection",
    "breed", "propagate", "reinforce",
    # General complex operations
    "create", "build", "write", "generate", "synthesize",
    "compare", "evaluate", "validate", "verify",
    "integrate", "orchestrate", "coordinate", "plan",
})


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class SkillEffortClassifier:
    """
    Rule-based skill complexity classifier.

    Classification order (first match wins):
      1. Exact skill name match in SKILL_COMPLEXITY map
      2. Keyword matching against skill name
      3. M* Phase mapping (if skill has phase metadata)
      4. Adaptive reclassification via FitnessTracker fitness score
      5. Default: MEDIUM
    """

    SKILL_COMPLEXITY: dict[str, EffortLevel] = {
        # --- Simple skills ---
        "skill_view": EffortLevel.LOW,
        "skills_list": EffortLevel.LOW,
        "bump_view": EffortLevel.LOW,
        "bump_use": EffortLevel.LOW,
        "session_search": EffortLevel.LOW,
        "memory": EffortLevel.LOW,
        # --- Medium complexity ---
        "烤烤酱的写作": EffortLevel.MEDIUM,
        "kaokao-writing": EffortLevel.MEDIUM,
        "oneclick-article": EffortLevel.MEDIUM,
        "huashu-proofreading": EffortLevel.MEDIUM,
        "huashu-article-to-x": EffortLevel.MEDIUM,
        # --- High complexity ---
        "mutate_skills": EffortLevel.HIGH,
        "reflect_on_failures": EffortLevel.HIGH,
        "reflector_agent": EffortLevel.HIGH,
        "population_pool": EffortLevel.HIGH,
        "evolution_engine": EffortLevel.HIGH,
        "quality_gates": EffortLevel.HIGH,
        "task_domain": EffortLevel.HIGH,
        "validation_set": EffortLevel.HIGH,
        "rtk_optimizer": EffortLevel.HIGH,
        "fitness_tracker": EffortLevel.HIGH,
        "forgetting": EffortLevel.HIGH,
        # --- Max complexity ---
        "systematic_debugging": EffortLevel.MAX,
        "subagent_driven_development": EffortLevel.MAX,
        "deep_research": EffortLevel.MAX,
    }

    PHASE_EFFORT: dict[int, EffortLevel] = {
        0: EffortLevel.LOW,
        1: EffortLevel.MEDIUM,
        2: EffortLevel.MEDIUM,
        3: EffortLevel.HIGH,
        4: EffortLevel.HIGH,
        5: EffortLevel.MEDIUM,
    }

    @classmethod
    def classify(
        cls,
        skill_name: str,
        skill_metadata: Optional[dict] = None,
        fitness_score: Optional[float] = None,
    ) -> EffortLevel:
        """Classify a skill's execution complexity."""
        name_lower = skill_name.lower().strip()

        # 1. Exact match
        if skill_name in cls.SKILL_COMPLEXITY:
            return cls.SKILL_COMPLEXITY[skill_name]

        # 2. Keyword matching
        if any(k in name_lower for k in _SIMPLE_KEYWORDS):
            return EffortLevel.LOW
        if any(k in name_lower for k in _COMPLEX_KEYWORDS):
            return EffortLevel.HIGH

        # 3. M* Phase mapping
        if skill_metadata:
            phase = skill_metadata.get("mstar_phase")
            if isinstance(phase, int) and phase in cls.PHASE_EFFORT:
                phase_effort = cls.PHASE_EFFORT[phase]
                if phase == 5 and fitness_score is not None and fitness_score < 0.4:
                    return EffortLevel.HIGH
                return phase_effort

            explicit = skill_metadata.get("complexity", "").lower()
            if explicit in ("low", "medium", "high", "max"):
                return EffortLevel(explicit)

        # 4. Adaptive via fitness
        if fitness_score is not None:
            if fitness_score < 0.4:
                return EffortLevel.HIGH
            elif fitness_score > 0.8:
                return EffortLevel.MEDIUM

        # 5. Default
        return EffortLevel.MEDIUM

    @classmethod
    def get_llm_config(
        cls,
        level: EffortLevel,
        model: Optional[str] = None,
    ) -> dict:
        """Return LLM API config dict for the given effort level."""
        max_tokens = DEFAULT_EFFORT_TOKENS[level]
        reasoning_effort = DEFAULT_EFFORT_REASONING[level]

        if model:
            model_lower = model.lower()
            if "opus" in model_lower and "4.7" in model_lower:
                if level in (EffortLevel.HIGH, EffortLevel.MAX):
                    max_tokens = max(max_tokens, 64000)
            if "sonnet" in model_lower and "4.6" in model_lower:
                if level == EffortLevel.HIGH:
                    max_tokens = min(max_tokens, 32000)

        return {
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }

    @classmethod
    def get_fitness_for_skill(cls, skill_name: str) -> Optional[float]:
        """Fetch current fitness score for a skill from FitnessTracker."""
        try:
            from hermes_mstar.evolution.fitness_tracker import FitnessTracker
            tracker = FitnessTracker.get_instance()
            skills = tracker.get_all_skills()
            record = next(
                (s for s in skills if getattr(s, "skill_id", None) == skill_name),
                None,
            )
            if record:
                return getattr(record, "fitness", None)
        except Exception:
            pass
        return None

    @classmethod
    def classify_with_adaptive(
        cls,
        skill_name: str,
        skill_metadata: Optional[dict] = None,
    ) -> tuple[EffortLevel, dict]:
        """Full classification + LLM config in one call. Returns (EffortLevel, llm_config)."""
        fitness = cls.get_fitness_for_skill(skill_name)
        level = cls.classify(skill_name, skill_metadata, fitness)
        config = cls.get_llm_config(level)
        return level, config
