"""In-Conversation Advisor — Zero-latency Strategic Guidance.

Provides high-intelligence advisor responses within the current LLM request,
without spawning a separate agent session (eliminates ~1-3s round-trip latency).

Inspired by: Claude CBU "advisor tool" pattern — executor calls advisor inside
a single API request; advisor response is embedded in the same completion.
"""

from __future__ import annotations

import os
import logging
import time
import hashlib
import json
import re
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision types that benefit from advisor guidance
# ---------------------------------------------------------------------------

ADVISOR_DECISION_TYPES = {
    "mutation_type",
    "fitness_threshold",
    "crossover_parents",
    "validation_strategy",
    "reflection_trigger",
    "forgetting_candidate",
    "effort_level",
}


@dataclass
class AdvisorContext:
    """Context bundle passed to the advisor at a decision point."""
    decision_type: str
    skill_name: str
    skill_id: str
    mutation_history: list = field(default_factory=list)
    population_diversity: float = 0.0
    validation_failures: int = 0
    recent_fitnesses: list = field(default_factory=list)
    custom: dict = field(default_factory=dict)


@dataclass
class AdvisorRecommendation:
    """Advisor's structured response at a decision point."""
    decision_type: str
    recommended_action: str
    confidence: float
    reasoning: str
    alternatives: list = field(default_factory=list)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Advisor system prompt
# ---------------------------------------------------------------------------

ADVISOR_SYSTEM_PROMPT = (
    "You are a strategic advisor for an AI skill evolution system (Hermes MSTAR). "
    "Your role is to provide concise, actionable recommendations at decision points. "
    "Respond ONLY with a RECOMMENDATION block (see format below). "
    "Be specific: name exact values, strategies, or actions. "
    "If uncertain, estimate confidence and state your assumptions. "
    "Output format:\n"
    "RECOMMENDATION\n"
    "type: {decision_type}\n"
    "action: {your specific recommendation}\n"
    "confidence: {0.0-1.0}\n"
    "reasoning: {2-3 sentence explanation}\n"
    "alternatives: {option1}, {option2}\n"
    "END_RECOMMENDATION"
)


# ---------------------------------------------------------------------------
# Advisor user prompt templates (one per decision type)
# ---------------------------------------------------------------------------

_ADVISOR_MUTATION_TYPE = (
    "You are advising on which mutation type to apply to a skill.\n\n"
    "Current skill: {skill_name}\n"
    "Recent mutation history: {mutation_history}\n"
    "Population diversity score: {population_diversity:.3f}\n"
    "Recent validation failures: {validation_failures}\n"
    "Recent fitness scores: {recent_fitnesses}\n\n"
    "Mutation types available:\n"
    "  - substitute: Replace a segment of the skill code\n"
    "  - scramble: Randomly shuffle a segment (preserves structure)\n"
    "  - expand: Add new capabilities to the skill\n"
    "  - contract: Remove unnecessary complexity\n"
    "  - regrow: Regenerate a specific section from scratch\n"
    "  - crossover: Combine with another high-fitness skill (requires 2 parents)\n\n"
    "Based on the current state, which mutation type has the highest probability "
    "of improving this skill? Consider: diversity (crossover helps when low), "
    "failure patterns, and fitness trajectory."
)

_ADVISOR_FITNESS_THRESHOLD = (
    "You are advising on whether to propagate a skill variant.\n\n"
    "Current fitness: {fitness:.4f}\n"
    "Population average fitness: {pop_avg:.4f}\n"
    "Number of evaluations: {n_evals}\n"
    "Fitness trend (last 5): {fitness_trend}\n\n"
    "Should we propagate this variant to the next generation? "
    "Consider: is fitness above population average? Is the trend improving? Enough samples?"
)

_ADVISOR_CROSSOVER_PARENTS = (
    "You are selecting parent programs for crossover breeding.\n\n"
    "Candidates:\n{candidates}\n\n"
    "Population diversity: {population_diversity:.3f}\n"
    "Top fitness in population: {top_fitness:.4f}\n\n"
    "Select the 2 best parents for crossover. "
    "Consider complementarity (different strengths) and "
    "diversity (unrelated programs tend to produce better offspring)."
)

_ADVISOR_VALIDATION_STRATEGY = (
    "You are choosing a validation strategy for a skill.\n\n"
    "Skill: {skill_name}\n"
    "Skill complexity: {complexity}\n"
    "Recent error rate: {error_rate:.3f}\n"
    "Available validation episodes: {n_episodes}\n\n"
    "Options:\n"
    "  - k=10: Fast, less reliable\n"
    "  - k=25: Balanced (M* Paper default)\n"
    "  - k=50: Thorough, more expensive\n\n"
    "Recommend a validation strategy."
)

_ADVISOR_REFLECTION_TRIGGER = (
    "You are deciding whether to invoke the LLM Reflector.\n\n"
    "Recent failures: {n_failures}\n"
    "Failure pattern: {failure_pattern}\n"
    "Current fitness: {fitness:.4f}\n\n"
    "The LLM Reflector (Phase 3) analyzes failures to suggest mutations. "
    "It costs ~500-1000 tokens per invocation. Should we invoke it now? "
    "Consider: is there a clear pattern? Is fitness suffering?"
)

ADVISOR_PROMPTS = {
    "mutation_type": _ADVISOR_MUTATION_TYPE,
    "fitness_threshold": _ADVISOR_FITNESS_THRESHOLD,
    "crossover_parents": _ADVISOR_CROSSOVER_PARENTS,
    "validation_strategy": _ADVISOR_VALIDATION_STRATEGY,
    "reflection_trigger": _ADVISOR_REFLECTION_TRIGGER,
}


# ---------------------------------------------------------------------------
# Main advisor class
# ---------------------------------------------------------------------------

class InConversationAdvisor:
    """
    Zero-latency advisor embedded in skill execution flow.

    Modes:
      aux   : Fast synchronous call to a smaller/cheaper model (PRODUCTION)
      shadow: Embed advisor hint in current request (experimental)
      disabled: No advisor calls

    Usage:
        advisor = InConversationAdvisor()
        rec = advisor.advise(AdvisorContext(decision_type="mutation_type", ...))
        if rec:
            mutation_type = rec.recommended_action
        else:
            mutation_type = default_selection()
    """

    DEFAULT_ADVISOR_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        mode: str = "aux",
        advisor_model: Optional[str] = None,
        enabled: bool = True,
        cache_ttl_seconds: float = 60.0,
    ):
        self.mode = mode
        self.advisor_model = advisor_model or self.DEFAULT_ADVISOR_MODEL
        self.enabled = enabled
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[AdvisorRecommendation, float]] = {}
        self._call_count = 0
        self._hit_count = 0

    @property
    def cache_hit_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._hit_count / self._call_count

    def advise(self, context: AdvisorContext) -> Optional[AdvisorRecommendation]:
        """Get advisor recommendation for a decision context."""
        if not self.enabled:
            return None

        self._call_count += 1

        cache_key = self._make_cache_key(context)
        if cache_key in self._cache:
            rec, cached_at = self._cache[cache_key]
            if time.monotonic() - cached_at < self.cache_ttl:
                self._hit_count += 1
                logger.debug("Advisor cache hit: %s", cache_key)
                return rec
            del self._cache[cache_key]

        prompt = self._build_prompt(context)
        if not prompt:
            return None

        if self.mode == "aux":
            rec = self._get_aux_recommendation(context, prompt)
        else:
            rec = None

        if rec:
            self._cache[cache_key] = (rec, time.monotonic())

        return rec

    def _make_cache_key(self, context: AdvisorContext) -> str:
        key_data = {
            "dt": context.decision_type,
            "sn": context.skill_name,
            "pd": context.population_diversity,
            "vf": context.validation_failures,
            "rf": context.recent_fitnesses[-3:] if context.recent_fitnesses else [],
            "cu": context.custom,
        }
        h = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        return h

    def _build_prompt(self, context: AdvisorContext) -> Optional[str]:
        template = ADVISOR_PROMPTS.get(context.decision_type)
        if not template:
            logger.warning("No advisor prompt for decision_type=%s", context.decision_type)
            return None

        try:
            # Build candidates string for crossover_parents
            if context.decision_type == "crossover_parents":
                candidates_list = context.custom.get("candidates", [])
                candidates_str = "\n".join(
                    f"  - {c.get('skill_id', '?')}: fitness={c.get('fitness', 0):.4f}, "
                    f"mutations={c.get('n_mutations', 0)}"
                    for c in candidates_list
                )
                context.custom["candidates"] = candidates_str

            prompt = template.format(
                skill_name=context.skill_name,
                skill_id=context.skill_id,
                mutation_history=context.mutation_history[-5:],
                population_diversity=context.population_diversity,
                validation_failures=context.validation_failures,
                recent_fitnesses=context.recent_fitnesses[-5:],
                candidates=context.custom.get("candidates", ""),
                pop_avg=context.custom.get("pop_avg", 0.5),
                top_fitness=context.custom.get("top_fitness", 0.7),
                n_evals=context.custom.get("n_evals", 10),
                fitness_trend=context.custom.get("fitness_trend", "unknown"),
                n_failures=context.custom.get("n_failures", 0),
                failure_pattern=context.custom.get("failure_pattern", "unknown"),
                fitness=context.custom.get("fitness", 0.5),
                complexity=context.custom.get("complexity", "medium"),
                error_rate=context.custom.get("error_rate", 0.1),
                n_episodes=context.custom.get("n_episodes", 25),
            )
            return prompt
        except (KeyError, TypeError) as e:
            logger.warning("Template variable error: %s", e)
            return None

    def _get_aux_recommendation(
        self, context: AdvisorContext, prompt: str
    ) -> Optional[AdvisorRecommendation]:
        """Make a fast synchronous call to a smaller/cheaper model."""
        start = time.monotonic()

        try:
            from agent.auxiliary_client import call_llm
            messages = [
                {"role": "system", "content": ADVISOR_SYSTEM_PROMPT.format(decision_type=context.decision_type)},
                {"role": "user", "content": prompt},
            ]
            response = call_llm(
                model=self.advisor_model,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            text = response if isinstance(response, str) else str(response)
            return self._parse_recommendation(context.decision_type, text, time.monotonic() - start)
        except ImportError:
            pass
        except Exception as e:
            logger.warning("Advisor aux call failed: %s", e)

        return self._call_direct_provider(prompt, start, context.decision_type)

    def _call_direct_provider(
        self, prompt: str, start: float, decision_type: str
    ) -> Optional[AdvisorRecommendation]:
        """Direct provider call as fallback."""
        try:
            import anthropic
        except ImportError:
            return None

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            return None

        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=self.advisor_model,
                max_tokens=512,
                system=ADVISOR_SYSTEM_PROMPT.format(decision_type=decision_type),
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            return self._parse_recommendation(decision_type, text, time.monotonic() - start)
        except Exception as e:
            logger.warning("Direct provider call failed: %s", e)
            return None

    def _parse_recommendation(
        self, decision_type: str, text: str, latency: float
    ) -> Optional[AdvisorRecommendation]:
        """Parse RECOMMENDATION block from advisor text."""
        pattern = (
            r"RECOMMENDATION\s*"
            r"type:\s*(\S+)\s*"
            r"action:\s*(.+?)\s*"
            r"confidence:\s*([\d.]+)\s*"
            r"reasoning:\s*(.+?)\s*"
            r"alternatives:\s*(.+?)\s*"
            r"END_RECOMMENDATION"
        )
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            action_match = re.search(r"action:\s*(.+?)(?:\n|$)", text)
            conf_match = re.search(r"confidence:\s*([\d.]+)", text)
            return AdvisorRecommendation(
                decision_type=decision_type,
                recommended_action=action_match.group(1).strip() if action_match else text[:100],
                confidence=float(conf_match.group(1)) if conf_match else 0.5,
                reasoning=text[:300],
                alternatives=[],
                latency_ms=latency * 1000,
            )

        return AdvisorRecommendation(
            decision_type=match.group(1).strip(),
            recommended_action=match.group(2).strip(),
            confidence=float(match.group(3)),
            reasoning=match.group(4).strip(),
            alternatives=[a.strip() for a in match.group(5).split(",")],
            latency_ms=latency * 1000,
        )

    def invalidate_cache(self, skill_id: Optional[str] = None) -> None:
        """Clear advisor cache. Pass skill_id to clear only that skill's entries."""
        if skill_id is None:
            self._cache.clear()
        else:
            to_remove = [k for k in self._cache if skill_id in k]
            for k in to_remove:
                del self._cache[k]
