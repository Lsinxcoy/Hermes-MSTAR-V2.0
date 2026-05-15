"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║           Hermes MSTAR: Reflector Agent — LLM-driven Failure Analysis                                           ║
║                                                                                                              ║
║  M* Paper Phase 3 Upgrade: Replace rules-based MSTARReflector with LLM-driven analysis                       ║
║                                                                                                              ║
║  M* paper Section 3.2:                                                                                      ║
║    "Coding Agent (GPT-5.3-Codex) analyzes failures → generate code patch"                                     ║
║                                                                                                              ║
║  This module provides:                                                                                      ║
║    - LLMReflector: Uses actual LLM to analyze failure context + generate mutation suggestions               ║
║    - ReflectorAgent: Facade that tries LLM first, falls back to MSTARReflector rules                        ║
║                                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .reflection import (
    MSTARReflector,
    FailurePattern,
    MutationProposal,
    ReflectionResult,
)
from ..memory_program import MutationType

logger = logging.getLogger("hermes.mstar.reflector_agent")

# ── LLM Client (simple, reads from environment/config) ──────────────────────────────────────────────────────

def _get_llm_client():
    """
    获取 LLM 客户端

    优先使用 OpenAI-compatible API（可配置 provider）
    读取环境变量或Hermes config
    """
    # Try Hermes's own LLM config
    try:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HERMES_LLM_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("HERMES_REFLECTOR_MODEL", "gpt-4o-mini")

        if not api_key:
            # Try to read from Hermes config
            from pathlib import Path
            config_path = Path.home() / ".hermes" / "config.yaml"
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path) as f:
                        cfg = yaml.safe_load(f) or {}
                    api_key = cfg.get("api_key", "") or cfg.get("llm", {}).get("api_key", "")
                    base_url = cfg.get("base_url", base_url)
                    model = cfg.get("model", model)
                except Exception:
                    pass

        if not api_key:
            return None, None, None

        return api_key, base_url, model
    except Exception as e:
        logger.debug(f"Could not init LLM client: {e}")
        return None, None, None


def _call_llm(prompt: str, model: Optional[str] = None, base_url: Optional[str] = None,
              api_key: Optional[str] = None, timeout: int = 30) -> Optional[str]:
    """
    简单 LLM 调用（OpenAI-compatible API）

    Returns: LLM response text, or None on failure
    """
    api_key, base_url, default_model = _get_llm_client()
    if not api_key:
        return None

    model = model or default_model
    base_url = base_url or "https://api.openai.com/v1"
    if base_url.endswith("/"):
        base_url = base_url[:-1]

    try:
        import urllib.request
        import urllib.error

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # Low temperature for analytical task
            "max_tokens": 800,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]

    except Exception as e:
        logger.debug(f"LLM call failed: {e}")
        return None


# ── Prompt Templates ─────────────────────────────────────────────────────────────────────────────────────────

REFLECTOR_SYSTEM_PROMPT = """You are a Memory Program Reflector for an AI agent system (Hermes MSTAR).

Your job is to analyze why a memory skill program failed or underperformed, and generate actionable mutation suggestions.

## Context
You are given:
1. The program's current state (name, trigger keywords, guidance, schema, etc.)
2. Recent execution results (success/failure, quality, latency, errors)
3. Failure patterns identified by the system

## Your Task
Analyze the failure context and produce a mutation recommendation.

## Output Format (JSON only, no extra text)
{
  "analysis": "2-3 sentence explanation of WHY the program is failing",
  "failure_patterns": ["timeout", "quality_low", "syntax_error", "context_overflow", "unknown"],
  "mutation_type": "EXTEND | SPECIALIZE | PRUNE | REFINE | CROSSOVER | GENERALIZE",
  "target_component": "instruction | schema | logic | keyword",
  "specific_change": "Concrete description of what to change (e.g., 'Add trigger keyword: coding, debug, syntax')",
  "expected_improvement": 0.0-0.3,
  "confidence": 0.0-1.0,
  "urgency": "high | medium | low",
  "reasoning": "Why this mutation is the best choice given the failure patterns"
}
"""

REFLECTOR_USER_PROMPT_TEMPLATE = """## Program: {program_name} (ID: {program_id})

**Current State:**
- Trigger Keywords: {trigger_keywords}
- Agent Guidance: {agent_guidance}
- Confidence Threshold: {confidence_threshold}
- Fitness Score: {fitness_score} (0.0-1.0, higher is better)
- Episode Count: {episode_count}
- Success Rate: {success_rate:.2f}
- Avg Quality: {avg_quality:.2f}
- Avg Latency: {avg_latency:.2f}s
- Failure Count: {failure_count}

**Recent Execution:**
- Success: {success}
- Quality: {quality:.2f}
- Latency: {latency:.2f}s
- Error: {error}

**System-Detected Failure Patterns:**
{failure_patterns}

Based on this context, what mutation would most improve this program?
"""


# ── LLM-driven Reflector ────────────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMReflectionResult:
    """LLM 反思结果"""
    analysis: str
    failure_patterns: List[str]
    mutation_type: str
    target_component: str
    specific_change: str
    expected_improvement: float
    confidence: float
    urgency: str
    reasoning: str


class LLMReflector:
    """
    LLM 驱动的反思器（M* paper Phase 3）

    使用 LLM 分析失败上下文，生成针对性的变异建议

    vs 规则系统（MSTARReflector）:
    - MSTARReflector: 关键字匹配，泛化能力强但缺乏深度理解
    - LLMReflector: 理解上下文，能生成更精准的变异建议
    """

    def __init__(self, timeout: int = 30, model: Optional[str] = None):
        self.timeout = timeout
        self.model = model
        self._llm_available: Optional[bool] = None

    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        if self._llm_available is None:
            api_key, _, _ = _get_llm_client()
            self._llm_available = bool(api_key)
        return self._llm_available

    def analyze(
        self,
        program_name: str,
        program_id: str,
        trigger_keywords: List[str],
        agent_guidance: str,
        confidence_threshold: float,
        fitness_score: float,
        episode_count: int,
        success_rate: float,
        avg_quality: float,
        avg_latency: float,
        failure_count: int,
        # Recent execution
        success: bool,
        quality: float,
        latency: float,
        error: Optional[str],
        # System-detected patterns
        system_failure_patterns: List[str],
    ) -> Optional[LLMReflectionResult]:
        """
        用 LLM 分析失败并生成变异建议

        Returns:
            LLMReflectionResult on success, None on failure (falls back to rules)
        """
        if not self.is_available():
            return None

        system_patterns_str = "\n".join(f"  - {p}" for p in system_failure_patterns) or "  - unknown"

        user_prompt = REFLECTOR_USER_PROMPT_TEMPLATE.format(
            program_name=program_name,
            program_id=program_id,
            trigger_keywords=", ".join(trigger_keywords[:20]),
            agent_guidance=(agent_guidance or "N/A")[:300],
            confidence_threshold=confidence_threshold,
            fitness_score=fitness_score,
            episode_count=episode_count,
            success_rate=success_rate,
            avg_quality=avg_quality,
            avg_latency=avg_latency,
            failure_count=failure_count,
            success="✓" if success else "✗",
            quality=quality,
            latency=latency,
            error=error or "None",
            failure_patterns=system_patterns_str,
        )

        full_prompt = f"{REFLECTOR_SYSTEM_PROMPT}\n\n{user_prompt}"

        response = _call_llm(full_prompt, timeout=self.timeout)
        if not response:
            return None

        return self._parse_response(response)

    def _parse_response(self, response: str) -> Optional[LLMReflectionResult]:
        """解析 LLM JSON 响应"""
        try:
            # Try to extract JSON from the response
            text = response.strip()

            # Handle markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                # Find the JSON start line
                json_start = -1
                json_end = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        if json_start == -1:
                            json_start = i + 1
                        else:
                            json_end = i
                            break
                if json_start > 0 and json_end > 0:
                    text = "\n".join(lines[json_start:json_end])
                else:
                    text = "\n".join(lines[1:] if lines[0].startswith("```") else lines)

            data = json.loads(text)

            return LLMReflectionResult(
                analysis=data.get("analysis", ""),
                failure_patterns=data.get("failure_patterns", []),
                mutation_type=data.get("mutation_type", "EXTEND"),
                target_component=data.get("target_component", "instruction"),
                specific_change=data.get("specific_change", ""),
                expected_improvement=float(data.get("expected_improvement", 0.1)),
                confidence=float(data.get("confidence", 0.5)),
                urgency=data.get("urgency", "medium"),
                reasoning=data.get("reasoning", ""),
            )
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse LLM response as JSON: {e}\nResponse: {response[:200]}")
            return None
        except Exception as e:
            logger.debug(f"Error parsing LLM response: {e}")
            return None


# ── Reflector Agent (Facade) ───────────────────────────────────────────────────────────────────────────────

class ReflectorAgent:
    """
    反思器代理（M* paper Phase 3 最终接口）

    Facade 模式：
    - 优先使用 LLMReflector（如果 LLM 可用）
    - LLM 失败时自动回退到 MSTARReflector（规则系统）
    - 保证总是返回有效结果

    M* paper 对应：
      "Coding Agent (GPT-5.3-Codex) analyzes failures → generate code patch"
    """

    def __init__(self):
        self._llm_reflector: Optional[LLMReflector] = None
        self._rules_reflector = MSTARReflector()

    def analyze(
        self,
        program,  # MemoryProgram
        success: bool,
        quality: float,
        latency: float,
        error: Optional[str] = None,
    ) -> ReflectionResult:
        """
        分析程序并生成反思结果

        优先 LLM，失败则回退规则系统
        """
        # 先用规则系统跑一遍，获取系统检测到的失败模式
        rules_result = self._rules_reflector.analyze(
            program=program,
            success=success,
            quality=quality,
            latency=latency,
            error=error,
        )

        # 尝试 LLM 分析
        llm_result = self._try_llm_analysis(program, rules_result, success, quality, latency, error)

        if llm_result is not None:
            return self._merge_results(rules_result, llm_result)

        return rules_result

    def _try_llm_analysis(
        self,
        program,
        rules_result: ReflectionResult,
        success: bool,
        quality: float,
        latency: float,
        error: Optional[str],
    ) -> Optional[LLMReflectionResult]:
        """尝试 LLM 分析，失败返回 None"""
        if self._llm_reflector is None:
            self._llm_reflector = LLMReflector()

        if not self._llm_reflector.is_available():
            return None

        system_patterns = [fp.value for fp in rules_result.failure_patterns]

        try:
            return self._llm_reflector.analyze(
                program_name=program.name,
                program_id=program.program_id,
                trigger_keywords=program.instructions.trigger_keywords if hasattr(program, "instructions") else [],
                agent_guidance=program.agent_guidance if hasattr(program, "agent_guidance") else "",
                confidence_threshold=program.confidence_threshold if hasattr(program, "confidence_threshold") else 0.5,
                fitness_score=program.fitness_score,
                episode_count=program.episode_count if hasattr(program, "episode_count") else 0,
                success_rate=rules_result.failure_count / max(1, rules_result.failure_count + 1),
                avg_quality=rules_result.quality or quality,
                avg_latency=rules_result.latency or latency,
                failure_count=rules_result.failure_count,
                success=success,
                quality=quality,
                latency=latency,
                error=error or rules_result.error,
                system_failure_patterns=system_patterns,
            )
        except Exception as e:
            logger.debug(f"LLM analysis failed: {e}")
            return None

    def _merge_results(
        self,
        rules_result: ReflectionResult,
        llm_result: LLMReflectionResult,
    ) -> ReflectionResult:
        """
        合并规则系统 + LLM 的结果

        策略：
        - 使用 LLM 的 failure_patterns（如果更丰富）
        - 使用 LLM 的 mutation_type 建议（高置信度时）
        - 保留规则系统的 needs_mutation / critical 判断作为安全网
        """
        # 更新 failure_patterns（如果 LLM 提供了更多）
        llm_patterns = []
        for p in llm_result.failure_patterns:
            try:
                llm_patterns.append(FailurePattern(p))
            except ValueError:
                llm_patterns.append(FailurePattern.UNKNOWN)

        # 如果 LLM 发现了更多模式，合并
        merged_patterns = list(rules_result.failure_patterns)
        for p in llm_patterns:
            if p not in merged_patterns:
                merged_patterns.append(p)

        # 生成 LLM 驱动的变异建议
        llm_proposals = []
        if llm_result.mutation_type and llm_result.confidence >= 0.5:
            try:
                mt = MutationType(llm_result.mutation_type.lower().replace("_", "_"))
            except ValueError:
                mt = MutationType.KEYWORD_ADD

            llm_proposals.append(MutationProposal(
                mutation_type=mt,
                priority=int(llm_result.confidence * 10),
                reason=f"LLM: {llm_result.reasoning}",
                expected_improvement=llm_result.expected_improvement,
                details={
                    "source": "llm",
                    "target_component": llm_result.target_component,
                    "specific_change": llm_result.specific_change,
                    "urgency": llm_result.urgency,
                },
            ))

        # 合并 proposals（LLM 优先）
        all_proposals = llm_proposals + rules_result.proposals

        # 决定最终动作（用规则系统的安全判断，但优先级用 LLM 的 urgency）
        urgency_map = {"high": 3, "medium": 2, "low": 1}
        if llm_result.urgency in urgency_map:
            if urgency_map.get(llm_result.urgency, 0) >= 3:
                needs_mutation = True
            elif not rules_result.needs_mutation and urgency_map.get(llm_result.urgency, 0) < 2:
                needs_mutation = False
            else:
                needs_mutation = rules_result.needs_mutation
        else:
            needs_mutation = rules_result.needs_mutation

        return ReflectionResult(
            program_id=rules_result.program_id,
            program_name=rules_result.program_name,
            timestamp=datetime.now().isoformat(),
            failure_patterns=merged_patterns,
            failure_count=rules_result.failure_count,
            recent_errors=rules_result.recent_errors,
            proposals=all_proposals,
            needs_mutation=needs_mutation,
            critical=rules_result.critical or llm_result.urgency == "high",
            suggested_action=rules_result.suggested_action,
            success=rules_result.success,
            quality=rules_result.quality,
            latency=rules_result.latency,
            error=rules_result.error,
        )
