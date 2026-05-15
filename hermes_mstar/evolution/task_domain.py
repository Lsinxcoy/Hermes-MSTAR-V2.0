"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║           Hermes MSTAR: Task Domain — Task-Specific Fitness Calibration                                             ║
║                                                                                                              ║
║  M* Paper Phase 5: Task-Specific Fitness μ                                                                    ║
║                                                                                                              ║
║  Paper specification:                                                                                       ║
║    "M* uses task-specific μ in the fitness function"                                                         ║
║                                                                                                              ║
║  Different task domains prioritize different metrics:                                                         ║
║    - CODING: success_rate (0.8) > quality (0.2), latency penalty                                             ║
║    - RESEARCH: success (0.6), quality (0.4), token efficiency                                                 ║
║    - WRITING: success (0.5), quality (0.5), creativity bonus                                                  ║
║    - GENERAL: balanced generic approach                                                                       ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

logger = logging.getLogger(__name__)


class TaskDomain(str, Enum):
    """任务领域枚举"""
    CODING = "coding"          # 代码/调试任务
    RESEARCH = "research"      # 研究/调研任务
    WRITING = "writing"        # 写作/创作任务
    GENERAL = "general"        # 通用任务


# ── M* Paper Phase 5: Task-Specific Fitness Weights ─────────────────────────────────────────────────────────

@dataclass
class FitnessWeights:
    """
    M* Paper Phase 5: Task-Specific Fitness Weights

    每个任务域有不同的 fitness 公式权重。

    基础公式：
        base = w_success * success_rate + w_quality * quality
        decay = time_decay ^ hours_since_update
        conf_factor = 0.5 + 0.5 * confidence
        token_factor = 1 + w_token * ln(1 + tokens)
        latency_factor = 1 + w_latency * ln(1 + latency)

        fitness = base * decay * conf_factor * token_factor * latency_factor

    M* Paper 基准（GENERAL）:
        success=0.7, quality=0.3, latency_weight=-0.1, token_weight=-0.05
    """
    # 核心权重
    success: float = 0.7
    quality: float = 0.3

    # 惩罚/奖励系数
    latency_weight: float = -0.1      # 负数 = 延迟越高 fitness 越低
    token_weight: float = -0.05      # 负数 = Token 越多 fitness 越低

    # 时间衰减（每小时）
    time_decay: float = 0.995         # 每小时保留 99.5%

    # Confidence 调节
    confidence_min: float = 0.5
    confidence_max: float = 1.0

    @classmethod
    def for_domain(cls, domain: TaskDomain) -> 'FitnessWeights':
        """获取指定任务域的权重配置"""
        return TASK_WEIGHTS.get(domain, TASK_WEIGHTS[TaskDomain.GENERAL])

    def apply(self, success_rate: float, quality: float,
              latency: float, tokens: int, hours_elapsed: float,
              confidence: float) -> float:
        """
        计算 task-specific fitness

        Args:
            success_rate: 成功率 [0.0, 1.0]
            quality: 质量分 [0.0, 1.0]
            latency: 延迟（秒）
            tokens: Token 消耗数
            hours_elapsed: 距上次更新的小时数
            confidence: 置信度 [0.0, 1.0]

        Returns:
            fitness score [0.0, 1.0]
        """
        # 1. 基础分数
        base = self.success * success_rate + self.quality * quality

        # 2. 时间衰减
        decay = math.pow(self.time_decay, hours_elapsed)

        # 3. Confidence 因子
        conf_max = self.confidence_max if self.confidence_max > 0 else 1.0
        conf_factor = 0.5 + 0.5 * (confidence / conf_max)

        # 4. Token 因子
        if self.token_weight != 0 and tokens > 0:
            token_factor = 1.0 + self.token_weight * math.log1p(tokens)
        else:
            token_factor = 1.0

        # 5. Latency 惩罚
        latency_factor = 1.0
        if self.latency_weight != 0 and latency > 0:
            latency_factor = 1.0 + self.latency_weight * math.log1p(latency)

        # 合成
        fitness = base * decay * conf_factor * token_factor * latency_factor

        return max(0.0, min(1.0, fitness))


# ── Task Domain Configurations ──────────────────────────────────────────────────────────────────────────────

TASK_WEIGHTS: Dict[TaskDomain, FitnessWeights] = {
    TaskDomain.CODING: FitnessWeights(
        success=0.8,
        quality=0.2,
        latency_weight=-0.15,     # Coding 对延迟更敏感
        token_weight=-0.03,
        time_decay=0.990,
        confidence_min=0.6,
        confidence_max=1.0,
    ),
    TaskDomain.RESEARCH: FitnessWeights(
        success=0.6,
        quality=0.4,
        latency_weight=-0.05,
        token_weight=-0.08,       # 更重视 Token 效率
        time_decay=0.998,
        confidence_min=0.4,
        confidence_max=1.0,
    ),
    TaskDomain.WRITING: FitnessWeights(
        success=0.5,
        quality=0.5,              # Writing 质量和成功率同等重要
        latency_weight=-0.02,
        token_weight=-0.02,
        time_decay=0.992,
        confidence_min=0.3,
        confidence_max=1.0,
    ),
    TaskDomain.GENERAL: FitnessWeights(
        success=0.7,
        quality=0.3,
        latency_weight=-0.10,
        token_weight=-0.05,
        time_decay=0.995,
        confidence_min=0.5,
        confidence_max=1.0,
    ),
}


# ── Domain Detection Heuristics ──────────────────────────────────────────────────────────────────────────────

DOMAIN_KEYWORDS: Dict[TaskDomain, List[str]] = {
    TaskDomain.CODING: [
        "code", "coding", "debug", "refactor", "function", "class",
        "python", "javascript", "bug", "syntax", "api", "module",
        "debug", "implement", "test", "script", "cli", "repo",
    ],
    TaskDomain.RESEARCH: [
        "research", "survey", "analyze", "investigate", "review",
        "compare", "evaluate", "benchmark", "find", "search",
        "arxiv", "paper", "study", "query", "explore",
    ],
    TaskDomain.WRITING: [
        "write", "draft", "article", "blog", "content", "copy",
        "story", "narrative", "script", "edit", "proofread",
        "summary", "explain", "describe", "creative",
    ],
}


def detect_domain(keywords: List[str]) -> TaskDomain:
    """
    从 trigger keywords 检测任务域

    Args:
        keywords: trigger_keywords 列表

    Returns:
        检测到的 TaskDomain，默认 GENERAL
    """
    if not keywords:
        return TaskDomain.GENERAL

    keyword_set = set(k.lower() for k in keywords)

    scores = {}
    for domain, domain_kws in DOMAIN_KEYWORDS.items():
        overlap = len(keyword_set & set(domain_kws))
        scores[domain] = overlap

    if max(scores.values()) > 0:
        return max(scores, key=scores.get)

    return TaskDomain.GENERAL


def detect_domain_from_skill_name(skill_name: str) -> TaskDomain:
    """
    从 skill name 检测任务域

    Args:
        skill_name: skill 的名称

    Returns:
        检测到的 TaskDomain
    """
    if not skill_name:
        return TaskDomain.GENERAL

    name_lower = skill_name.lower()
    scores = {}

    for domain, domain_kws in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in domain_kws if kw in name_lower)
        scores[domain] = count

    if max(scores.values()) > 0:
        return max(scores, key=scores.get)

    return TaskDomain.GENERAL