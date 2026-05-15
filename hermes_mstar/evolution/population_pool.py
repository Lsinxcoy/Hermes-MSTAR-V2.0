"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║              Hermes MSTAR: Population Pool — M* Paper Population-based Search                                    ║
║                                                                                                              ║
║  Ported from M* paper (arXiv:2604.11811) Section 3.3: Population-based Search                                  ║
║                                                                                                              ║
║  Key features:                                                                                               ║
║    - Softmax selection: P(x_i) = softmax(s(x_i) / τ), τ = 0.15                                               ║
║    - Diversity preservation via lineage tracking                                                            ║
║    - Population pruning (top-k by fitness + random diversity)                                               ║
║    - Crossover-ready parent selection                                                                        ║
║                                                                                                              ║
║  vs Linear search: 0.459 vs 0.318 (+44%) — M* paper Table 3                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .fitness_tracker import FitnessTracker
    from ..memory_program import MemoryProgram, TaskDomain

logger = logging.getLogger("hermes.mstar.population_pool")


class SelectionStrategy(str, Enum):
    """Selection strategy for population"""
    SOFTMAX = "softmax"           # M* paper: softmax with τ=0.15
    TOURNAMENT = "tournament"    # Tournament selection
    ROULETTE = "roulette"        # Roulette wheel (fitness-proportionate)
    UNIFORM = "uniform"          # Uniform random (baseline)


@dataclass
class PopulationMember:
    """
    种群成员 = MemoryProgram + 进化元数据

    扩展了 M* paper 的概念，增加了：
    - lineage（血缘追踪，用于多样性维护）
    - generation（代数，防止近亲繁殖）
    - parent_ids（多个父节点，用于 crossover）
    """
    program: MemoryProgram
    fitness_score: float
    last_evaluated: str
    generation: int = 1
    parent_ids: List[str] = field(default_factory=list)
    lineage: List[str] = field(default_factory=list)

    @classmethod
    def from_program(cls, program: MemoryProgram) -> "PopulationMember":
        return cls(
            program=program,
            fitness_score=program.fitness_score,
            last_evaluated=datetime.now().isoformat(),
            generation=1,
            parent_ids=[],
            lineage=[],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program.program_id,
            "name": self.program.name,
            "fitness_score": self.fitness_score,
            "last_evaluated": self.last_evaluated,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "lineage": self.lineage,
            "status": self.program.status.value if hasattr(self.program.status, "value") else str(self.program.status),
        }


class PopulationPool:
    """
    M* Paper 种群池实现

    核心算法 (Section 3.3):
        Selection probability: P(x_i) = softmax(s(x_i) / τ)
                                τ = 0.15 (temperature)

    特性:
    - 线程安全
    - 持久化到 SQLite（通过 FitnessTracker）
    - 自适应温度（当 fitness 差异过大时降低温度）
    - 多样性保护（禁止同 lineage 频繁交配）
    """

    # M* paper 参数
    DEFAULT_TAU = 0.15          # Softmax temperature (M* paper value)
    MIN_TAU = 0.05             # 最低温度（最激进的选择）
    MAX_TAU = 0.5             # 最高温度（最温和的选择）
    DEFAULT_MAX_SIZE = 20     # 种群上限
    DEFAULT_MIN_SIZE = 5      # 最低存活数

    def __init__(
        self,
        fitness_tracker: FitnessTracker,
        max_size: int = DEFAULT_MAX_SIZE,
        min_size: int = DEFAULT_MIN_SIZE,
        tau: float = DEFAULT_TAU,
    ):
        self.fitness_tracker = fitness_tracker
        self.max_size = max_size
        self.min_size = min_size
        self.tau = tau
        self._lock = threading.RLock()
        self._pool: List[PopulationMember] = []
        self._generation_counter = 0

    # ── Core Selection (M* Paper: Softmax) ─────────────────────────────────────────────────────────────

    def select(self, k: int = 1, strategy: SelectionStrategy = SelectionStrategy.SOFTMAX) -> List[PopulationMember]:
        """
        选择 k 个候选成员进行变异

        M* Paper: P(x_i) = softmax(s(x_i) / τ), τ = 0.15

        Args:
            k: 选择数量
            strategy: 选择策略（默认 softmax）

        Returns:
            选中的 PopulationMember 列表
        """
        if not self._pool:
            return []

        if strategy == SelectionStrategy.SOFTMAX:
            selected = self._softmax_select(k)
        elif strategy == SelectionStrategy.TOURNAMENT:
            selected = self._tournament_select(k)
        elif strategy == SelectionStrategy.ROULETTE:
            selected = self._roulette_select(k)
        else:
            selected = self._uniform_select(k)

        logger.debug(f"Selected {len(selected)} members via {strategy.value} (tau={self.tau})")
        return selected

    def _softmax_select(self, k: int) -> List[PopulationMember]:
        """
        M* Paper Softmax Selection

        P(x_i) = exp(s(x_i) / τ) / Σ exp(s(x_j) / τ)
        τ = 0.15 (temperature)

        实现: 使用 log-sum-exp 技巧防止数值溢出
        """
        if not self._pool:
            return []

        # 动态温度调整：如果 fitness 差异过大，降低温度
        fitness_scores = [m.fitness_score for m in self._pool]
        fitness_range = max(fitness_scores) - min(fitness_scores) if fitness_scores else 1.0

        if fitness_range > 0.5:
            # Fitness 分散，用更高温度让选择更均匀
            effective_tau = min(self.tau * 1.5, self.MAX_TAU)
        elif fitness_range < 0.1:
            # Fitness 集中，用更低温度让 top 更容易被选中
            effective_tau = max(self.tau * 0.7, self.MIN_TAU)
        else:
            effective_tau = self.tau

        # Softmax 概率计算（log-sum-exp 技巧）
        scores = [m.fitness_score for m in self._pool]
        max_score = max(scores)

        # exp(x_i / τ) / Σ exp(x_j / τ)
        # = exp((x_i - max) / τ) / Σ exp((x_j - max) / τ)
        exp_scores = [math.exp((s - max_score) / effective_tau) for s in scores]
        sum_exp = sum(exp_scores)

        if sum_exp == 0 or not math.isfinite(sum_exp):
            # 退化到均匀选择
            probs = [1.0 / len(self._pool)] * len(self._pool)
        else:
            probs = [e / sum_exp for e in exp_scores]

        # 采样 k 次（不放回）
        selected = []
        remaining_indices = list(range(len(self._pool)))
        remaining_probs = probs.copy()

        for _ in range(min(k, len(self._pool))):
            # 归一化剩余概率
            total = sum(remaining_probs)
            if total <= 0:
                idx = random.randrange(len(remaining_indices))
            else:
                norm_probs = [p / total for p in remaining_probs]
                idx = random.choices(range(len(remaining_indices)), weights=norm_probs, k=1)[0]

            selected.append(self._pool[remaining_indices[idx]])
            # 移除已选（不放回）
            removed_idx = remaining_indices.pop(idx)
            remaining_probs.pop(idx)
            # 重新归一化（对于下一次选择）

        return selected

    def _tournament_select(self, k: int, tournament_size: int = 3) -> List[PopulationMember]:
        """Tournament Selection: 随机选 tournament_size 个，赌最强"""
        selected = []
        for _ in range(k):
            candidates = random.sample(self._pool, min(tournament_size, len(self._pool)))
            winner = max(candidates, key=lambda m: m.fitness_score)
            selected.append(winner)
        return selected

    def _roulette_select(self, k: int) -> List[PopulationMember]:
        """Roulette Wheel: fitness-proportionate selection"""
        total_fitness = sum(m.fitness_score for m in self._pool)
        if total_fitness <= 0:
            return self._uniform_select(k)
        probs = [m.fitness_score / total_fitness for m in self._pool]
        return random.choices(self._pool, weights=probs, k=k)

    def _uniform_select(self, k: int) -> List[PopulationMember]:
        """Uniform random selection (baseline)"""
        return random.sample(self._pool, min(k, len(self._pool)))

    # ── Diversity ─────────────────────────────────────────────────────────────────────────────────────

    def get_diverse_members(self, k: int = 3) -> List[PopulationMember]:
        """
        获取多样性样本（血缘尽可能分散）

        用于 crossover 父代选择，避免近亲繁殖
        """
        if len(self._pool) <= k:
            return list(self._pool)

        # 按 lineage 聚类：优先选择 lineage 不同的
        selected: List[PopulationMember] = []
        remaining = list(self._pool)

        # 先选 fitness 最高的
        if remaining:
            best = max(remaining, key=lambda m: m.fitness_score)
            selected.append(best)
            remaining = [m for m in remaining if m.program.program_id not in best.lineage[-3:]]
            k -= 1

        # 再选 lineage 差异最大的
        while k > 0 and remaining:
            # 选择与已选成员 lineage 最不同的
            def lineage_distance(m: PopulationMember) -> float:
                distances = []
                for s in selected:
                    shared = set(m.lineage) & set(s.lineage)
                    distances.append(len(shared))
                return -min(distances) if distances else 0  # 越小越不同

            best = max(remaining, key=lineage_distance)
            selected.append(best)
            remaining = [m for m in remaining if m.program.program_id not in best.lineage[-3:]]
            k -= 1

        return selected

    def _enforce_diversity(self):
        """
        多样性强制执行：当种群血缘过于集中时，随机淘汰一些成员

        血缘过于集中的判断：top-3 members 共享超过 50% 的 lineage
        """
        if len(self._pool) < 5:
            return

        top_members = sorted(self._pool, key=lambda m: m.fitness_score, reverse=True)[:3]
        if len(top_members) < 2:
            return

        # 计算 top members 的 lineage 重叠度
        lineages = [set(m.lineage[-5:]) for m in top_members]  # 最近 5 代
        overlaps = []
        for i in range(len(lineages)):
            for j in range(i + 1, len(lineages)):
                overlap = len(lineages[i] & lineages[j]) / max(len(lineages[i] | lineages[j]), 1)
                overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0

        if avg_overlap > 0.5:
            # 血缘过于集中，随机替换 1-2 个低适应度成员
            low_members = sorted(self._pool, key=lambda m: m.fitness_score)[:2]
            for m in low_members[:1]:
                self._pool.remove(m)
                logger.info(f"Diversity enforcement: removed {m.program.name} (lineage overlap={avg_overlap:.2f})")

    # ── Population Management ──────────────────────────────────────────────────────────────────────────

    def add(self, member: PopulationMember) -> bool:
        """
        添加成员到种群

        如果已存在（program_id 相同），更新 fitness
        如果超过 max_size，触发 prune
        """
        with self._lock:
            existing = self._find_by_id(member.program.program_id)
            if existing is not None:
                # 更新已有成员
                idx = self._pool.index(existing)
                self._pool[idx] = member
                logger.debug(f"Updated existing member: {member.program.name}")
                return True

            self._pool.append(member)
            logger.info(f"Added member: {member.program.name} (fitness={member.fitness_score:.4f})")

            # 超载时修剪
            if len(self._pool) > self.max_size:
                self.prune()

            return True

    def add_from_program(self, program: MemoryProgram, parent_ids: Optional[List[str]] = None) -> PopulationMember:
        """从 MemoryProgram 创建并添加成员"""
        parent_ids = parent_ids or []
        member = PopulationMember.from_program(program)
        member.parent_ids = parent_ids

        if parent_ids:
            # 继承 lineage
            for pid in parent_ids:
                parent = self._find_by_id(pid)
                if parent:
                    member.lineage = parent.lineage + [program.program_id]
                    member.generation = parent.generation + 1
                    break

        self.add(member)
        return member

    def prune(self, keep_count: Optional[int] = None):
        """
        修剪种群，保留最优 + 随机多样性成员

        M* paper: 保持种群多样性，不只是保留 top-N
        """
        if keep_count is None:
            keep_count = self.min_size

        if len(self._pool) <= keep_count:
            return

        # 保留策略：60% top fitness，40% 随机多样性
        sorted_pool = sorted(self._pool, key=lambda m: m.fitness_score, reverse=True)
        keep_top = math.ceil(keep_count * 0.6)
        keep_random = keep_count - keep_top

        top_members = sorted_pool[:keep_top]
        remaining = sorted_pool[keep_top:]

        # 从 remaining 中随机选多样性样本
        random.shuffle(remaining)
        diverse_sample = remaining[:keep_random]

        survivors = top_members + diverse_sample
        removed = [m for m in self._pool if m not in survivors]

        self._pool = survivors
        self._generation_counter += 1

        logger.info(f"Pruned {len(removed)} members, kept {len(survivors)} "
                   f"(top={keep_top}, diverse={keep_random})")

    def _find_by_id(self, program_id: str) -> Optional[PopulationMember]:
        for m in self._pool:
            if m.program.program_id == program_id:
                return m
        return None

    def remove(self, program_id: str) -> bool:
        """移除成员"""
        with self._lock:
            for i, m in enumerate(self._pool):
                if m.program.program_id == program_id:
                    self._pool.pop(i)
                    return True
        return False

    # ── Persistence ──────────────────────────────────────────────────────────────────────────────────

    def save_to_db(self):
        """持久化到 FitnessTracker 数据库"""
        for member in self._pool:
            self.fitness_tracker._save_skill(member.program)

    def load_from_db(self, status: Optional[str] = "active") -> "PopulationPool":
        """从数据库加载种群"""
        programs = self.fitness_tracker.get_all_skills(status=status, limit=100)
        for program in programs:
            self.add_from_program(program)
        logger.info(f"Loaded {len(self._pool)} members from DB")
        return self

    # ── Crossover Support ──────────────────────────────────────────────────────────────────────────────

    def get_crossover_parents(self, k: int = 2) -> List[PopulationMember]:
        """
        获取 crossover 父代

        M* paper: crossover combines traits from multiple parents
        选 2-3 个血缘尽可能远的成员
        """
        return self.get_diverse_members(k=k)

    # ── Statistics ───────────────────────────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """种群统计"""
        if not self._pool:
            return {"size": 0, "avg_fitness": 0.0, "max_fitness": 0.0, "min_fitness": 0.0}

        fitnesses = [m.fitness_score for m in self._pool]
        return {
            "size": len(self._pool),
            "max_size": self.max_size,
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "max_fitness": max(fitnesses),
            "min_fitness": min(fitnesses),
            "tau": self.tau,
            "generation_counter": self._generation_counter,
            "fitness_variance": self._variance(fitnesses),
        }

    @staticmethod
    def _variance(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def __len__(self) -> int:
        return len(self._pool)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"PopulationPool(size={stats['size']}/{stats['max_size']}, "
                f"avg_fit={stats['avg_fitness']:.3f}, τ={self.tau})")
