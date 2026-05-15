"""
EvolutionEngine — 自适应进化引擎
Ported from hermes-mstar agent/evolution_engine.py

自适应间隔: 3-50 sessions，根据成功率动态调整

M* Paper Upgrade (Phase 1):
  - PopulationPool + Softmax Selection (τ=0.15)
  - 种群搜索 vs 线性搜索: +44% fitness (0.459 vs 0.318)
"""
import json
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

from .population_pool import PopulationPool, SelectionStrategy

if TYPE_CHECKING:
    from .fitness_tracker import FitnessTracker

logger = logging.getLogger("hermes.mstar.evolution_engine")


class EvolutionEngine:
    """
    自适应进化引擎

    核心功能:
    - 评估每个 session 的 fitness
    - 根据成功率动态调整进化间隔
    - 选择候选 program 进行变异
    - 协调 mutator + reflector + quality_gates
    - 触发 archive/merge/delete 决策
    """

    _instance: Optional['EvolutionEngine'] = None
    _lock = threading.Lock()

    def __init__(
        self,
        fitness_tracker: 'FitnessTracker',
        archive_dir: str = "",
        backup_dir: str = "",
    ):
        self.fitness_tracker = fitness_tracker
        self.archive_dir = archive_dir
        self.backup_dir = backup_dir

        # 进化状态
        self._enabled = True
        self.cycles_run = 0
        self.mutations_applied = 0
        self.mutations_rejected = 0
        self.programs_archived = 0
        self.programs_deleted = 0
        self.programs_merged = 0

        # 自适应间隔
        self._min_interval = 3    # 最少 3 sessions
        self._max_interval = 50   # 最多 50 sessions
        self._current_interval = 10  # 当前间隔
        self._sessions_since_evolution = 0
        self._success_rates: List[float] = []

        # Mutator & Quality Gates
        self._mutator = None
        self._reflector = None
        self._quality_gates = None
        self._forgetting = None

        # Thread safety
        self._engine_lock = threading.RLock()
        self._last_cycle_time = 0.0

        # Phase 1: Population Pool (M* Paper)
        self._population_pool: Optional['PopulationPool'] = None

    @property
    def population_pool(self) -> 'PopulationPool':
        """Lazy-load population pool"""
        if self._population_pool is None:
            from .population_pool import PopulationPool
            self._population_pool = PopulationPool(
                fitness_tracker=self.fitness_tracker,
                max_size=20,
                min_size=5,
                tau=0.15,  # M* paper temperature
            )
            # 从 DB 加载已有 programs
            self._population_pool.load_from_db(status="active")
        return self._population_pool

    @classmethod
    def init(cls, fitness_tracker, archive_dir="", backup_dir=""):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(fitness_tracker, archive_dir, backup_dir)
        return cls._instance

    @classmethod
    def instance(cls) -> Optional['EvolutionEngine']:
        return cls._instance

    # ── Core evaluation ────────────────────────────────────────────────────────

    def evaluate_session(
        self,
        session_id: str,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        评估 session 并决定是否触发进化
        Returns: evolution result summary
        """
        with self._engine_lock:
            if not self._enabled:
                return {"enabled": False}

            self._sessions_since_evolution += 1
            success_rate = stats.get("success_rate", 0.0)
            self._success_rates.append(success_rate)
            if len(self._success_rates) > 10:
                self._success_rates = self._success_rates[-10:]

            # 自适应调整间隔
            avg_success = sum(self._success_rates) / len(self._success_rates)
            if avg_success > 0.8:
                self._current_interval = min(self._max_interval, self._current_interval + 5)
            elif avg_success < 0.4:
                self._current_interval = max(self._min_interval, self._current_interval - 3)

            result = {
                "session_id": session_id,
                "sessions_since_evolution": self._sessions_since_evolution,
                "current_interval": self._current_interval,
                "avg_success_rate": round(avg_success, 3),
                "evolution_triggered": False,
            }

            # 检查是否需要进化
            if self._sessions_since_evolution >= self._current_interval:
                self._sessions_since_evolution = 0
                try:
                    evo_result = self._run_evolution_cycle()
                    result.update(evo_result)
                    result["evolution_triggered"] = True
                except Exception as e:
                    logger.error(f"Evolution cycle failed: {e}")
                    result["error"] = str(e)

            return result

    def _run_evolution_cycle(self) -> Dict[str, Any]:
        """
        运行一个进化周期 (M* Paper Population-based Search)

        替换旧逻辑：不再只选 fitness<0.5 的单 program
        新逻辑：用 PopulationPool + Softmax Selection 选择候选
        """
        self.cycles_run += 1
        self._last_cycle_time = time.time()

        result = {
            "cycles_run": self.cycles_run,
            "mutations_attempted": 0,
            "mutations_applied": 0,
            "mutations_rejected": 0,
            "programs_archived": 0,
            "programs_merged": 0,
            "programs_deleted": 0,
            "evolved_programs": [],
            "population_size": len(self.population_pool),
            "population_avg_fitness": self.population_pool.get_statistics().get("avg_fitness", 0),
        }

        # M* Paper: 如果种群太小（<3），退化为线性搜索
        if len(self.population_pool) < 3:
            logger.info(f"Population too small ({len(self.population_pool)}), using fallback selection")
            candidates = self.fitness_tracker.get_all_skills(status="active", limit=10)
            low_fitness = sorted(candidates, key=lambda p: p.fitness_score)[:3]
            to_evolve = low_fitness
            crossover_parents = []
        else:
            # M* Paper Softmax Selection: 选择 k=3 个候选
            selected_members = self.population_pool.select(k=3, strategy=SelectionStrategy.SOFTMAX)
            if not selected_members:
                logger.info("Softmax selection returned no candidates, using fallback")
                candidates = self.fitness_tracker.get_all_skills(status="active", limit=10)
                to_evolve = sorted(candidates, key=lambda p: p.fitness_score)[:3]
                crossover_parents = []
            else:
                to_evolve = [m.program for m in selected_members]

                # Crossover 父代：选血缘尽可能远的成员
                crossover_parents = [m.program for m in self.population_pool.get_diverse_members(k=2)]

        # 执行变异
        for program in to_evolve:
            evo_result = self._evolve_program(program, crossover_parents=crossover_parents)
            if evo_result:
                result["mutations_applied"] += 1
                result["evolved_programs"].append(evo_result)
                self.mutations_applied += 1

                # 添加到种群池
                evolved_member = self.population_pool.add_from_program(
                    program=self.fitness_tracker.get_skill(evo_result["new_program_id"]) or
                            program,  # fallback
                    parent_ids=[program.program_id]
                )
            else:
                result["mutations_rejected"] += 1
                self.mutations_rejected += 1

        # 多样性强制执行
        self.population_pool._enforce_diversity()

        # 评估遗忘
        all_members = [m.program for m in self.population_pool._pool]
        self._evaluate_forgetting(all_members, result)

        # 持久化种群
        self.population_pool.save_to_db()

        return result

    def _evolve_program(self, program, crossover_parents: Optional[List] = None) -> Optional[Dict[str, Any]]:
        """对单个 program 进行变异"""
        try:
            from .mutator import MSTARMutator, MutationResult
            from .quality_gates import QualityGates
            from ..memory_program import MutationType

            mutator = self._get_mutator() or MSTARMutator()
            gates = self._get_quality_gates() or QualityGates()

            old_fitness = program.fitness_score

            # 选择变异类型
            mutation_type = mutator._select_mutation_type(program)

            # 执行变异（Phase 1: 支持 crossover_parents）
            parent_program = None
            if mutation_type == MutationType.CROSSOVER and crossover_parents:
                parent_program = crossover_parents[0] if crossover_parents else None

            mutation_result = mutator.mutate(program, mutation_type, parent_program=parent_program)

            if not mutation_result.success:
                return None

            mutated = mutation_result.program

            # 质量门检查
            gate_report = gates.run_all(mutated)

            if not gate_report.all_passed and not gate_report.passed:
                logger.debug(f"Quality gates failed for {program.program_id}")
                return None

            # 保存变异后的 program
            self.fitness_tracker._save_skill(mutated)

            # 立即激活变异子代 — MSTARMutator 默认设为 EVALUATING,
            # 但通过质量门后应该进入 ACTIVE 状态接受真实执行反馈
            from hermes_mstar.memory_program import ProgramStatus
            mutated.status = ProgramStatus.ACTIVE
            self.fitness_tracker._save_skill(mutated)

            # 使 RTK 缓存失效 (跳过如果 RTK 不可用)
            try:
                from hermes_mstar.rtk.rtk_optimizer import get_rtk_optimizer
                rtk = get_rtk_optimizer()
                rtk.invalidate_skill(program.program_id)
            except Exception:
                pass  # RTK not available, skip cache invalidation

            new_fitness = mutated.fitness_score
            improvement = new_fitness - old_fitness

            logger.info(f"Evolved {program.program_id}: {old_fitness:.3f} -> {new_fitness:.3f} "
                       f"(+{improvement:.3f}, type={mutation_type})")

            # 记录变异事件到 DB
            self.fitness_tracker.record_mutation(
                parent_id=program.program_id,
                child_id=mutated.program_id,
                mutation_type=mutation_type.value if hasattr(mutation_type, 'value') else str(mutation_type),
                fitness_before=old_fitness,
                fitness_after=new_fitness,
                details={"improvement": improvement}
            )

            return {
                "program_id": program.program_id,
                "new_program_id": mutated.program_id,
                "old_fitness": old_fitness,
                "new_fitness": new_fitness,
                "improvement": improvement,
                "mutation_type": mutation_type,
            }
        except Exception as e:
            logger.error(f"_evolve_program failed: {e}")
            return None

    def _evaluate_forgetting(self, programs: List, result: Dict[str, Any]):
        """评估遗忘决策"""
        try:
            forgetting = self._get_forgetting()
            if not forgetting:
                return

            candidates = forgetting.evaluate_batch(programs)
            for candidate in candidates:
                decision = forgetting.decide(candidate)
                if decision == "archive":
                    forgetting.archive(candidate.program)
                    result["programs_archived"] += 1
                    self.programs_archived += 1
                elif decision == "merge":
                    forgetting.merge(candidate.program)
                    result["programs_merged"] += 1
                    self.programs_merged += 1
                elif decision == "delete":
                    forgetting.delete(candidate.program)
                    result["programs_deleted"] += 1
                    self.programs_deleted += 1
        except Exception as e:
            logger.debug(f"Forgetting evaluation failed: {e}")

    def _get_mutator(self):
        if self._mutator is None:
            try:
                from .mutator import MSTARMutator
                self._mutator = MSTARMutator()
            except:
                pass
        return self._mutator

    def _get_quality_gates(self):
        if self._quality_gates is None:
            try:
                from .quality_gates import QualityGates
                self._quality_gates = QualityGates()
            except:
                pass
        return self._quality_gates

    def _get_forgetting(self):
        if self._forgetting is None:
            try:
                from .forgetting import ForgettingMechanism
                self._forgetting = ForgettingMechanism(self.archive_dir)
            except:
                pass
        return self._forgetting

    # ── Public API ────────────────────────────────────────────────────────────

    def trigger_manual_evolution(self) -> Dict[str, Any]:
        """手动触发进化"""
        with self._engine_lock:
            logger.info("Manual evolution triggered")
            return self._run_evolution_cycle()

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "cycles_run": self.cycles_run,
            "mutations_applied": self.mutations_applied,
            "mutations_rejected": self.mutations_rejected,
            "programs_archived": self.programs_archived,
            "programs_deleted": self.programs_deleted,
            "programs_merged": self.programs_merged,
            "current_interval": self._current_interval,
            "sessions_since_evolution": self._sessions_since_evolution,
            "avg_recent_success": round(
                sum(self._success_rates) / len(self._success_rates), 3
            ) if self._success_rates else 0,
        }
