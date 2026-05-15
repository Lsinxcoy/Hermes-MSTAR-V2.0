"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║              Hermes-MSTAR: Unified MemoryProgram Schema                               ║
║                                                                                      ║
║  融合 MSTAR (arxiv:2604.11811) 的结构化记忆格式 + Hermes-MSTAR 的增强适应度公式        ║
║                                                                                      ║
║  Components:                                                                         ║
║      1. Schema      - 定义存储什么                                                   ║
║      2. Logic       - 定义如何读写                                                   ║
║      3. Instructions - Agent 使用指导                                                ║
║      4. Evolution   - 进化元数据                                                     ║
║      5. QualityGates - 质量门禁                                                      ║
║      6. EpisodeResult - 单次执行结果 (用于适应度计算)                                  ║
║                                                                                      ║
║  增强适应度公式:                                                                       ║
║      fitness = (0.7 * success_rate + 0.3 * quality)                                  ║
║              * time_decay * (0.5 + 0.5 * confidence) * token_factor                 ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

核心概念:
    MemoryProgram 是 Hermes 记忆系统的统一格式
    每个程序包含:
        - Schema: 定义存储什么
        - Logic: 定义如何读写
        - Instructions: Agent 使用指导
        - Evolution: 进化元数据
        - EpisodeResult: 单次执行结果记录
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════════════════
#  Enums: Storage, Domain, Status, Fitness Level, Mutation Types
# ═══════════════════════════════════════════════════════════════════════════════════════

class StorageType(Enum):
    """记忆存储类型"""
    VECTOR       = "vector"       # 向量存储 (语义搜索)
    TEMPORAL     = "temporal"     # 时序存储 (对话历史)
    SEMANTIC     = "semantic"     # 语义存储 (概念关联)
    STRUCTURED   = "structured"   # 结构化存储 (键值对)
    PROCEDURAL   = "procedural"   # 程序化存储 (技能/本能)


class TaskDomain(Enum):
    """任务领域分类 (MSTAR 任务专用化)"""
    CODING       = "coding"       # 编程任务
    RESEARCH     = "research"     # 研究任务
    CREATIVE     = "creative"     # 创意任务
    ANALYSIS     = "analysis"     # 分析任务
    GENERAL      = "general"      # 通用任务
    SYSTEM       = "system"       # 系统任务


class ProgramStatus(Enum):
    """程序状态 (融合 Hermes 状态机)"""
    ACTIVE       = "active"       # 活跃使用
    EVALUATING   = "evaluating"  # 评估中
    DEPRECATED   = "deprecated"  # 已废弃
    EXPERIMENTAL = "experimental" # 实验性
    STALE        = "stale"        # 闲置
    ARCHIVED     = "archived"     # 已归档
    PINNED       = "pinned"       # 固定保护


class FitnessLevel(Enum):
    """适应度等级"""
    EXCELLENT    = "excellent"    # ≥0.8
    GOOD         = "good"         # ≥0.6
    AVERAGE      = "average"      # ≥0.4
    FAIR         = "fair"         # ≥0.4 (兼容旧命名)
    POOR         = "poor"         # <0.4
    FAILING      = "failing"      # <0.2


class MutationType(Enum):
    """
    14种变异类型 (MSTAR Evolution)
    
    Schema 变异:
        - SCHEMA_FIELD_ADD: 添加新字段
        - SCHEMA_FIELD_REMOVE: 删除字段
        - SCHEMA_FIELD_MODIFY: 修改字段
    
    Logic 变异:
        - LOGIC_READ_MODIFY: 修改读取逻辑
        - LOGIC_WRITE_MODIFY: 修改写入逻辑
        - LOGIC_QUERY_ADD: 添加查询条件
    
    Instructions 变异:
        - KEYWORD_ADD: 添加触发关键词
        - KEYWORD_REMOVE: 删除触发关键词
        - KEYWORD_REPLACE: 替换触发关键词
        - THRESHOLD_ADJUST: 调整阈值
        - PRIORITY_ADJUST: 调整优先级
        - GUIDANCE_MODIFY: 修改指导文本
    
    组合/高级变异:
        - CROSSOVER: 交叉重组
        - ENSEMBLE: 集成融合
        - RANDOM_CHANGE: 随机变异
    """
    # Schema 变异
    SCHEMA_FIELD_ADD = "schema_field_add"
    SCHEMA_FIELD_REMOVE = "schema_field_remove"
    SCHEMA_FIELD_MODIFY = "schema_field_modify"
    
    # Logic 变异
    LOGIC_READ_MODIFY = "logic_read_modify"
    LOGIC_WRITE_MODIFY = "logic_write_modify"
    LOGIC_QUERY_ADD = "logic_query_add"
    
    # Instructions 变异
    KEYWORD_ADD = "keyword_add"
    KEYWORD_REMOVE = "keyword_remove"
    KEYWORD_REPLACE = "keyword_replace"
    THRESHOLD_ADJUST = "threshold_adjust"
    PRIORITY_ADJUST = "priority_adjust"
    GUIDANCE_MODIFY = "guidance_modify"
    
    # 组合变异
    CROSSOVER = "crossover"
    ENSEMBLE = "ensemble"
    
    # 随机变异
    RANDOM_CHANGE = "random_change"


# ═══════════════════════════════════════════════════════════════════════════════════════
#  EpisodeResult: 单次执行结果 (来自 Hermes-MSTAR skill_program.py)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeResult:
    """
    单次执行结果
    
    用于记录每次记忆程序调用的结果，包含:
    - success: 任务是否成功
    - quality: 输出质量评分 [0.0, 1.0]
    - latency: 执行延迟 (秒)
    - confidence: 执行置信度 [0.0, 1.0]
    - tokens_consumed: Token 消耗数量 (RTK 效率计算)
    - error: 错误信息 (如果有)
    
    Example:
        result = EpisodeResult(
            success=True,
            quality=0.85,
            latency=1.2,
            confidence=0.9,
            tokens_consumed=500
        )
    """
    success: bool                          # 任务是否成功
    quality: float = 0.5                   # 输出质量 [0.0, 1.0]
    latency: float = 0.0                  # 执行延迟 (秒)
    confidence: float = 0.5               # 执行置信度 [0.0, 1.0]
    tokens_consumed: int = 0               # Token 消耗数量
    error: Optional[str] = None            # 错误信息
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "success": self.success,
            "quality": self.quality,
            "latency": self.latency,
            "confidence": self.confidence,
            "tokens_consumed": self.tokens_consumed,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeResult":
        """从字典反序列化"""
        return cls(
            success=data.get("success", False),
            quality=data.get("quality", 0.5),
            latency=data.get("latency", 0.0),
            confidence=data.get("confidence", 0.5),
            tokens_consumed=data.get("tokens_consumed", 0),
            error=data.get("error")
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
#  Schema: 定义存储什么 (MSTAR Memory Program Component 1)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Schema:
    """
    Schema: 定义存储什么 (MSTAR Memory Program Component 1)
    
    Example:
        schema = Schema(
            name="correction_storage",
            fields=[
                {"name": "uuid", "type": "str", "required": True},
                {"name": "wrong_answer", "type": "str", "required": True},
                {"name": "correct_answer", "type": "str", "required": True},
            ],
            storage_type=StorageType.STRUCTURED,
        )
    """
    name: str                              # Schema 名称
    description: str = ""                  # Schema 描述
    fields: list[dict] = field(default_factory=list)  # 字段定义
    storage_type: StorageType = StorageType.STRUCTURED
    domain: TaskDomain = TaskDomain.GENERAL
    parent_program: Optional[str] = None   # 父程序 (进化链)
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "fields": self.fields,
            "storage_type": self.storage_type.value,
            "domain": self.domain.value,
            "parent_program": self.parent_program,
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
#  Logic: 定义如何读写 (MSTAR Memory Program Component 2)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Logic:
    """
    Logic: 定义如何读写 (MSTAR Memory Program Component 2)
    
    Example:
        logic = Logic(
            read_template="SELECT * FROM corrections WHERE trigger LIKE '%{query}%'",
            write_template="INSERT INTO corrections VALUES ({uuid}, {wrong}, {correct})",
            query_fields=["trigger_keywords", "context"],
            response_template="Based on past mistakes: {correct_answer}",
        )
    """
    read_template: str = ""               # 读取模板
    write_template: str = ""               # 写入模板
    query_fields: list[str] = field(default_factory=list)  # 支持的查询字段
    response_template: str = ""            # 响应模板
    preconditions: list[str] = field(default_factory=list)  # 前置条件
    postconditions: list[str] = field(default_factory=list)  # 后置条件
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "read_template": self.read_template,
            "write_template": self.write_template,
            "query_fields": self.query_fields,
            "response_template": self.response_template,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
#  Instructions: Agent 使用指导 (MSTAR Memory Program Component 3)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Instructions:
    """
    Instructions: Agent 使用指导 (MSTAR Memory Program Component 3)
    
    Example:
        instructions = Instructions(
            trigger_keywords=["错误", "修复", "bug"],
            usage_examples=[
                "用户说: '我之前犯过这个错误'",
                "查询 correction_program, 返回 correct_answer",
            ],
            agent_guidance="When user triggers error keywords, consult this memory first.",
        )
    """
    trigger_keywords: list[str] = field(default_factory=list)  # 触发关键词
    usage_examples: list[str] = field(default_factory=list)    # 使用示例
    agent_guidance: str = ""                   # Agent 指导
    confidence_threshold: float = 0.5          # 置信度阈值
    priority: int = 0                          # 优先级 (更高更优先)
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "trigger_keywords": self.trigger_keywords,
            "usage_examples": self.usage_examples,
            "agent_guidance": self.agent_guidance,
            "confidence_threshold": self.confidence_threshold,
            "priority": self.priority,
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
#  Evolution: 进化元数据 (MSTAR Core)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Evolution:
    """
    Evolution: 进化元数据
    
    Tracks the evolutionary history of this memory program.
    包含适应度计算所需的完整统计信息。
    
    适应度公式 (增强版):
        fitness = (0.7 * success_rate + 0.3 * avg_quality)
                * time_decay * (0.5 + 0.5 * confidence) * token_factor
    """
    program_id: str                          # 唯一标识
    version: int = 1                         # 版本号
    parent_id: Optional[str] = None          # 父程序 ID
    root_id: Optional[str] = None            # 根程序 ID (溯源)
    created_at: str = ""                     # 创建时间
    updated_at: str = ""                     # 更新时间
    
    # ── 基础统计 ───────────────────────────────────────────────────────────────────
    fitness_score: float = 0.5               # 适应度分数 [0, 1]
    episode_count: int = 0                   # 评估次数
    success_count: int = 0                   # 成功次数
    failure_count: int = 0                   # 失败次数
    
    # ── 增强统计 (用于改进适应度公式) ───────────────────────────────────────────────
    avg_quality: float = 0.5                  # 平均质量
    avg_latency: float = 0.0                  # 平均延迟
    avg_tokens: float = 0.0                   # 平均Token消耗
    confidence: float = 0.0                   # 置信度 min(1.0, episodes/20)
    
    # ── 变异追踪 ───────────────────────────────────────────────────────────────────
    mutation_count: int = 0                   # 变异次数
    last_mutation: Optional[str] = None      # 上次变异类型
    lineage: list[str] = field(default_factory=list)  # 进化链
    mutation_history: list[dict] = field(default_factory=list)  # 变异历史
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "program_id": self.program_id,
            "version": self.version,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "fitness_score": self.fitness_score,
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_quality": self.avg_quality,
            "avg_latency": self.avg_latency,
            "avg_tokens": self.avg_tokens,
            "confidence": self.confidence,
            "mutation_count": self.mutation_count,
            "last_mutation": self.last_mutation,
            "lineage": self.lineage,
            "mutation_history": self.mutation_history,
        }
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.episode_count == 0:
            return 0.0
        return self.success_count / self.episode_count
    
    @property
    def failure_rate(self) -> float:
        """计算失败率"""
        if self.episode_count == 0:
            return 0.0
        return self.failure_count / self.episode_count
    
    @property
    def fitness_level(self) -> FitnessLevel:
        """获取适应度等级"""
        if self.fitness_score >= 0.8:
            return FitnessLevel.EXCELLENT
        elif self.fitness_score >= 0.6:
            return FitnessLevel.GOOD
        elif self.fitness_score >= 0.4:
            return FitnessLevel.AVERAGE
        elif self.fitness_score >= 0.2:
            return FitnessLevel.POOR
        else:
            return FitnessLevel.FAILING


# ═══════════════════════════════════════════════════════════════════════════════════════
#  QualityGates: 质量门禁状态
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityGates:
    """
    Quality Gates: 质量门禁状态
    
    Tracks the status of each quality gate.
    每个门禁状态: pending/passed/failed
    """
    compile_status: str = "pending"          # pending/passed/failed
    runtime_status: str = "pending"          # pending/passed/failed
    logic_status: str = "pending"            # pending/passed/failed
    quality_status: str = "pending"          # pending/passed/failed
    
    compile_errors: list[str] = field(default_factory=list)
    runtime_errors: list[str] = field(default_factory=list)
    logic_errors: list[str] = field(default_factory=list)
    quality_errors: list[str] = field(default_factory=list)
    
    last_check: str = ""                     # 上次检查时间
    
    def all_passed(self) -> bool:
        """所有门禁是否通过"""
        return all([
            self.compile_status == "passed",
            self.runtime_status == "passed",
            self.logic_status == "passed",
            self.quality_status == "passed",
        ])
    
    def any_failed(self) -> bool:
        """是否有任何门禁失败"""
        return any([
            self.compile_status == "failed",
            self.runtime_status == "failed",
            self.logic_status == "failed",
            self.quality_status == "failed",
        ])
    
    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "compile_status": self.compile_status,
            "runtime_status": self.runtime_status,
            "logic_status": self.logic_status,
            "quality_status": self.quality_status,
            "compile_errors": self.compile_errors,
            "runtime_errors": self.runtime_errors,
            "logic_errors": self.logic_errors,
            "quality_errors": self.quality_errors,
            "last_check": self.last_check,
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
#  MemoryProgram: 主类 (融合 MSTAR + Hermes-MSTAR 增强)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryProgram:
    """
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                              MemoryProgram                                          ║
    ║                                                                                      ║
    ║  基于 MSTAR (arxiv:2604.11811) 的统一记忆格式                                        ║
    ║  + Hermes-MSTAR 增强适应度公式:                                                       ║
    ║      fitness = (0.7 * success_rate + 0.3 * quality)                                   ║
    ║              * time_decay * (0.5 + 0.5 * confidence) * token_factor                   ║
    ║                                                                                      ║
    ║  Components:                                                                         ║
    ║      1. Schema     - 定义存储什么                                                    ║
    ║      2. Logic      - 定义如何读写                                                    ║
    ║      3. Instructions - Agent 使用指导                                                 ║
    ║      4. Evolution  - 进化元数据                                                      ║
    ║      5. Quality    - 质量门禁                                                        ║
    ║                                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    
    Example:
        program = MemoryProgram(
            program_id="correction_001",
            name="Self-Improving Correction Memory",
            description="Stores user corrections to avoid repeating mistakes",
            schema=Schema(
                name="correction",
                fields=[
                    {"name": "uuid", "type": "str", "required": True},
                    {"name": "wrong", "type": "str", "required": True},
                    {"name": "correct", "type": "str", "required": True},
                ],
                storage_type=StorageType.STRUCTURED,
            ),
            logic=Logic(
                read_template="Find correction for: {query}",
                write_template="Add correction: {uuid} -> {correct}",
            ),
            instructions=Instructions(
                trigger_keywords=["错误", "修复", "wrong"],
                agent_guidance="When user corrects you, store in this memory.",
            ),
            evolution=Evolution(program_id="correction_001"),
            quality=QualityGates(),
            source_system="self-improving",
            source_file="corrections.md",
        )
    """
    
    # ── 身份 ─────────────────────────────────────────────────────────────────────────
    program_id: str                          # 唯一标识
    name: str                                # 程序名称
    description: str = ""                    # 程序描述
    
    # ── MSTAR Components ─────────────────────────────────────────────────────────────
    schema: Schema = field(default_factory=lambda: Schema(name="default"))
    logic: Logic = field(default_factory=Logic)
    instructions: Instructions = field(default_factory=Instructions)
    evolution: Optional[Evolution] = None  # 延迟初始化，在 __post_init__ 中设置
    quality: QualityGates = field(default_factory=QualityGates)
    
    # ── 元数据 ─────────────────────────────────────────────────────────────────────
    source_system: str = ""                  # 来源系统
    source_file: str = ""                    # 来源文件
    status: ProgramStatus = ProgramStatus.ACTIVE
    version: int = 1                        # 版本号
    
    # ── 标签/组织 ─────────────────────────────────────────────────────────────────
    tags: list[str] = field(default_factory=list)
    category: str = ""                       # 分类
    domain: TaskDomain = TaskDomain.GENERAL  # 任务领域
    
    # ── Hermes 兼容字段 ─────────────────────────────────────────────────────────────
    pin: bool = False                        # 固定保护
    created_at: str = ""                     # 创建时间
    last_activity_at: str = ""               # 上次活动时间
    stale_at: str = ""                       # 闲置时间
    archived_at: str = ""                     # 归档时间
    
    # ── 伞形合并 ─────────────────────────────────────────────────────────────────
    parent_skill_id: Optional[str] = None    # 父技能 ID
    absorbed_into: Optional[str] = None     # 被吸收到
    is_umbrella: bool = False                # 是否为伞形
    children_ids: list[str] = field(default_factory=list)  # 子程序 ID 列表
    
    # ── 质量门禁 ─────────────────────────────────────────────────────────────────
    quality_gates_passed: bool = False       # 所有质量门禁是否通过
    
    def __post_init__(self):
        """初始化后处理"""
        # 生成 program_id 如果为空
        if not self.program_id:
            self.program_id = self._generate_id()

        # 初始化 Evolution 如果为空
        if self.evolution is None:
            self.evolution = Evolution(program_id=self.program_id)

        # instructions 可能是 dict，转换为 Instructions dataclass
        if isinstance(self.instructions, dict):
            self.instructions = Instructions(**self.instructions)
        elif self.instructions is None:
            self.instructions = Instructions(trigger_keywords=[])

        # 初始化时间戳
        now = datetime.now().isoformat()
        if not self.evolution.created_at:
            self.evolution.created_at = now
        if not self.evolution.updated_at:
            self.evolution.updated_at = now
        if not self.created_at:
            self.created_at = now
        if not self.last_activity_at:
            self.last_activity_at = now
    
    @property
    def fitness_score(self) -> float:
        """便利属性：直接从 evolution 获取 fitness"""
        return self.evolution.fitness_score if self.evolution else 0.0
    
    @fitness_score.setter
    def fitness_score(self, value: float):
        if self.evolution:
            self.evolution.fitness_score = value
        
        # 初始化 Evolution 如果为空
        if self.evolution is None:
            self.evolution = Evolution(program_id=self.program_id)
        
        # 初始化时间戳
        now = datetime.now().isoformat()
        if not self.evolution.created_at:
            self.evolution.created_at = now
        if not self.evolution.updated_at:
            self.evolution.updated_at = now
        if not self.created_at:
            self.created_at = now
        if not self.last_activity_at:
            self.last_activity_at = now
    
    def _generate_id(self) -> str:
        """生成唯一 ID"""
        content = f"{self.name}{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _days_since(self, timestamp: str) -> int:
        """计算天数差"""
        if not timestamp:
            return 0
        try:
            dt = datetime.fromisoformat(timestamp)
            return (datetime.now() - dt).days
        except (ValueError, TypeError):
            return 0

    def _calculate_token_factor(self, tokens_consumed: int, weight: float = -0.05) -> float:
        """
        M* Paper Phase 5: Task-Specific Token Efficiency Factor

        factor = 1 + weight * ln(1 + tokens)

        Args:
            tokens_consumed: Token 消耗数量
            weight: Token 权重系数（负数 = 惩罚）

        Returns:
            float: Token 效率因子
        """
        if weight == 0 or tokens_consumed <= 0:
            return 1.0
        import math
        return 1.0 + weight * math.log1p(tokens_consumed)

    def _calculate_time_decay(self, base_decay: float = 0.995) -> float:
        """
        M* Paper Phase 5: Task-Specific Time Decay

        decay = base_decay ^ hours_since_activity

        Args:
            base_decay: 每小时衰减系数

        Returns:
            float: 时间衰减因子 [0.0, 1.0]
        """
        if not self.last_activity_at:
            return 1.0
        try:
            last = datetime.fromisoformat(self.last_activity_at)
            hours = (datetime.now() - last).total_seconds() / 3600
            return pow(base_decay, max(0, hours))
        except Exception:
            return 1.0

    def _calculate_latency_factor(self, latency: float, weight: float = -0.1) -> float:
        """
        M* Paper Phase 5: Latency Penalty Factor

        factor = 1 + weight * ln(1 + latency)

        Args:
            latency: 延迟（秒）
            weight: Latency 权重系数（负数 = 惩罚）

        Returns:
            float: Latency 惩罚因子
        """
        if weight == 0 or latency <= 0:
            return 1.0
        import math
        return 1.0 + weight * math.log1p(latency)
    
    # ── 核心方法 ─────────────────────────────────────────────────────────────────────
    
    def trigger(self, query: str) -> dict:
        """
        检查是否应该触发此记忆程序
        
        Args:
            query: 用户查询
            
        Returns:
            dict: {"trigger": bool, "confidence": float, "matched_keywords": list}
        """
        query_lower = query.lower()
        matched = []
        
        for keyword in self.instructions.trigger_keywords:
            if keyword.lower() in query_lower:
                matched.append(keyword)
        
        # MSTAR 置信度公式
        if len(self.instructions.trigger_keywords) > 0:
            confidence = min(1.0, len(matched) / (len(self.instructions.trigger_keywords) * 0.3))
        else:
            confidence = 0.0
        
        return {
            "trigger": confidence >= self.instructions.confidence_threshold,
            "confidence": confidence,
            "matched_keywords": matched,
            "priority": self.instructions.priority,
        }
    
    def update_fitness(self, episode_result: EpisodeResult) -> float:
        """
        更新适应度分数 (增强版 MSTAR fitness evaluation)

        M* Paper Phase 5: Task-Specific Fitness

        基础公式:
            fitness = (w_success * success_rate + w_quality * quality)
                     * time_decay * conf_factor * token_factor * latency_factor

        Args:
            episode_result: 单次执行结果

        Returns:
            更新后的 fitness_score
        """
        # ── Task Domain Detection ──────────────────────────────────────────────────
        domain = self._detect_task_domain()
        weights = self._get_fitness_weights(domain)

        # ── 更新统计 ──────────────────────────────────────────────────────────────
        self.evolution.episode_count += 1

        if episode_result.success:
            self.evolution.success_count += 1
        else:
            self.evolution.failure_count += 1

        # ── 计算各因子 ────────────────────────────────────────────────────────────

        # 1. 成功率 + 质量 (M* Paper Phase 5: task-specific weights)
        success_rate = self.evolution.success_count / self.evolution.episode_count
        base = weights.success * success_rate + weights.quality * episode_result.quality

        # 2. 时间衰减 (Hermes 活跃度)
        decay = self._calculate_time_decay(weights.time_decay)

        # 3. 置信度 (MSTAR: 样本量置信)
        self.evolution.confidence = min(1.0, self.evolution.episode_count / 20)
        conf_factor = 0.5 + 0.5 * (self.evolution.confidence / weights.confidence_max)

        # 4. Token 效率因子 (RTK 增强)
        token_factor = self._calculate_token_factor(
            episode_result.tokens_consumed, weights.token_weight
        )

        # 5. Latency 惩罚 (M* Paper Phase 5)
        latency_factor = self._calculate_latency_factor(
            episode_result.latency, weights.latency_weight
        )

        # ── 最终适应度计算 ────────────────────────────────────────────────────────
        self.evolution.fitness_score = base * decay * conf_factor * token_factor * latency_factor

        # ── 更新统计指标 ─────────────────────────────────────────────────────────
        n = self.evolution.episode_count
        self.evolution.avg_quality = (
            (self.evolution.avg_quality * (n - 1) + episode_result.quality) / n
        )
        self.evolution.avg_latency = (
            (self.evolution.avg_latency * (n - 1) + episode_result.latency) / n
        )
        if episode_result.tokens_consumed > 0:
            self.evolution.avg_tokens = (
                (self.evolution.avg_tokens * (n - 1) + episode_result.tokens_consumed) / n
            )

        # ── 更新时间 ─────────────────────────────────────────────────────────────
        self.evolution.updated_at = datetime.now().isoformat()
        self.last_activity_at = self.evolution.updated_at

        # ── Hermes 状态机更新 ────────────────────────────────────────────────────
        self._update_status()

        return self.evolution.fitness_score

    def _detect_task_domain(self):
        """检测任务域"""
        try:
            from ..evolution.task_domain import detect_domain
            keywords = self.instructions.trigger_keywords if hasattr(self, 'instructions') else []
            return detect_domain(keywords)
        except Exception:
            from ..evolution.task_domain import TaskDomain
            return TaskDomain.GENERAL

    def _get_fitness_weights(self, domain):
        """获取任务域对应的 fitness 权重"""
        try:
            from ..evolution.task_domain import FitnessWeights
            return FitnessWeights.for_domain(domain)
        except Exception:
            from ..evolution.task_domain import FitnessWeights
            return FitnessWeights()

    def _calculate_time_decay(self, base_decay: float = 0.995) -> float:
        """计算时间衰减因子"""
        if not self.last_activity_at:
            return 1.0
        try:
            from datetime import datetime
            last = datetime.fromisoformat(self.last_activity_at)
            hours = (datetime.now() - last).total_seconds() / 3600
            return pow(base_decay, max(0, hours))
        except Exception:
            return 1.0

    def _calculate_token_factor(self, tokens: int, weight: float = -0.05) -> float:
        """计算 Token 效率因子"""
        if weight == 0 or tokens <= 0:
            return 1.0
        import math
        return 1.0 + weight * math.log1p(tokens)

    def _calculate_latency_factor(self, latency: float, weight: float = -0.1) -> float:
        """计算 Latency 惩罚因子"""
        if weight == 0 or latency <= 0:
            return 1.0
        import math
        return 1.0 + weight * math.log1p(latency)
    
    def _update_status(self):
        """
        Hermes 状态机更新
        
        状态转换:
            ACTIVE -> STALE (30天无活动)
            STALE -> ACTIVE (重新活跃)
            STALE -> ARCHIVED (90天无活动)
        """
        if self.pin:
            return  # pinned 不更新状态
        
        days = self._days_since(self.last_activity_at)
        
        if self.status == ProgramStatus.ACTIVE and days > 30:
            self.status = ProgramStatus.STALE
            self.stale_at = datetime.now().isoformat()
        elif self.status == ProgramStatus.STALE and days > 90:
            self.status = ProgramStatus.ARCHIVED
            self.archived_at = datetime.now().isoformat()
        elif self.status == ProgramStatus.STALE and days <= 30:
            # 重新激活
            self.status = ProgramStatus.ACTIVE
            self.stale_at = ""
    
    def evolve(self, parent_id: str, mutation_type: MutationType) -> "MemoryProgram":
        """
        创建进化后的子程序 (MSTAR 变异)
        
        Args:
            parent_id: 父程序 ID
            mutation_type: 变异类型
            
        Returns:
            MemoryProgram: 新的子程序
        """
        child_data = self.to_dict()
        
        # 生成新 ID
        new_id = str(uuid.uuid4())
        
        # 更新进化信息
        child_data["program_id"] = new_id
        child_data["version"] = self.evolution.version + 1
        child_data["evolution"] = {
            "program_id": new_id,
            "version": self.evolution.version + 1,
            "parent_id": parent_id,
            "root_id": self.evolution.root_id or parent_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "fitness_score": 0.5,  # 新版本重置为中性
            "episode_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "avg_quality": 0.5,
            "avg_latency": 0.0,
            "avg_tokens": 0.0,
            "confidence": 0.0,
            "mutation_count": self.evolution.mutation_count + 1,
            "last_mutation": mutation_type.value,
            "lineage": self.evolution.lineage + [parent_id],
            "mutation_history": self.evolution.mutation_history + [
                {"type": mutation_type.value, "parent_id": parent_id, "timestamp": datetime.now().isoformat()}
            ],
        }
        
        child_data["status"] = ProgramStatus.EVALUATING
        child_data["source_system"] = "mstar-evolution"
        child_data["quality_gates_passed"] = False
        
        # 重置质量门禁
        child_data["quality"] = {
            "compile_status": "pending",
            "runtime_status": "pending",
            "logic_status": "pending",
            "quality_status": "pending",
            "compile_errors": [],
            "runtime_errors": [],
            "logic_errors": [],
            "quality_errors": [],
            "last_check": "",
        }
        
        return MemoryProgram.from_dict(child_data)
    
    def should_mutate(self, threshold: float = 0.3) -> bool:
        """
        检查是否应该触发变异
        
        条件:
            1. 适应度低于阈值
            2. 至少有 3 次评估
            3. 未被固定
        """
        if self.pin:
            return False
        return self.evolution.fitness_score < threshold and self.evolution.episode_count >= 3
    
    def should_forget(self, threshold: float = 0.2) -> bool:
        """
        检查是否应该遗忘
        
        条件:
            1. 适应度低于阈值
            2. 未被固定
        """
        if self.pin:
            return False
        return self.evolution.fitness_score < threshold
    
    def mark_pinned(self):
        """标记为固定 (Hermes 保护机制)"""
        self.pin = True
        self.status = ProgramStatus.PINNED
    
    def unpin(self):
        """取消固定"""
        self.pin = False
        self.status = ProgramStatus.ACTIVE
    
    def archive(self):
        """归档 (可恢复)"""
        self.status = ProgramStatus.ARCHIVED
        self.archived_at = datetime.now().isoformat()
    
    def activate(self):
        """激活"""
        self.status = ProgramStatus.ACTIVE
        self.stale_at = ""
        self.last_activity_at = datetime.now().isoformat()
    
    def is_stale(self, days: int = 30) -> bool:
        """检查是否闲置"""
        return self._days_since(self.last_activity_at) > days
    
    def should_archive(self, days: int = 90) -> bool:
        """检查是否应该归档"""
        if self.pin:
            return False
        return self._days_since(self.last_activity_at) > days
    
    def get_fitness_level(self) -> FitnessLevel:
        """获取适应度等级"""
        return self.evolution.fitness_level
    
    # ── 序列化/反序列化 ─────────────────────────────────────────────────────────────
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "program_id": self.program_id,
            "name": self.name,
            "description": self.description,
            "schema": self.schema.to_dict(),
            "logic": self.logic.to_dict(),
            "instructions": self.instructions.to_dict(),
            "evolution": self.evolution.to_dict(),
            "quality": self.quality.to_dict(),
            "source_system": self.source_system,
            "source_file": self.source_file,
            "status": self.status.value,
            "version": self.version,
            "tags": self.tags,
            "category": self.category,
            "domain": self.domain.value,
            "pin": self.pin,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "stale_at": self.stale_at,
            "archived_at": self.archived_at,
            "parent_skill_id": self.parent_skill_id,
            "absorbed_into": self.absorbed_into,
            "is_umbrella": self.is_umbrella,
            "children_ids": self.children_ids,
            "quality_gates_passed": self.quality_gates_passed,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryProgram":
        """从字典创建"""
        # 解析枚举
        if "storage_type" in data.get("schema", {}):
            data["schema"]["storage_type"] = StorageType(data["schema"]["storage_type"])
        if "domain" in data.get("schema", {}):
            data["schema"]["domain"] = TaskDomain(data["schema"]["domain"])
        if "status" in data:
            data["status"] = ProgramStatus(data["status"])
        if "domain" in data:
            data["domain"] = TaskDomain(data.get("domain", "general"))
        
        # 处理 evolution
        evolution_data = data.get("evolution", {})
        evolution_data["program_id"] = evolution_data.get("program_id", data["program_id"])
        
        # 处理 quality
        quality_data = data.get("quality", {})
        
        return cls(
            program_id=data["program_id"],
            name=data["name"],
            description=data.get("description", ""),
            schema=Schema(**data.get("schema", {})),
            logic=Logic(**data.get("logic", {})),
            instructions=Instructions(**data.get("instructions", {})),
            evolution=Evolution(**evolution_data),
            quality=QualityGates(**quality_data),
            source_system=data.get("source_system", ""),
            source_file=data.get("source_file", ""),
            status=data.get("status", ProgramStatus.ACTIVE),
            version=data.get("version", 1),
            tags=data.get("tags", []),
            category=data.get("category", ""),
            domain=data.get("domain", TaskDomain.GENERAL),
            pin=data.get("pin", False),
            created_at=data.get("created_at", ""),
            last_activity_at=data.get("last_activity_at", ""),
            stale_at=data.get("stale_at", ""),
            archived_at=data.get("archived_at", ""),
            parent_skill_id=data.get("parent_skill_id"),
            absorbed_into=data.get("absorbed_into"),
            is_umbrella=data.get("is_umbrella", False),
            children_ids=data.get("children_ids", []),
            quality_gates_passed=data.get("quality_gates_passed", False),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "MemoryProgram":
        """从 JSON 创建"""
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        return (
            f"MemoryProgram(id={self.program_id}, name={self.name}, "
            f"fitness={self.evolution.fitness_score:.3f}, "
            f"status={self.status.value})"
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
#  便捷函数
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_memory_program(
    name: str,
    description: str = "",
    trigger_keywords: list[str] = None,
    storage_type: StorageType = StorageType.STRUCTURED,
    domain: TaskDomain = TaskDomain.GENERAL,
    tags: list[str] = None,
    source_system: str = "agent",
) -> MemoryProgram:
    """
    创建新的 MemoryProgram 实例
    
    Args:
        name: 程序名称
        description: 描述
        trigger_keywords: 触发关键词列表
        storage_type: 存储类型
        domain: 任务领域
        tags: 标签
        source_system: 来源系统
        
    Returns:
        MemoryProgram 实例
    """
    if tags is None:
        tags = []
    if trigger_keywords is None:
        trigger_keywords = []
    
    program = MemoryProgram(
        program_id="",  # 自动生成
        name=name,
        description=description,
        schema=Schema(
            name=name.lower().replace(" ", "_"),
            storage_type=storage_type,
            domain=domain,
        ),
        instructions=Instructions(
            trigger_keywords=trigger_keywords,
        ),
        tags=tags,
        source_system=source_system,
    )
    
    return program


def calculate_fitness(
    success_rate: float,
    quality: float,
    time_decay: float,
    confidence: float,
    token_factor: float,
) -> float:
    """
    计算适应度分数 (独立函数版本)
    
    公式:
        fitness = (0.7 * success_rate + 0.3 * quality)
                * time_decay * (0.5 + 0.5 * confidence) * token_factor
    
    Args:
        success_rate: 成功率 [0.0, 1.0]
        quality: 质量评分 [0.0, 1.0]
        time_decay: 时间衰减因子 [0.5, 1.0]
        confidence: 置信度 [0.0, 1.0]
        token_factor: Token效率因子 [0.5, 1.5]
        
    Returns:
        适应度分数 [0.0, 1.0]
    """
    mstar_base = 0.7 * success_rate + 0.3 * quality
    confidence_factor = 0.5 + 0.5 * confidence
    return mstar_base * time_decay * confidence_factor * token_factor


# ── Global Integration ─────────────────────────────────────────────────────────

_hermes_provider: Optional[Any] = None


def init_mstar_integration(hermes_home: Optional[str] = None) -> Any:
    """
    初始化 Hermes MSTAR 集成
    Returns: HermesMSTARProvider 实例
    """
    global _hermes_provider
    if _hermes_provider is None:
        try:
            from .hermes_provider import HermesMSTARProvider
            _hermes_provider = HermesMSTARProvider(hermes_home)
            _hermes_provider.initialize()
        except Exception as e:
            import logging
            logging.getLogger("hermes.mstar").error(f"Failed to init MSTAR: {e}")
            return None
    return _hermes_provider


def get_mstar_integration() -> Optional[Any]:
    """获取已初始化的集成实例"""
    return _hermes_provider
