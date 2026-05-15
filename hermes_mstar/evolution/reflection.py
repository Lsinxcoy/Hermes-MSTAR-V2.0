"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          Hermes MSTAR: MSTARReflector - 反思器                                                       ║
║                                                                                                          ║
║  MSTAR 的反思器，分析失败原因并提出改进建议                                                                     ║
║  功能:                                                                                                  ║
║    - 分析技能执行失败的原因                                                                                ║
║    - 提出针对性的变异建议                                                                                  ║
║    - 判断是否为 critical 情况                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..memory_program import MemoryProgram, MutationType

logger = logging.getLogger(__name__)


class FailurePattern(str, Enum):
    """失败模式"""
    TIMEOUT = "timeout"                      # 超时
    SYNTAX_ERROR = "syntax_error"            # 语法错误
    LOGIC_ERROR = "logic_error"             # 逻辑错误
    RUNTIME_ERROR = "runtime_error"         # 运行时错误
    QUALITY_LOW = "quality_low"              # 质量低
    CONTEXT_OVERFLOW = "context_overflow"    # 上下文溢出
    RATE_LIMIT = "rate_limit"               # 速率限制
    UNKNOWN = "unknown"                      # 未知原因


@dataclass
class MutationProposal:
    """变异建议"""
    mutation_type: MutationType
    priority: int  # 1-10
    reason: str
    expected_improvement: float  # 预期改善百分比
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionResult:
    """反思结果"""
    program_id: str
    program_name: str
    timestamp: str
    
    # 失败分析
    failure_patterns: List[FailurePattern]
    failure_count: int
    recent_errors: List[str]
    
    # 变异建议
    proposals: List[MutationProposal]
    
    # 决策
    needs_mutation: bool
    critical: bool  # 是否为 critical 情况 (需要立即变异)
    suggested_action: str  # "mutate", "archive", "merge", "keep"
    
    # 指标
    success: bool = False
    quality: float = 0.0
    latency: float = 0.0
    error: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def best_proposal(self) -> Optional[MutationProposal]:
        """获取最佳变异建议"""
        if not self.proposals:
            return None
        return max(self.proposals, key=lambda p: p.priority)


class MSTARReflector:
    """
    MSTAR 反思器
    
    分析技能的失败模式，提出变异建议。
    
    设计原则:
    1. 数据驱动 - 基于实际执行数据分析
    2. 针对性 - 每个失败模式有对应的变异策略
    3. 优先级 - 区分 critical 和普通情况
    """
    
    def __init__(self):
        """初始化反思器"""
        # 阈值配置
        self.critical_failure_rate = 0.7  # 失败率 > 70% 为 critical
        self.high_quality_threshold = 0.3  # 质量低于此值为低质量
        self.high_latency_seconds = 10.0   # 延迟高于此值为高延迟
        self.stale_days = 30              # 30天未使用为过时
        
        # 失败模式到变异类型的映射
        self.pattern_to_mutation = {
            FailurePattern.TIMEOUT: [
                (MutationType.GUIDANCE_MODIFY, 7),
                (MutationType.SCHEMA_FIELD_REMOVE, 5),
            ],
            FailurePattern.SYNTAX_ERROR: [
                (MutationType.GUIDANCE_MODIFY, 9),
                (MutationType.KEYWORD_REPLACE, 6),
            ],
            FailurePattern.LOGIC_ERROR: [
                (MutationType.LOGIC_READ_MODIFY, 8),
                (MutationType.GUIDANCE_MODIFY, 7),
            ],
            FailurePattern.RUNTIME_ERROR: [
                (MutationType.GUIDANCE_MODIFY, 8),
                (MutationType.SCHEMA_FIELD_REMOVE, 6),
            ],
            FailurePattern.QUALITY_LOW: [
                (MutationType.GUIDANCE_MODIFY, 8),
                (MutationType.KEYWORD_REPLACE, 6),
            ],
            FailurePattern.CONTEXT_OVERFLOW: [
                (MutationType.SCHEMA_FIELD_REMOVE, 8),
                (MutationType.GUIDANCE_MODIFY, 6),
            ],
            FailurePattern.RATE_LIMIT: [
                (MutationType.THRESHOLD_ADJUST, 7),
                (MutationType.GUIDANCE_MODIFY, 5),
            ],
            FailurePattern.UNKNOWN: [
                (MutationType.KEYWORD_ADD, 5),
                (MutationType.GUIDANCE_MODIFY, 4),
            ],
        }
    
    def analyze(
        self,
        program: MemoryProgram,
        success: bool,
        quality: float,
        latency: float,
        error: Optional[str] = None
    ) -> ReflectionResult:
        """
        分析程序并生成变异建议
        
        Args:
            program: 待分析的技能
            success: 执行是否成功
            quality: 输出质量评分 [0.0, 1.0]
            latency: 执行延迟 (秒)
            error: 错误信息 (如果有)
            
        Returns:
            ReflectionResult
        """
        recent_errors = [error] if error else []
        
        # 分析失败模式
        failure_patterns = self._analyze_failure_patterns(program, success, quality, latency, error)
        
        # 生成变异建议
        proposals = self._generate_proposals(program, failure_patterns)
        
        # 决策
        needs_mutation = self._should_mutate(program, failure_patterns)
        critical = self._is_critical(program, failure_patterns)
        suggested_action = self._determine_action(program, failure_patterns)
        
        return ReflectionResult(
            program_id=program.id,
            program_name=program.name,
            timestamp=datetime.now().isoformat(),
            failure_patterns=failure_patterns,
            failure_count=program.failure_count,
            recent_errors=recent_errors,
            proposals=proposals,
            needs_mutation=needs_mutation,
            critical=critical,
            suggested_action=suggested_action,
            success=success,
            quality=quality,
            latency=latency,
            error=error
        )
    
    def _analyze_failure_patterns(
        self,
        program: MemoryProgram,
        success: bool,
        quality: float,
        latency: float,
        error: Optional[str]
    ) -> List[FailurePattern]:
        """分析失败模式"""
        patterns = []
        
        # 如果有错误，首先分析错误类型
        if error:
            error_pattern = self.analyze_error(error)
            if error_pattern not in patterns:
                patterns.append(error_pattern)
        
        # 计算关键指标
        use_count = program.use_count or 1
        failure_rate = program.failure_count / use_count
        avg_quality = program.avg_quality if program.avg_quality > 0 else quality
        avg_latency = program.avg_latency if program.avg_latency > 0 else latency
        days_since_activity = self._days_since(program.last_activity_at)
        
        # 模式: 质量低
        if avg_quality < self.high_quality_threshold or quality < self.high_quality_threshold:
            if FailurePattern.QUALITY_LOW not in patterns:
                patterns.append(FailurePattern.QUALITY_LOW)
        
        # 模式: 延迟高
        if avg_latency > self.high_latency_seconds or latency > self.high_latency_seconds:
            if FailurePattern.TIMEOUT not in patterns:
                patterns.append(FailurePattern.TIMEOUT)
        
        # 模式: 技能过时
        if days_since_activity > self.stale_days:
            if FailurePattern.UNKNOWN not in patterns:
                # 时过时的技能标记为UNKNOWN模式
                patterns.append(FailurePattern.UNKNOWN)
        
        # 模式: 高失败率导致需要调整
        if failure_rate > 0.3:
            if FailurePattern.QUALITY_LOW not in patterns and FailurePattern.UNKNOWN not in patterns:
                patterns.append(FailurePattern.UNKNOWN)
        
        # 如果没有识别出明显模式，标记为未知
        if not patterns:
            patterns.append(FailurePattern.UNKNOWN)
        
        return patterns
    
    def _generate_proposals(
        self,
        program: MemoryProgram,
        patterns: List[FailurePattern]
    ) -> List[MutationProposal]:
        """根据失败模式生成变异建议"""
        proposals = []
        seen_types = set()
        
        for pattern in patterns:
            mutations = self.pattern_to_mutation.get(pattern, [])
            
            for mutation_type, base_priority in mutations:
                if mutation_type in seen_types:
                    continue
                
                # 计算预期改善
                expected = self._calculate_expected_improvement(
                    MutationProposal(
                        mutation_type=mutation_type,
                        priority=base_priority,
                        reason=f"Addresses {pattern.value}",
                        expected_improvement=0.0,
                        details={"pattern": pattern.value}
                    )
                )
                
                proposals.append(MutationProposal(
                    mutation_type=mutation_type,
                    priority=base_priority,
                    reason=f"Addresses {pattern.value}",
                    expected_improvement=expected,
                    details={"pattern": pattern.value}
                ))
                
                seen_types.add(mutation_type)
        
        # 按优先级排序
        proposals.sort(key=lambda p: p.priority, reverse=True)
        
        return proposals[:5]  # 最多返回5个建议
    
    def _calculate_expected_improvement(
        self,
        proposal: MutationProposal
    ) -> float:
        """计算预期改善"""
        base_improvement = {
            MutationType.KEYWORD_ADD: 0.15,
            MutationType.KEYWORD_REMOVE: 0.10,
            MutationType.KEYWORD_REPLACE: 0.12,
            MutationType.THRESHOLD_ADJUST: 0.20,
            MutationType.PRIORITY_ADJUST: 0.05,
            MutationType.GUIDANCE_MODIFY: 0.18,
            MutationType.SCHEMA_FIELD_REMOVE: 0.08,
            MutationType.LOGIC_READ_MODIFY: 0.15,
            MutationType.LOGIC_WRITE_MODIFY: 0.15,
            MutationType.LOGIC_QUERY_ADD: 0.10,
        }.get(proposal.mutation_type, 0.10)
        
        return base_improvement
    
    def _should_mutate(self, program: MemoryProgram, patterns: List[FailurePattern]) -> bool:
        """判断是否应该变异"""
        # 基础条件
        if program.pin:
            return False
        
        if program.use_count < 3:
            return False
        
        # 检查失败率
        use_count = program.use_count or 1
        failure_rate = program.failure_count / use_count
        
        if failure_rate > 0.3:
            return True
        
        # 检查是否有过时模式
        if FailurePattern.UNKNOWN in patterns:
            return True
        
        # 检查适应度
        if program.fitness_score < 0.3:
            return True
        
        return False
    
    def _is_critical(self, program: MemoryProgram, patterns: List[FailurePattern]) -> bool:
        """判断是否为 critical 情况"""
        # 极端失败率
        use_count = program.use_count or 1
        failure_rate = program.failure_count / use_count
        
        if failure_rate > self.critical_failure_rate:
            return True
        
        # 极端低质量
        if program.avg_quality < 0.2:
            return True
        
        # 极端延迟
        if program.avg_latency > 60.0:
            return True
        
        # Critical 错误类型
        critical_patterns = {FailurePattern.SYNTAX_ERROR, FailurePattern.RUNTIME_ERROR, FailurePattern.CONTEXT_OVERFLOW}
        if any(p in critical_patterns for p in patterns):
            return True
        
        return False
    
    def _determine_action(
        self,
        program: MemoryProgram,
        patterns: List[FailurePattern]
    ) -> str:
        """决定建议动作: 'mutate' | 'archive' | 'merge' | 'keep'"""
        # Critical 情况需要立即变异
        if self._is_critical(program, patterns):
            return "mutate"
        
        # 需要变异的情况
        if self._should_mutate(program, patterns):
            return "mutate"
        
        # 检查是否应该归档
        if self._days_since(program.last_activity_at) > 60:
            return "archive"
        
        # 检查是否应该合并 (相似技能可以合并)
        if program.is_umbrella or program.parent_skill_id:
            return "merge"
        
        # 否则保持现状
        return "keep"
    
    def _days_since(self, timestamp: str) -> int:
        """计算天数差"""
        if not timestamp:
            return 0
        try:
            dt = datetime.fromisoformat(timestamp)
            return (datetime.now() - dt).days
        except (ValueError, TypeError):
            return 0
    
    def analyze_error(self, error_str: str) -> FailurePattern:
        """
        分析单个错误并归类
        
        Args:
            error_str: 错误信息
            
        Returns:
            FailurePattern
        """
        error_lower = error_str.lower()
        
        # 超时相关
        if 'timeout' in error_lower or 'timed out' in error_lower or 'took too long' in error_lower:
            return FailurePattern.TIMEOUT
        
        # 语法错误
        if 'syntax' in error_lower or 'parse' in error_lower or 'invalid' in error_lower:
            return FailurePattern.SYNTAX_ERROR
        
        # 逻辑错误
        if 'logic' in error_lower or 'wrong' in error_lower or 'incorrect' in error_lower:
            return FailurePattern.LOGIC_ERROR
        
        # 运行时错误
        if 'runtime' in error_lower or 'exception' in error_lower or 'crash' in error_lower:
            return FailurePattern.RUNTIME_ERROR
        
        # 质量相关
        if 'quality' in error_lower or 'poor' in error_lower or 'bad result' in error_lower:
            return FailurePattern.QUALITY_LOW
        
        # 上下文溢出
        if 'context' in error_lower and ('overflow' in error_lower or 'limit' in error_lower or 'exceed' in error_lower):
            return FailurePattern.CONTEXT_OVERFLOW
        
        if 'context' in error_lower and ('length' in error_lower or 'too long' in error_lower):
            return FailurePattern.CONTEXT_OVERFLOW
        
        # 速率限制
        if 'rate limit' in error_lower or 'too many' in error_lower or '429' in error_lower:
            return FailurePattern.RATE_LIMIT
        
        # 上下文相关但不是溢出
        if 'context' in error_lower or 'missing' in error_lower or 'required' in error_lower:
            return FailurePattern.UNKNOWN
        
        return FailurePattern.UNKNOWN
    
    def get_best_proposal(self, result: ReflectionResult) -> Optional[MutationProposal]:
        """获取最佳变异建议"""
        return result.best_proposal
