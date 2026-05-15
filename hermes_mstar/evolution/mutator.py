"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          Hermes MSTAR: MSTARMutator - 15种变异操作                                                  ║
║                                                                                                          ║
║  MSTAR 的变异器，实现 15 种变异操作                                                                            ║
║  分类:                                                                                                    ║
║    Schema 变异 (3): field_add, field_remove, field_modify                                                 ║
║    Logic 变异 (3): read_modify, write_modify, query_add                                                  ║
║    Instructions 变异 (7): keyword_add/remove/replace, threshold/priority/guidance, crossover, ensemble    ║
║    随机变异 (1): random_change                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import copy
import logging
import random
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .memory_program import MemoryProgram, MutationType

logger = logging.getLogger(__name__)


class MutationResult:
    """
    变异结果

    记录变异操作的输出
    """
    success: bool = True  # M* Paper Phase 4: Bug Fix —evolution_engine checks this

    def __init__(
        self,
        program: MemoryProgram,
        mutation_type: MutationType,
        description: str,
        changed_fields: List[str]
    ):
        self.program = program
        self.mutation_type = mutation_type
        self.description = description
        self.changed_fields = changed_fields

    def __repr__(self) -> str:
        return f"MutationResult({self.mutation_type.value}, {self.changed_fields})"


class MSTARMutator:
    """
    MSTAR 变异器
    
    实现 15 种变异操作，分为 4 类:
    1. Schema 变异 - 修改程序的结构定义
    2. Logic 变异 - 修改程序的逻辑流程
    3. Instructions 变异 - 修改触发指令和配置
    4. 组合变异 - 多个程序的组合
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        初始化变异器
        
        Args:
            random_seed: 随机种子，用于可重现性
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # 变异概率配置
        self.mutation_probabilities = {
            MutationType.SCHEMA_FIELD_ADD: 0.10,
            MutationType.SCHEMA_FIELD_REMOVE: 0.05,
            MutationType.SCHEMA_FIELD_MODIFY: 0.10,
            MutationType.LOGIC_READ_MODIFY: 0.15,
            MutationType.LOGIC_WRITE_MODIFY: 0.15,
            MutationType.LOGIC_QUERY_ADD: 0.10,
            MutationType.KEYWORD_ADD: 0.15,
            MutationType.KEYWORD_REMOVE: 0.10,
            MutationType.KEYWORD_REPLACE: 0.15,
            MutationType.THRESHOLD_ADJUST: 0.05,
            MutationType.PRIORITY_ADJUST: 0.05,
            MutationType.GUIDANCE_MODIFY: 0.10,
            MutationType.CROSSOVER: 0.05,
            MutationType.ENSEMBLE: 0.05,
            MutationType.RANDOM_CHANGE: 0.10,
        }
        
        # 默认权重
        self.default_weights = {
            MutationType.KEYWORD_ADD: 1.0,
            MutationType.KEYWORD_REMOVE: 0.8,
            MutationType.KEYWORD_REPLACE: 1.2,
            MutationType.THRESHOLD_ADJUST: 0.5,
            MutationType.PRIORITY_ADJUST: 0.5,
            MutationType.GUIDANCE_MODIFY: 0.7,
        }
        
        # 变异统计
        self._mutation_counts = {mt.value: 0 for mt in MutationType}
    
    def mutate(
        self,
        program: MemoryProgram,
        mutation_type: Optional[MutationType] = None,
        parent_program: Optional[MemoryProgram] = None
    ) -> MutationResult:
        """
        对程序进行变异
        
        Args:
            program: 原始程序
            mutation_type: 指定变异类型 (None = 自动选择)
            parent_program: 父程序 (用于 crossover)
            
        Returns:
            MutationResult
        """
        # 如果未指定变异类型，先选择
        if mutation_type is None:
            mutation_type = self._select_mutation_type(program)
        
        # 创建子程序
        child = program.evolve(parent_id=program.program_id, mutation_type=mutation_type)
        
        changed_fields = []
        
        try:
            # 根据变异类型执行
            if mutation_type == MutationType.SCHEMA_FIELD_ADD:
                changed_fields = self._schema_field_add(child)
            elif mutation_type == MutationType.SCHEMA_FIELD_REMOVE:
                changed_fields = self._schema_field_remove(child)
            elif mutation_type == MutationType.SCHEMA_FIELD_MODIFY:
                changed_fields = self._schema_field_modify(child)
            elif mutation_type == MutationType.LOGIC_READ_MODIFY:
                changed_fields = self._logic_read_modify(child)
            elif mutation_type == MutationType.LOGIC_WRITE_MODIFY:
                changed_fields = self._logic_write_modify(child)
            elif mutation_type == MutationType.LOGIC_QUERY_ADD:
                changed_fields = self._logic_query_add(child)
            elif mutation_type == MutationType.KEYWORD_ADD:
                changed_fields = self._keyword_add(child)
            elif mutation_type == MutationType.KEYWORD_REMOVE:
                changed_fields = self._keyword_remove(child)
            elif mutation_type == MutationType.KEYWORD_REPLACE:
                changed_fields = self._keyword_replace(child)
            elif mutation_type == MutationType.THRESHOLD_ADJUST:
                changed_fields = self._threshold_adjust(child)
            elif mutation_type == MutationType.PRIORITY_ADJUST:
                changed_fields = self._priority_adjust(child)
            elif mutation_type == MutationType.GUIDANCE_MODIFY:
                changed_fields = self._guidance_modify(child)
            elif mutation_type == MutationType.CROSSOVER:
                if parent_program:
                    changed_fields = self._crossover(child, parent_program)
                else:
                    changed_fields = self._keyword_add(child)
            elif mutation_type == MutationType.ENSEMBLE:
                changed_fields = self._ensemble(child)
            elif mutation_type == MutationType.RANDOM_CHANGE:
                changed_fields = self._random_change(child)
            else:
                logger.warning(f"Unknown mutation type: {mutation_type}")
                changed_fields = self._keyword_add(child)
            
            # 更新统计
            self._mutation_counts[mutation_type.value] += 1
            
            logger.info(f"Mutated {program.name} -> {mutation_type.value}: {changed_fields}")
            return MutationResult(child, mutation_type, f"Applied {mutation_type.value}", changed_fields)
            
        except Exception as e:
            logger.error(f"Mutation failed for {program.name}: {e}")
            # 变异失败，返回保守的关键词调整
            child = program.evolve(parent_id=program.program_id, mutation_type=MutationType.KEYWORD_ADD)
            self._keyword_add(child)
            return MutationResult(child, MutationType.KEYWORD_ADD, f"Fallback: {str(e)}", ["trigger_keywords"])
    
    def _select_mutation_type(self, program: MemoryProgram) -> MutationType:
        """根据程序状态选择变异类型"""
        # 根据适应度选择
        if program.fitness_score < 0.2:
            # 适应度很低，使用激进变异
            candidates = [
                MutationType.KEYWORD_ADD,
                MutationType.KEYWORD_REPLACE,
                MutationType.GUIDANCE_MODIFY,
            ]
        elif program.fitness_score < 0.4:
            # 适应度较低，使用中等变异
            candidates = [
                MutationType.KEYWORD_REPLACE,
                MutationType.THRESHOLD_ADJUST,
                MutationType.GUIDANCE_MODIFY,
                MutationType.PRIORITY_ADJUST,
            ]
        else:
            # 适应度还行，使用保守变异
            candidates = [
                MutationType.KEYWORD_ADD,
                MutationType.KEYWORD_REMOVE,
                MutationType.THRESHOLD_ADJUST,
            ]
        
        # 根据程序特征调整
        if len(program.instructions.trigger_keywords) < 3:
            candidates.append(MutationType.KEYWORD_ADD)
        if len(program.instructions.trigger_keywords) > 15:
            candidates.append(MutationType.KEYWORD_REMOVE)
        
        # 添加默认变异
        candidates.extend([
            MutationType.KEYWORD_ADD,
            MutationType.KEYWORD_REPLACE,
        ])
        
        return random.choice(candidates)
    
    # ── Schema 变异 ──────────────────────────────────────────────────────────────
    
    def _schema_field_add(self, program: MemoryProgram) -> List[str]:
        """Schema Field Add: 添加新字段"""
        changed = []
        
        # 添加新的元数据字段
        if not program.category:
            program.category = "general"
            changed.append("category")
        
        if not program.tags:
            program.tags = ["auto-generated"]
            changed.append("tags")
        
        if not program.agent_guidance:
            program.agent_guidance = self._generate_guidance(program)
            changed.append("agent_guidance")
        
        # 添加新的 trigger keyword
        new_keywords = self._generate_keywords(program)
        added = [k for k in new_keywords if k not in program.instructions.trigger_keywords]
        if added:
            program.instructions.trigger_keywords.extend(added[:2])
            changed.append("trigger_keywords")
        
        return changed
    
    def _schema_field_remove(self, program: MemoryProgram) -> List[str]:
        """Schema Field Remove: 移除冗余字段"""
        changed = []
        
        # 移除过长的关键词
        if len(program.instructions.trigger_keywords) > 5:
            removed = program.instructions.trigger_keywords[-2:]
            program.instructions.trigger_keywords = program.instructions.trigger_keywords[:-2]
            changed.append(f"removed_keywords: {removed}")
        
        # 移除空的 guidance
        if program.agent_guidance and len(program.agent_guidance) < 10:
            program.agent_guidance = ""
            changed.append("agent_guidance")
        
        return changed
    
    def _schema_field_modify(self, program: MemoryProgram) -> List[str]:
        """Schema Field Modify: 修改字段值"""
        changed = []
        
        # 修改优先级
        if program.priority == 5:
            new_priority = random.choice([3, 4, 6, 7])
            program.priority = new_priority
            changed.append(f"priority: 5 -> {new_priority}")
        
        # 修改阈值
        old_threshold = program.instructions.confidence_threshold
        if program.instructions.confidence_threshold == 0.3:
            program.instructions.confidence_threshold = random.choice([0.2, 0.25, 0.35, 0.4])
            changed.append(f"trigger_threshold: {old_threshold} -> {program.instructions.confidence_threshold}")
        
        return changed
    
    # ── Logic 变异 ──────────────────────────────────────────────────────────────
    
    def _logic_read_modify(self, program: MemoryProgram) -> List[str]:
        """Logic Read Modify: 修改读取逻辑"""
        changed = []
        
        # 修改 trigger threshold
        old = program.instructions.confidence_threshold
        if program.instructions.confidence_threshold > 0.2:
            program.instructions.confidence_threshold -= 0.05
            changed.append(f"trigger_threshold: {old:.2f} -> {program.instructions.confidence_threshold:.2f}")
        
        return changed
    
    def _logic_write_modify(self, program: MemoryProgram) -> List[str]:
        """Logic Write Modify: 修改写入逻辑"""
        changed = []
        
        # 增加 priority
        if program.priority < 9:
            old = program.priority
            program.priority = min(10, program.priority + 1)
            changed.append(f"priority: {old} -> {program.priority}")
        
        return changed
    
    def _logic_query_add(self, program: MemoryProgram) -> List[str]:
        """Logic Query Add: 添加查询条件"""
        changed = []
        
        # 添加新的 trigger keywords
        new_kws = self._generate_keywords(program)
        existing = set(program.instructions.trigger_keywords)
        to_add = [k for k in new_kws if k not in existing]
        
        if to_add:
            program.instructions.trigger_keywords.extend(to_add[:2])
            changed.append(f"added_keywords: {to_add[:2]}")
        
        return changed
    
    # ── Instructions 变异 ───────────────────────────────────────────────────────
    
    def _keyword_add(self, program: MemoryProgram) -> List[str]:
        """Keyword Add: 添加触发关键词"""
        new_kws = self._generate_keywords(program)
        existing = set(program.instructions.trigger_keywords)
        to_add = [k for k in new_kws if k not in existing]
        
        if to_add:
            program.instructions.trigger_keywords.extend(to_add[:min(3, len(to_add))])
            return [f"added: {to_add[:3]}"]
        return []
    
    def _keyword_remove(self, program: MemoryProgram) -> List[str]:
        existing = set(program.instructions.trigger_keywords)
        if len(program.instructions.trigger_keywords) > 3:
            removed = program.instructions.trigger_keywords.pop()
            return [f"removed: {removed}"]
        return []
    
    def _keyword_replace(self, program: MemoryProgram) -> List[str]:
        """Keyword Replace: 替换关键词"""
        if program.instructions.trigger_keywords:
            idx = random.randrange(len(program.instructions.trigger_keywords))
            old_kw = program.instructions.trigger_keywords[idx]
            
            # 生成替代词
            new_kws = self._generate_keywords(program, seed=old_kw)
            if new_kws:
                new_kw = new_kws[0]
                program.instructions.trigger_keywords[idx] = new_kw
                return [f"replaced: {old_kw} -> {new_kw}"]
        return []
    
    def _threshold_adjust(self, program: MemoryProgram) -> List[str]:
        """Threshold Adjust: 调整触发阈值"""
        old = program.instructions.confidence_threshold
        
        # 随机调整 ±0.05
        delta = random.choice([-0.05, -0.03, 0.03, 0.05])
        new_threshold = max(0.1, min(0.9, program.instructions.confidence_threshold + delta))
        
        program.instructions.confidence_threshold = round(new_threshold, 2)
        return [f"trigger_threshold: {old:.2f} -> {program.instructions.confidence_threshold:.2f}"]
    
    def _priority_adjust(self, program: MemoryProgram) -> List[str]:
        """Priority Adjust: 调整优先级"""
        old = program.priority
        
        delta = random.choice([-1, 1, 2, -2])
        new_priority = max(1, min(10, program.priority + delta))
        
        program.priority = new_priority
        return [f"priority: {old} -> {new_priority}"]
    
    def _guidance_modify(self, program: MemoryProgram) -> List[str]:
        """Guidance Modify: 修改指导信息"""
        if program.agent_guidance:
            old = program.agent_guidance
            # 简单修改
            if "IMPORTANT" in old:
                program.agent_guidance = old.replace("IMPORTANT", "NOTE")
            else:
                program.agent_guidance = "IMPORTANT: " + old[:200]
            return [f"agent_guidance modified"]
        else:
            program.agent_guidance = self._generate_guidance(program)
            return ["agent_guidance added"]
    
    # ── 组合变异 ────────────────────────────────────────────────────────────────
    
    def _crossover(self, child: MemoryProgram, parent: MemoryProgram) -> List[str]:
        """
        Crossover: 交叉组合
        
        从父程序中提取特征并合并到子程序中
        """
        changed = []
        
        # 交换部分关键词
        child_kws = set(child.trigger_keywords)
        parent_new = [k for k in parent.trigger_keywords if k not in child_kws]
        
        if parent_new:
            child.trigger_keywords.extend(parent_new[:2])
            changed.append(f"crossover_keywords: {parent_new[:2]}")
        
        # 交换 guidance
        if parent.agent_guidance and not child.agent_guidance:
            child.agent_guidance = parent.agent_guidance
            changed.append("crossover_guidance")
        
        # 交换 category
        if parent.category and not child.category:
            child.category = parent.category
            changed.append("crossover_category")
        
        # 混合 priority (取平均)
        if parent.priority and child.priority:
            avg_priority = round((parent.priority + child.priority) / 2)
            child.priority = avg_priority
            changed.append(f"crossover_priority: avg({parent.priority}, {child.priority}) = {avg_priority}")
        
        # 混合 threshold
        if parent.trigger_threshold and child.trigger_threshold:
            avg_threshold = round((parent.trigger_threshold + child.trigger_threshold) / 2, 2)
            child.trigger_threshold = avg_threshold
            changed.append(f"crossover_threshold: avg({parent.trigger_threshold:.2f}, {child.trigger_threshold:.2f}) = {avg_threshold:.2f}")
        
        return changed
    
    def _ensemble(self, program: MemoryProgram) -> List[str]:
        """
        Ensemble: 集成多个策略
        
        应用多种变异策略的组合，强化程序性能
        """
        changed = []
        
        # 提高优先级
        if program.priority < 8:
            old = program.priority
            program.priority = min(10, program.priority + 2)
            changed.append(f"priority: {old} -> {program.priority}")
        
        # 降低阈值
        if program.instructions.confidence_threshold > 0.2:
            old = program.instructions.confidence_threshold
            program.instructions.confidence_threshold = max(0.15, program.instructions.confidence_threshold - 0.1)
            changed.append(f"trigger_threshold: {old:.2f} -> {program.instructions.confidence_threshold:.2f}")
        
        # 添加更多关键词
        new_kws = self._generate_keywords(program)
        existing = set(program.instructions.trigger_keywords)
        to_add = [k for k in new_kws if k not in existing]
        if to_add:
            program.instructions.trigger_keywords.extend(to_add[:3])
            changed.append(f"ensemble_keywords: {to_add[:3]}")
        
        # 强化 guidance
        if program.agent_guidance:
            old = program.agent_guidance
            if not old.startswith("IMPORTANT"):
                program.agent_guidance = "IMPORTANT: " + old
            changed.append("ensemble_guidance")
        
        return changed
    
    # ── 随机变异 ────────────────────────────────────────────────────────────────
    
    def _random_change(self, program: MemoryProgram) -> List[str]:
        """
        Random Change: 随机改变
        
        从多种变异策略中随机选择一种应用
        """
        mutations = [
            lambda: self._keyword_add(program),
            lambda: self._keyword_remove(program),
            lambda: self._threshold_adjust(program),
            lambda: self._priority_adjust(program),
            lambda: self._guidance_modify(program),
            lambda: self._schema_field_add(program) if len(program.instructions.trigger_keywords) < 3 else self._keyword_add(program),
        ]
        
        selected = random.choice(mutations)
        return selected()
    
    # ── 辅助方法 ────────────────────────────────────────────────────────────────
    
    def _generate_keywords(self, program: MemoryProgram, seed: Optional[str] = None) -> List[str]:
        """生成与程序相关的关键词"""
        # 基于内容的关键词
        guidance = getattr(program.instructions, 'agent_guidance', '') or ''
        desc = getattr(program, 'description', '') or ''
        content_lower = (guidance + ' ' + desc).lower()
        
        words = re.findall(r'\b[a-z]{3,15}\b', content_lower)
        
        # 过滤常见词
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
            'has', 'her', 'was', 'one', 'our', 'out', 'now', 'see', 'way',
            'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were',
            'being', 'some', 'into', 'only', 'other', 'than', 'then', 'also',
            'will', 'just', 'your', 'what', 'about', 'which', 'when', 'make',
            'like', 'time', 'very', 'take', 'them', 'would', 'there', 'could',
            'more', 'very', 'after', 'most', 'also', 'back', 'only', 'even',
            'first', 'well', 'such', 'must', 'much', 'know', 'here', 'many',
            'many', 'want', 'because', 'these', 'give', 'day', 'doesn', 'didn'
        }
        words = [w for w in words if w not in stop_words]
        
        # 频率排序
        word_freq = Counter(words)
        top_words = [w for w, _ in word_freq.most_common(10)]
        
        if seed:
            # 如果有种子词，生成相似词
            return top_words[:5] if top_words else [seed + "_variant"]
        
        return top_words[:5] if top_words else ["task_" + str(uuid.uuid4())[:8]]
    
    def _generate_guidance(self, program: MemoryProgram) -> str:
        """生成指导信息"""
        guidance_templates = [
            "Use this skill when {trigger} is requested. Focus on quality and efficiency.",
            "This skill handles {category} tasks. Follow the established patterns.",
            "When invoking, consider the context and apply appropriate modifications.",
            "Priority: {priority}/10. Execute with attention to detail.",
            "Handle {trigger} requests with care and precision.",
            "Focus on delivering high-quality results for {category} tasks.",
        ]
        
        template = random.choice(guidance_templates)
        return template.format(
            trigger=program.instructions.trigger_keywords[0] if program.instructions.trigger_keywords else "tasks",
            category=program.category or "general",
            priority=program.priority
        )
    
    def get_mutation_stats(self) -> Dict[str, int]:
        """获取变异统计"""
        return copy.copy(self._mutation_counts)
    
    def reset_stats(self):
        """重置统计"""
        self._mutation_counts = {mt.value: 0 for mt in MutationType}
    
    def get_all_mutation_types(self) -> List[MutationType]:
        """获取所有变异类型"""
        return list(MutationType)
    
    def get_mutation_probability(self, mutation_type: MutationType) -> float:
        """获取指定变异类型的概率"""
        return self.mutation_probabilities.get(mutation_type, 0.0)
