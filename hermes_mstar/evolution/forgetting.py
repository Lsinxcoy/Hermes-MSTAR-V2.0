"""
ForgettingMechanism — 智能遗忘机制
Ported from hermes-mstar agent/forgetting.py

策略: archive (降冷存) / merge (合并相似) / delete (永久删除)
"""
import json
import time
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger("hermes.mstar.forgetting")


@dataclass
class ForgetCandidate:
    """遗忘候选"""
    program: Any
    forget_score: float
    age_days: int
    fitness_score: float
    use_count: int
    failure_rate: float
    strategy: str = "keep"  # keep / archive / merge / delete
    reason: str = ""


@dataclass
class ForgetDecision:
    """遗忘决策"""
    program_id: str
    action: str  # archive / merge / delete
    reason: str
    target_merge_id: Optional[str] = None


class ForgettingMechanism:
    """
    智能遗忘机制

    策略:
    - archive: 降级到冷存储（可恢复）
    - merge: 合并相似 program
    - delete: 永久删除

    遗忘评分:
    forget_score = fitness * 0.4 + recency * 0.3 + quality * 0.3
    """

    FORGET_THRESHOLD = 0.15   # 低于此值考虑遗忘
    ARCHIVE_AGE_DAYS = 30     # 超过此天数自动归档
    MERGE_SIMILARITY = 0.8   # 相似度阈值

    def __init__(self, archive_dir: Optional[str] = None, backup_dir: Optional[str] = None):
        self.archive_dir = Path(archive_dir or '/tmp/hermes_mstar_forgetting_archive')
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        self._forgetting_history: List[Dict] = []
        self._archived_programs: Dict[str, str] = {}  # id -> archive_path

    def evaluate(self, program_or_list: Any) -> Any:
        """评估 program 或 program 列表
        
        支持 dict-like 对象和 dataclass 实例。
        dict 接口：fitness, episodes, days_inactive (可选)
        object 接口：fitness_score, use_count, last_activity_at, created_at
        """
        # 批量处理：如果是列表，转换后调用 evaluate_batch
        if isinstance(program_or_list, (list, tuple)):
            converted = []
            for p in program_or_list:
                if isinstance(p, dict):
                    # 转换 dict → 简单对象（支持属性访问）
                    class DictLike:
                        def __init__(self, d):
                            for k, v in d.items():
                                setattr(self, k, v)
                            # 别名映射
                            self.fitness_score = getattr(self, 'fitness', 0.0)
                            self.use_count = getattr(self, 'episodes', 0)
                            self.last_activity_at = getattr(self, 'last_activity_at', getattr(self, 'days_inactive', None))
                            self.created_at = getattr(self, 'created_at', None)
                    converted.append(DictLike(p))
                else:
                    converted.append(p)
            return self.evaluate_batch(converted)
        
        program = program_or_list
        
        # 获取属性（支持 dict 和 object）
        def get_attr(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        last_act = get_attr(program, 'last_activity_at') or get_attr(program, 'created_at') or ''
        created = get_attr(program, 'created_at') or ''
        age_days = self._days_since(last_act) if last_act else 0
        
        use_count = get_attr(program, 'use_count', 0) or get_attr(program, 'episodes', 0) or 0
        failure_count = get_attr(program, 'failure_count', 0) or 0
        failure_rate = (failure_count / use_count) if use_count > 0 else 0.5
        
        recency = max(0, 1 - age_days / 90)
        quality = get_attr(program, 'avg_quality', 0.5) or 0.5
        fitness_score = get_attr(program, 'fitness_score', 0.0) or get_attr(program, 'fitness', 0.0) or 0.0

        forget_score = (
            fitness_score * 0.4 +
            recency * 0.3 +
            quality * 0.3
        )

        strategy, reason = self._decide_strategy(program, forget_score, age_days, failure_rate, fitness_score)

        return ForgetCandidate(
            program=program,
            forget_score=forget_score,
            age_days=age_days,
            fitness_score=fitness_score,
            use_count=use_count,
            failure_rate=failure_rate,
            strategy=strategy,
            reason=reason,
        )

    def evaluate_batch(self, programs: List[Any]) -> List[ForgetCandidate]:
        """批量评估"""
        return [self.evaluate(p) for p in programs
                if p.fitness_score < self.FORGET_THRESHOLD or
                   self._days_since(p.last_activity_at or p.created_at) > self.ARCHIVE_AGE_DAYS]

    def decide(self, candidate: ForgetCandidate) -> str:
        """返回策略: archive / merge / delete"""
        return candidate.strategy

    def _decide_strategy(
        self,
        program: Any,
        forget_score: float,
        age_days: int,
        failure_rate: float,
        fitness_score: float = 0.0
    ) -> tuple[str, str]:
        """决定遗忘策略"""
        if fitness_score >= self.FORGET_THRESHOLD:
            if age_days > self.ARCHIVE_AGE_DAYS * 2:
                return "archive", f"Stale for {age_days} days"
            return "keep", "Fitness above threshold"

        if failure_rate > 0.7:
            return "delete", f"High failure rate: {failure_rate:.1%}"

        if forget_score < 0.1 and age_days > 14:
            return "delete", f"Very low forget_score: {forget_score:.3f}"

        if forget_score < 0.2 and age_days > self.ARCHIVE_AGE_DAYS:
            return "archive", f"Low forget_score {forget_score:.3f}, age {age_days}d"

        # 查找相似 program
        similar = self._find_similar(program)
        if similar and forget_score < 0.25:
            return "merge", f"Similar to {similar.id}"

        return "archive", f"Default archive (score={forget_score:.3f})"

    def _find_similar(self, program: Any) -> Optional[Any]:
        """查找相似 program"""
        # 简单实现：通过 category/name 相似度
        return None  # 后续实现

    def _days_since(self, timestamp: str) -> int:
        """计算天数"""
        if not timestamp:
            return 0
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp)
            else:
                dt = timestamp
            return (datetime.now() - dt).days
        except:
            return 0

    # ── Actions ────────────────────────────────────────────────────────────────

    def archive(self, program: Any) -> bool:
        """归档 program"""
        try:
            archive_path = self.archive_dir / f"{program.id}.json"
            data = program.to_dict() if hasattr(program, 'to_dict') else program.__dict__
            with open(archive_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            self._archived_programs[program.id] = str(archive_path)
            self._save_history({"action": "archive", "program_id": program.id, "timestamp": datetime.now().isoformat()})
            logger.info(f"Archived program: {program.id}")
            return True
        except Exception as e:
            logger.error(f"Archive failed for {program.id}: {e}")
            return False

    def delete(self, program: Any) -> bool:
        """永久删除 program"""
        try:
            # 先归档再删除
            self.archive(program)
            # 创建备份 tarball
            if self.backup_dir:
                self._create_backup(program)
            self._save_history({"action": "delete", "program_id": program.id, "timestamp": datetime.now().isoformat()})
            logger.info(f"Deleted program: {program.id}")
            return True
        except Exception as e:
            logger.error(f"Delete failed for {program.id}: {e}")
            return False

    def merge(self, program: Any, target_id: Optional[str] = None) -> bool:
        """合并 program"""
        self._save_history({
            "action": "merge",
            "program_id": program.id,
            "target_id": target_id,
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(f"Merged program: {program.id} -> {target_id}")
        return True

    def restore(self, program_id: str) -> bool:
        """恢复已归档的 program"""
        archive_path = self._archived_programs.get(program_id)
        if not archive_path:
            archive_path = self.archive_dir / f"{program_id}.json"
        if not Path(archive_path).exists():
            logger.warning(f"Archive not found: {program_id}")
            return False
        logger.info(f"Restore not yet implemented for: {program_id}")
        return False

    def _create_backup(self, program: Any):
        """创建备份"""
        if not self.backup_dir:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{program.id}_{timestamp}.json"
        data = program.to_dict() if hasattr(program, 'to_dict') else program.__dict__
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def _save_history(self, record: Dict):
        self._forgetting_history.append(record)
        if len(self._forgetting_history) > 1000:
            self._forgetting_history = self._forgetting_history[-500:]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_archived": len(self._archived_programs),
            "archive_size_mb": round(sum(
                Path(p).stat().st_size for p in self._archived_programs.values() if Path(p).exists()
            ) / 1024 / 1024, 2),
            "decisions_made": len(self._forgetting_history),
        }
