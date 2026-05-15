"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    Hermes MSTAR: FitnessTracker - 适应度追踪器                         ║
║                                                                                      ║
║  SQLite-based fitness tracking system for skill programs                             ║
║  功能:                                                                               ║
║    - 每次 tool_call 后实时更新适应度                                                   ║
║    - 置信度计算 (min(1.0, episodes/20))                                              ║
║    - 适应度阈值检测 (触发变异/遗忘)                                                   ║
║    - SQLite 持久化                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..memory_program import MemoryProgram

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """单次执行结果
    
    用于记录每次记忆程序调用的结果，包含:
    - success: 任务是否成功
    - quality: 输出质量评分 [0.0, 1.0]
    - latency: 执行延迟 (秒)
    - confidence: 执行置信度 [0.0, 1.0]
    - tokens_consumed: Token 消耗数量
    - error: 错误信息 (如果有)
    """
    success: bool
    quality: float = 0.5
    latency: float = 0.0
    confidence: float = 0.5
    tokens_consumed: int = 0
    error: Optional[str] = None
    
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


class FitnessTracker:
    """
    SQLite-based 实时适应度追踪器
    
    设计原则:
    1. 线程安全 - 多线程环境安全
    2. 增量更新 - 每次 tool_call 后立即更新
    3. SQLite 持久化 - WAL 模式
    4. 阈值检测 - 自动触发变异/遗忘评估
    """
    
    def __init__(self, db_path: str):
        """
        初始化适应度追踪器

        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._local = threading.local()
        self._lock = threading.RLock()
        self._init_db()
        
        # 配置阈值
        self.mutation_threshold = 0.3  # 适应度 < 0.3 触发变异
        self.forget_threshold = 0.2     # 适应度 < 0.2 触发遗忘
        self.stale_days = 30            # 30天后变为 stale
        self.archive_days = 90          # 90天后归档
        
        # 统计
        self.total_updates = 0
        self.total_mutations_triggered = 0
        self.total_forgets_triggered = 0
        
        self._init_db()
    
    @property
    def conn(self) -> sqlite3.Connection:
        """获取线程本地的数据库连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                isolation_level=None,  # 自动提交
                check_same_thread=False
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=20000")
        return self._local.conn
    
    def _init_db(self):
        """初始化数据库表"""
        with self._lock:
            conn = self.conn
            conn.executescript(f"""
                CREATE TABLE IF NOT EXISTS skill_programs (
                    program_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    parent_id TEXT,
                    status TEXT DEFAULT 'active',
                    pin INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_activity_at TEXT NOT NULL,
                    stale_at TEXT,
                    archived_at TEXT,
                    
                    -- Schema
                    schema_name TEXT,
                    schema_description TEXT,
                    schema_fields TEXT,
                    schema_storage_type TEXT,
                    schema_domain TEXT,
                    
                    -- Logic
                    logic_read_template TEXT,
                    logic_write_template TEXT,
                    logic_query_fields TEXT,
                    logic_response_template TEXT,
                    logic_preconditions TEXT,
                    logic_postconditions TEXT,
                    
                    -- Instructions
                    trigger_keywords TEXT,
                    usage_examples TEXT,
                    agent_guidance TEXT,
                    confidence_threshold REAL DEFAULT 0.5,
                    priority INTEGER DEFAULT 5,
                    
                    -- Evolution
                    fitness_score REAL DEFAULT 0.5,
                    episode_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    avg_quality REAL DEFAULT 0.5,
                    avg_latency REAL DEFAULT 0.0,
                    avg_tokens REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    mutation_count INTEGER DEFAULT 0,
                    last_mutation TEXT,
                    lineage TEXT,
                    mutation_history TEXT,
                    root_id TEXT,
                    
                    -- Quality Gates
                    quality_gates_passed INTEGER DEFAULT 0,
                    compile_status TEXT DEFAULT 'pending',
                    runtime_status TEXT DEFAULT 'pending',
                    logic_status TEXT DEFAULT 'pending',
                    quality_status TEXT DEFAULT 'pending',
                    
                    -- Umbrella
                    parent_skill_id TEXT,
                    absorbed_into TEXT,
                    is_umbrella INTEGER DEFAULT 0,
                    children_ids TEXT,
                    
                    -- Metadata
                    source_system TEXT DEFAULT 'user',
                    source_file TEXT,
                    tags TEXT,
                    category TEXT,
                    domain TEXT DEFAULT 'general',
                    description TEXT,
                    updated_at TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_programs_fitness ON skill_programs(fitness_score);
                CREATE INDEX IF NOT EXISTS idx_programs_status ON skill_programs(status);
                CREATE INDEX IF NOT EXISTS idx_programs_name ON skill_programs(name);
                CREATE INDEX IF NOT EXISTS idx_programs_updated ON skill_programs(updated_at);
                
                CREATE TABLE IF NOT EXISTS fitness_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_id TEXT,
                    event_type TEXT NOT NULL,
                    success INTEGER,
                    quality REAL,
                    latency REAL,
                    confidence REAL,
                    fitness_score REAL,
                    tokens_consumed INTEGER,
                    error TEXT,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    details TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_fitness_events_program ON fitness_events(program_id);
                CREATE INDEX IF NOT EXISTS idx_fitness_events_type ON fitness_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_fitness_events_timestamp ON fitness_events(timestamp);
                
                CREATE TABLE IF NOT EXISTS mutation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    mutation_type TEXT NOT NULL,
                    fitness_before REAL,
                    fitness_after REAL,
                    status TEXT DEFAULT 'evaluating',
                    timestamp TEXT NOT NULL,
                    details TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_mutation_parent ON mutation_events(parent_id);
                CREATE INDEX IF NOT EXISTS idx_mutation_type ON mutation_events(mutation_type);
                CREATE INDEX IF NOT EXISTS idx_mutation_status ON mutation_events(status);
            """)
            logger.info("FitnessTracker 数据库表初始化完成")
    
    # ── 核心更新方法 ─────────────────────────────────────────────────────────────────
    
    def record_execution(
        self,
        skill_id: str,
        skill_name: str,
        success: bool,
        quality: float = 0.5,
        latency: float = 0.0,
        tokens_consumed: int = 0
    ) -> Optional[MemoryProgram]:
        """
        便捷方法: 记录一次技能执行
        
        Args:
            skill_id: 技能ID
            skill_name: 技能名称
            success: 是否成功
            quality: 质量分数 (0.0-1.0)
            latency: 延迟 (秒)
            tokens_consumed: 消耗的Token数
            
        Returns:
            更新后的 MemoryProgram
        """
        return self.update(
            skill_id=skill_id,
            success=success,
            quality=quality,
            latency=latency,
            confidence=0.5,
            tokens_consumed=tokens_consumed,
            skill_name=skill_name
        )
    
    def update(
        self,
        skill_id: str,
        success: bool,
        quality: float = 0.5,
        latency: float = 0.0,
        confidence: float = 0.5,
        error: Optional[str] = None,
        skill_name: Optional[str] = None,
        tokens_consumed: int = 0
    ) -> Optional[MemoryProgram]:
        """
        实时更新适应度 (每次 tool_call 后调用)
        
        Args:
            skill_id: 技能ID
            success: 是否成功
            quality: 质量分数 (0.0-1.0)
            latency: 延迟 (秒)
            confidence: 置信度
            error: 错误信息
            skill_name: 技能名称 (用于创建新记录)
            tokens_consumed: Token消耗数
            
        Returns:
            更新后的 MemoryProgram，如果不存在则返回 None
        """
        with self._lock:
            try:
                # 获取或创建 MemoryProgram
                program = self.get_skill(skill_id)
                
                if program is None:
                    # 创建新记录
                    program = self._create_skill_record(skill_id, skill_name or "unknown")
                
                # 构建 EpisodeResult
                episode = EpisodeResult(
                    success=success,
                    quality=quality,
                    latency=latency,
                    confidence=confidence,
                    tokens_consumed=tokens_consumed,
                    error=error
                )
                
                # 更新适应度
                old_fitness = program.evolution.fitness_score
                program.update_fitness(episode)
                new_fitness = program.evolution.fitness_score
                
                # 持久化
                self._save_skill(program)
                
                # 记录事件
                self._record_event(
                    program_id=skill_id,
                    event_type="fitness_update",
                    success=success,
                    quality=quality,
                    latency=latency,
                    confidence=confidence,
                    fitness_score=new_fitness,
                    tokens_consumed=tokens_consumed,
                    error=error
                )
                
                # 检查阈值
                needs_mutation = program.should_mutate(self.mutation_threshold)
                needs_forget = program.should_forget(self.forget_threshold)
                
                if needs_mutation:
                    self.total_mutations_triggered += 1
                    logger.info(f"适应度低于阈值 [{skill_id}]: {old_fitness:.3f} -> {new_fitness:.3f}")
                
                if needs_forget:
                    self.total_forgets_triggered += 1
                    logger.warning(f"适应度过低，建议遗忘 [{skill_id}]: {new_fitness:.3f}")
                
                self.total_updates += 1
                
                return program
                
            except Exception as e:
                logger.error(f"更新适应度失败 [{skill_id}]: {e}")
                return None
    
    # ── 查询方法 ─────────────────────────────────────────────────────────────────
    
    def get_skill(self, skill_id: str) -> Optional[MemoryProgram]:
        """
        获取技能
        
        Args:
            skill_id: 技能ID
            
        Returns:
            MemoryProgram 或 None
        """
        with self._lock:
            try:
                row = self.conn.execute(
                    "SELECT * FROM skill_programs WHERE program_id = ?",
                    (skill_id,)
                ).fetchone()
                
                if row is None:
                    return None
                
                return self._row_to_program(row)
                
            except Exception as e:
                logger.error(f"获取技能失败 [{skill_id}]: {e}")
                return None
    
    def get_all_skills(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryProgram]:
        """
        获取所有技能
        
        Args:
            status: 过滤状态
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            MemoryProgram 列表
        """
        with self._lock:
            try:
                if status:
                    query = "SELECT * FROM skill_programs WHERE status = ? ORDER BY fitness_score DESC LIMIT ? OFFSET ?"
                    rows = self.conn.execute(query, (status, limit, offset)).fetchall()
                else:
                    query = "SELECT * FROM skill_programs ORDER BY fitness_score DESC LIMIT ? OFFSET ?"
                    rows = self.conn.execute(query, (limit, offset)).fetchall()
                
                return [self._row_to_program(row) for row in rows]
                
            except Exception as e:
                logger.error(f"获取技能列表失败: {e}")
                return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            try:
                stats = {}
                
                # 总技能数
                row = self.conn.execute("SELECT COUNT(*) FROM skill_programs").fetchone()
                stats['total_programs'] = row[0] if row else 0
                
                # 按状态统计
                rows = self.conn.execute(
                    "SELECT status, COUNT(*) FROM skill_programs GROUP BY status"
                ).fetchall()
                stats['by_status'] = {r[0]: r[1] for r in rows}
                
                # 平均适应度
                row = self.conn.execute("SELECT AVG(fitness_score) FROM skill_programs").fetchone()
                stats['avg_fitness'] = row[0] if row and row[0] else 0.5
                
                # 适应度分布
                rows = self.conn.execute("""
                    SELECT 
                        CASE 
                            WHEN fitness_score >= 0.8 THEN 'excellent'
                            WHEN fitness_score >= 0.6 THEN 'good'
                            WHEN fitness_score >= 0.4 THEN 'average'
                            WHEN fitness_score >= 0.2 THEN 'poor'
                            ELSE 'failing'
                        END as level,
                        COUNT(*) as count
                    FROM skill_programs
                    GROUP BY level
                """).fetchall()
                stats['fitness_distribution'] = {r[0]: r[1] for r in rows}
                
                # 待处理伞形合并数
                row = self.conn.execute("""
                    SELECT COUNT(*) FROM fitness_events
                    WHERE event_type = 'umbrella_merge'
                    AND timestamp > datetime('now', '-7 days')
                """).fetchone()
                stats['pending_umbrella_merges'] = row[0] if row and row[0] else 0
                
                # 总更新数
                stats['total_updates'] = self.total_updates
                stats['total_mutations_triggered'] = self.total_mutations_triggered
                stats['total_forgets_triggered'] = self.total_forgets_triggered
                
                return stats
                
            except Exception as e:
                logger.error(f"获取统计信息失败: {e}")
                return {}
    
    def get_token_aware_candidates(self, min_use_count: int = 10) -> List[MemoryProgram]:
        """
        获取应优化的候选技能 (Token感知)
        
        标准:
        1. fitness_score < 0.3 (低适应度)
        2. avg_latency > 平均值 (高延迟)
        3. episode_count >= min_use_count (有足够数据)
        
        Args:
            min_use_count: 最小使用次数
            
        Returns:
            候选技能列表
        """
        with self._lock:
            try:
                # 获取平均延迟
                row = self.conn.execute(
                    "SELECT AVG(avg_latency) FROM skill_programs WHERE avg_latency > 0"
                ).fetchone()
                avg_latency = row[0] if row and row[0] else 1000  # 默认1秒
                
                # 获取候选技能
                rows = self.conn.execute("""
                    SELECT * FROM skill_programs
                    WHERE fitness_score < 0.3
                    AND avg_latency > ?
                    AND episode_count >= ?
                    AND status = 'active'
                    ORDER BY fitness_score ASC, avg_latency DESC
                    LIMIT 20
                """, (avg_latency, min_use_count)).fetchall()
                
                return [self._row_to_program(row) for row in rows]
                
            except Exception as e:
                logger.error(f"获取Token感知候选失败: {e}")
                return []
    
    def calculate_token_efficiency(self, skill_id: str) -> float:
        """
        计算Token效率
        
        效率 = 产出价值 / Token消耗
        高效率 = 少Token + 高产出
        
        Args:
            skill_id: 技能ID
            
        Returns:
            Token效率分数 (0.0 - 10.0+)
        """
        skill = self.get_skill(skill_id)
        if not skill:
            return 0.0
        
        # 效率 = fitness_score / (avg_latency / 1000)
        # 归一化到 0-10 范围
        if skill.evolution.avg_latency > 0:
            efficiency = skill.evolution.fitness_score / (skill.evolution.avg_latency / 1000)
            return min(10.0, efficiency)  # 上限10.0
        return skill.evolution.fitness_score * 10  # 无延迟数据时用fitness估算
    
    # ── 持久化方法 ─────────────────────────────────────────────────────────────────
    
    def _save_skill(self, program: MemoryProgram) -> bool:
        """内部保存方法"""
        try:
            program.evolution.updated_at = datetime.now().isoformat()
            program.last_activity_at = program.evolution.updated_at
            
            schema = program.schema
            logic = program.logic
            instructions = program.instructions
            evolution = program.evolution
            quality = program.quality
            
            self.conn.execute("""
                INSERT OR REPLACE INTO skill_programs (
                    program_id, name, version, parent_id, status, pin,
                    created_at, last_activity_at, stale_at, archived_at,
                    schema_name, schema_description, schema_fields,
                    schema_storage_type, schema_domain,
                    logic_read_template, logic_write_template, logic_query_fields,
                    logic_response_template, logic_preconditions, logic_postconditions,
                    trigger_keywords, usage_examples, agent_guidance,
                    confidence_threshold, priority,
                    fitness_score, episode_count, success_count, failure_count,
                    avg_quality, avg_latency, avg_tokens, confidence,
                    mutation_count, last_mutation, lineage, mutation_history,
                    root_id,
                    quality_gates_passed, compile_status, runtime_status,
                    logic_status, quality_status,
                    parent_skill_id, absorbed_into, is_umbrella, children_ids,
                    source_system, source_file, tags, category, domain,
                    description, updated_at
                 ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                program.program_id, program.name, program.version, evolution.parent_id,
                program.status.value if hasattr(program.status, 'value') else program.status,
                int(program.pin),
                program.created_at, program.last_activity_at, program.stale_at, program.archived_at,
                schema.name, schema.description, json.dumps(schema.fields),
                schema.storage_type.value if hasattr(schema.storage_type, 'value') else schema.storage_type,
                schema.domain.value if hasattr(schema.domain, 'value') else schema.domain,
                logic.read_template, logic.write_template, json.dumps(logic.query_fields),
                logic.response_template, json.dumps(logic.preconditions), json.dumps(logic.postconditions),
                json.dumps(instructions.trigger_keywords), json.dumps(instructions.usage_examples),
                instructions.agent_guidance, instructions.confidence_threshold, instructions.priority,
                evolution.fitness_score, evolution.episode_count, evolution.success_count,
                evolution.failure_count, evolution.avg_quality, evolution.avg_latency,
                evolution.avg_tokens, evolution.confidence, evolution.mutation_count,
                evolution.last_mutation, json.dumps(evolution.lineage),
                json.dumps(evolution.mutation_history), evolution.root_id,
                int(quality.all_passed()),
                quality.compile_status, quality.runtime_status,
                quality.logic_status, quality.quality_status,
                program.parent_skill_id, program.absorbed_into, int(program.is_umbrella),
                json.dumps(program.children_ids),
                program.source_system, program.source_file, json.dumps(program.tags),
                program.category, program.domain.value if hasattr(program.domain, 'value') else program.domain,
                program.description, program.evolution.updated_at
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"保存技能失败 [{program.program_id}]: {e}")
            return False
    # ── 事件记录 ─────────────────────────────────────────────────────────────────
    
    def _record_event(
        self,
        program_id: Optional[str] = None,
        event_type: str = "unknown",
        success: Optional[bool] = None,
        quality: Optional[float] = None,
        latency: Optional[float] = None,
        confidence: Optional[float] = None,
        fitness_score: Optional[float] = None,
        tokens_consumed: Optional[int] = None,
        error: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Any] = None
    ):
        """记录进化事件"""
        try:
            # 序列化 details 为 JSON 字符串
            details_str = None
            if details is not None:
                if isinstance(details, str):
                    details_str = details
                else:
                    try:
                        details_str = json.dumps(details)
                    except (TypeError, ValueError):
                        details_str = str(details)
            
            self.conn.execute("""
                INSERT INTO fitness_events (
                    program_id, event_type, success, quality, latency,
                    confidence, fitness_score, tokens_consumed, error,
                    session_id, timestamp, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                program_id, event_type,
                int(success) if success is not None else None,
                quality, latency, confidence, fitness_score, tokens_consumed,
                error, session_id, datetime.now().isoformat(), details_str
            ))
        except Exception as e:
            logger.error(f"记录事件失败: {e}")
    
    def record_mutation(
        self,
        parent_id: str,
        child_id: str,
        mutation_type: str,
        fitness_before: float,
        fitness_after: float = 0.0,
        details: Optional[Dict] = None
    ):
        """
        记录变异事件
        
        Args:
            parent_id: 父程序ID
            child_id: 子程序ID
            mutation_type: 变异类型
            fitness_before: 变异前适应度
            fitness_after: 变异后适应度 (初始为0)
            details: 额外详情
        """
        with self._lock:
            try:
                details_str = json.dumps(details) if details else None
                
                self.conn.execute("""
                    INSERT INTO mutation_events (
                        parent_id, child_id, mutation_type,
                        fitness_before, fitness_after, status, timestamp, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    parent_id, child_id, mutation_type,
                    fitness_before, fitness_after, 'evaluating',
                    datetime.now().isoformat(), details_str
                ))
                
                self._record_event(
                    program_id=parent_id,
                    event_type="mutation",
                    details={
                        "child_id": child_id,
                        "mutation_type": mutation_type,
                        "fitness_before": fitness_before
                    }
                )
                
            except Exception as e:
                logger.error(f"记录变异事件失败: {e}")
    
    # ── 私有方法 ─────────────────────────────────────────────────────────────────
    
    def _create_skill_record(self, skill_id: str, skill_name: str) -> MemoryProgram:
        """创建新的技能记录"""
        from ..memory_program import create_memory_program, TaskDomain, StorageType
        
        program = create_memory_program(
            name=skill_name,
            description=f"Auto-created skill: {skill_name}",
            trigger_keywords=[],
            storage_type=StorageType.PROCEDURAL,
            domain=TaskDomain.GENERAL,
            source_system="fitness_tracker"
        )
        program.program_id = skill_id
        
        self._save_skill(program)
        return program
    
    def _row_to_program(self, row: sqlite3.Row) -> MemoryProgram:
        """将数据库行转换为 MemoryProgram"""
        try:
            columns = list(row.keys()) if hasattr(row, 'keys') else [desc[0] for desc in row.description]
            data = {col: row[idx] for idx, col in enumerate(columns)}
        except (TypeError, ValueError, AttributeError):
            # sqlite3.Row has no .description; sqlite3.tuple has no .keys() or .description.
            # Use the connection cursor to get column names from a zero-row SELECT.
            columns = [desc[0] for desc in self.conn.execute("SELECT * FROM skill_programs LIMIT 0").description]
            data = {col: row[idx] for idx, col in enumerate(columns)}
        
        # JSON 字段解码
        if data.get('schema_fields'):
            try:
                data['schema_fields'] = json.loads(data['schema_fields'])
            except:
                data['schema_fields'] = []
        
        if data.get('lineage'):
            try:
                data['lineage'] = json.loads(data['lineage'])
            except:
                data['lineage'] = []
        
        if data.get('mutation_history'):
            try:
                data['mutation_history'] = json.loads(data['mutation_history'])
            except:
                data['mutation_history'] = []
        
        if data.get('trigger_keywords'):
            try:
                data['trigger_keywords'] = json.loads(data['trigger_keywords'])
            except:
                data['trigger_keywords'] = []
        
        if data.get('usage_examples'):
            try:
                data['usage_examples'] = json.loads(data['usage_examples'])
            except:
                data['usage_examples'] = []
        
        if data.get('logic_query_fields'):
            try:
                data['logic_query_fields'] = json.loads(data['logic_query_fields'])
            except:
                data['logic_query_fields'] = []
        
        if data.get('logic_preconditions'):
            try:
                data['logic_preconditions'] = json.loads(data['logic_preconditions'])
            except:
                data['logic_preconditions'] = []
        
        if data.get('logic_postconditions'):
            try:
                data['logic_postconditions'] = json.loads(data['logic_postconditions'])
            except:
                data['logic_postconditions'] = []
        
        if data.get('children_ids'):
            try:
                data['children_ids'] = json.loads(data['children_ids'])
            except:
                data['children_ids'] = []
        
        if data.get('tags'):
            try:
                data['tags'] = json.loads(data['tags'])
            except:
                data['tags'] = []
        
        # 布尔字段
        data['pin'] = bool(data.get('pin'))
        data['quality_gates_passed'] = bool(data.get('quality_gates_passed'))
        data['is_umbrella'] = bool(data.get('is_umbrella'))
        
        # 构建嵌套对象
        from ..memory_program import Schema, Logic, Instructions, Evolution, QualityGates, StorageType, TaskDomain, ProgramStatus
        
        schema = Schema(
            name=data.get('schema_name', 'default'),
            description=data.get('schema_description', ''),
            fields=data.get('schema_fields', []),
            storage_type=StorageType(data.get('schema_storage_type', 'structured')) if data.get('schema_storage_type') else StorageType.STRUCTURED,
            domain=TaskDomain(data.get('schema_domain', 'general')) if data.get('schema_domain') else TaskDomain.GENERAL
        )
        
        logic = Logic(
            read_template=data.get('logic_read_template', ''),
            write_template=data.get('logic_write_template', ''),
            query_fields=data.get('logic_query_fields', []),
            response_template=data.get('logic_response_template', ''),
            preconditions=data.get('logic_preconditions', []),
            postconditions=data.get('logic_postconditions', [])
        )
        
        instructions = Instructions(
            trigger_keywords=data.get('trigger_keywords', []),
            usage_examples=data.get('usage_examples', []),
            agent_guidance=data.get('agent_guidance', ''),
            confidence_threshold=data.get('confidence_threshold', 0.5),
            priority=data.get('priority', 5)
        )
        
        evolution = Evolution(
            program_id=data.get('program_id', ''),
            version=data.get('version', 1),
            parent_id=data.get('parent_id'),
            root_id=data.get('root_id'),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            fitness_score=data.get('fitness_score', 0.5),
            episode_count=data.get('episode_count', 0),
            success_count=data.get('success_count', 0),
            failure_count=data.get('failure_count', 0),
            avg_quality=data.get('avg_quality', 0.5),
            avg_latency=data.get('avg_latency', 0.0),
            avg_tokens=data.get('avg_tokens', 0.0),
            confidence=data.get('confidence', 0.0),
            mutation_count=data.get('mutation_count', 0),
            last_mutation=data.get('last_mutation'),
            lineage=data.get('lineage', []),
            mutation_history=data.get('mutation_history', [])
        )
        
        quality = QualityGates(
            compile_status=data.get('compile_status', 'pending'),
            runtime_status=data.get('runtime_status', 'pending'),
            logic_status=data.get('logic_status', 'pending'),
            quality_status=data.get('quality_status', 'pending')
        )
        
        return MemoryProgram(
            program_id=data.get('program_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            schema=schema,
            logic=logic,
            instructions=instructions,
            evolution=evolution,
            quality=quality,
            source_system=data.get('source_system', ''),
            source_file=data.get('source_file', ''),
            status=ProgramStatus(data.get('status', 'active')) if data.get('status') else ProgramStatus.ACTIVE,
            version=data.get('version', 1),
            tags=data.get('tags', []),
            category=data.get('category', ''),
            domain=TaskDomain(data.get('domain', 'general')) if data.get('domain') else TaskDomain.GENERAL,
            pin=data.get('pin', False),
            created_at=data.get('created_at', ''),
            last_activity_at=data.get('last_activity_at', ''),
            stale_at=data.get('stale_at', ''),
            archived_at=data.get('archived_at', ''),
            parent_skill_id=data.get('parent_skill_id'),
            absorbed_into=data.get('absorbed_into'),
            is_umbrella=data.get('is_umbrella', False),
            children_ids=data.get('children_ids', []),
            quality_gates_passed=data.get('quality_gates_passed', False)
        )
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取技能排行榜
        
        Args:
            limit: 返回数量限制
            
        Returns:
            排行榜数据列表
        """
        with self._lock:
            try:
                rows = self.conn.execute("""
                    SELECT 
                        program_id, name, fitness_score, episode_count,
                        success_count, avg_quality, avg_latency, confidence,
                        status, priority, mutation_count
                    FROM skill_programs
                    WHERE status = 'active'
                    ORDER BY fitness_score DESC
                    LIMIT ?
                """, (limit,)).fetchall()
                
                leaderboard = []
                for rank, row in enumerate(rows, 1):
                    leaderboard.append({
                        'rank': rank,
                        'program_id': row[0],
                        'name': row[1],
                        'fitness_score': row[2],
                        'episode_count': row[3],
                        'success_count': row[4],
                        'avg_quality': row[5],
                        'avg_latency': row[6],
                        'confidence': row[7],
                        'status': row[8],
                        'priority': row[9],
                        'mutation_count': row[10],
                        'success_rate': row[4] / row[3] if row[3] > 0 else 0.0
                    })
                
                return leaderboard
                
            except Exception as e:
                logger.error(f"获取排行榜失败: {e}")
                return []
    
    def _load_recent_events(self, limit: int = 500) -> List[Dict[str, Any]]:
        """加载最近的 fitness_events（供 ValidationSet 使用）"""
        with self._lock:
            try:
                rows = self.conn.execute("""
                    SELECT id, program_id, event_type, success, quality,
                           latency, confidence, fitness_score, tokens_consumed,
                           error, session_id, timestamp, details
                    FROM fitness_events
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,)).fetchall()

                columns = [desc[0] for desc in self.conn.execute(
                    "SELECT * FROM fitness_events LIMIT 0").description]

                events = []
                for row in rows:
                    data = dict(zip(columns, row))
                    # Convert SQLite int (0/1) to bool
                    data["success"] = bool(data.get("success"))
                    events.append(data)
                return events
            except Exception as e:
                logger.error(f"_load_recent_events failed: {e}")
                return []

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
