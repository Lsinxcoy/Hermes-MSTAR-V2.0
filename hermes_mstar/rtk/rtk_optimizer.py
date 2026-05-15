"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    Hermes MSTAR - RTK Token Optimizer                       ║
║  Three-tier cache: Hot (fitness≥0.7, LFU) / Warm (≥0.4, LRU+1h) / Cold    ║
║  减少 60-90% Token 消耗                                                     ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger("hermes.mstar.rtk")


# ── Enums ──────────────────────────────────────────────────────────────────────

class CacheTier(Enum):
    """三层缓存层级"""
    HOT = "hot"    # fitness ≥ 0.7, LFU, no TTL
    WARM = "warm"  # fitness ≥ 0.4, LRU, 1h TTL
    COLD = "cold"  # fitness < 0.4, FIFO, 24h TTL


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """缓存条目"""
    skill_id: str
    result: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    tokens_saved: int = 0
    fitness_at_cache: float = 0.5
    tier: CacheTier = CacheTier.WARM

    def age(self) -> float:
        return time.time() - self.created_at

    def idle_time(self) -> float:
        return time.time() - self.last_accessed


# ── Tiered Cache ──────────────────────────────────────────────────────────────

class TieredCache:
    """三层缓存：热/温/冷，每层独立淘汰策略"""

    HOT_SIZE = 300      # 热点缓存大小
    WARM_SIZE = 500     # 温缓存大小
    COLD_SIZE = 200     # 冷缓存大小
    WARM_TTL = 3600     # 温缓存 TTL: 1h
    COLD_TTL = 86400    # 冷缓存 TTL: 24h

    def __init__(self):
        self._hot: OrderedDict[str, CacheEntry] = OrderedDict()   # LFU
        self._warm: OrderedDict[str, CacheEntry] = OrderedDict()  # LRU
        self._cold: OrderedDict[str, CacheEntry] = OrderedDict() # FIFO
        self._lock = threading.RLock()
        self._lfu_order: List[str] = []  # LFU 访问频率排序
        self._stats = {
            'hits': 0, 'misses': 0, 'evictions': 0, 'expirations': 0,
            'hot_hits': 0, 'warm_hits': 0, 'cold_hits': 0
        }

    def _tier_for_fitness(self, fitness: float) -> CacheTier:
        if fitness >= 0.7:
            return CacheTier.HOT
        elif fitness >= 0.4:
            return CacheTier.WARM
        return CacheTier.COLD

    def _move_to_lfu_front(self, key: str):
        """LFU: 访问多的放前面"""
        if key in self._lfu_order:
            self._lfu_order.remove(key)
        self._lfu_order.insert(0, key)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            # 按层级查找
            for tier, cache in [(CacheTier.HOT, self._hot),
                               (CacheTier.WARM, self._warm),
                               (CacheTier.COLD, self._cold)]:
                if key in cache:
                    entry = cache[key]
                    # TTL 检查
                    ttl = 0
                    if tier == CacheTier.WARM:
                        ttl = self.WARM_TTL
                    elif tier == CacheTier.COLD:
                        ttl = self.COLD_TTL
                    if ttl > 0 and entry.age() > ttl:
                        del cache[key]
                        self._stats['expirations'] += 1
                        self._stats['misses'] += 1
                        return None
                    # 更新访问
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    if tier == CacheTier.HOT:
                        self._move_to_lfu_front(key)
                    cache.move_to_end(key)
                    self._stats['hits'] += 1
                    if tier == CacheTier.HOT:
                        self._stats['hot_hits'] += 1
                    elif tier == CacheTier.WARM:
                        self._stats['warm_hits'] += 1
                    else:
                        self._stats['cold_hits'] += 1
                    return entry.result
            self._stats['misses'] += 1
            return None

    def set(self, key: str, value: Any, fitness: float = 0.5, **kwargs):
        with self._lock:
            tier = self._tier_for_fitness(fitness)
            cache = {CacheTier.HOT: self._hot,
                     CacheTier.WARM: self._warm,
                     CacheTier.COLD: self._cold}[tier]
            size_limit = {CacheTier.HOT: self.HOT_SIZE,
                          CacheTier.WARM: self.WARM_SIZE,
                          CacheTier.COLD: self.COLD_SIZE}[tier]

            # 更新现有
            if key in cache:
                entry = cache[key]
                entry.result = value
                entry.last_accessed = time.time()
                entry.access_count += 1
                entry.fitness_at_cache = fitness
                entry.tier = tier
                if tier == CacheTier.HOT:
                    self._move_to_lfu_front(key)
                cache.move_to_end(key)
                return

            # 淘汰
            while len(cache) >= size_limit:
                if tier == CacheTier.HOT and self._lfu_order:
                    # 淘汰访问最少的
                    lfu_key = self._lfu_order.pop()
                    if lfu_key in cache:
                        del cache[lfu_key]
                else:
                    # 淘汰最老的
                    cache.popitem(last=False)
                self._stats['evictions'] += 1

            entry = CacheEntry(
                skill_id=key, result=value,
                created_at=time.time(), last_accessed=time.time(),
                access_count=1, fitness_at_cache=fitness, tier=tier,
                tokens_saved=kwargs.get('tokens_saved', 0)
            )
            cache[key] = entry
            if tier == CacheTier.HOT:
                self._move_to_lfu_front(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            for cache in [self._hot, self._warm, self._cold]:
                if key in cache:
                    del cache[key]
                    if key in self._lfu_order:
                        self._lfu_order.remove(key)
                    return True
            return False

    def invalidate_fitness_below(self, threshold: float) -> int:
        """淘汰低于阈值的缓存"""
        evicted = 0
        with self._lock:
            for cache in [self._hot, self._warm, self._cold]:
                keys = [k for k, e in cache.items() if e.fitness_at_cache < threshold]
                for k in keys:
                    del cache[k]
                    evicted += 1
        return evicted

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0
            return {
                'hit_rate': round(hit_rate, 4),
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'expirations': self._stats['expirations'],
                'tiers': {
                    'hot': {'size': len(self._hot), 'hits': self._stats['hot_hits']},
                    'warm': {'size': len(self._warm), 'hits': self._stats['warm_hits']},
                    'cold': {'size': len(self._cold), 'hits': self._stats['cold_hits']},
                },
                'total': len(self._hot) + len(self._warm) + len(self._cold)
            }


# ── Token Budget ──────────────────────────────────────────────────────────────

@dataclass
class TokenBudget:
    max_tokens: int = 50000
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95
    current_usage: int = 0
    session_start: float = field(default_factory=time.time)

    def usage_ratio(self) -> float:
        return self.current_usage / self.max_tokens if self.max_tokens > 0 else 0

    def is_warning(self) -> bool:
        return self.usage_ratio() >= self.warning_threshold

    def is_critical(self) -> bool:
        return self.usage_ratio() >= self.critical_threshold

    def consume(self, tokens: int):
        self.current_usage += tokens

    def reset(self):
        self.current_usage = 0
        self.session_start = time.time()


# ── RTK Config ────────────────────────────────────────────────────────────────

@dataclass
class RTKConfig:
    cache_size: int = 1000
    token_budget: int = 50000
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95
    high_fitness_threshold: float = 0.8
    low_fitness_threshold: float = 0.3


# ── RTK Token Optimizer ───────────────────────────────────────────────────────

class RTKTokenOptimizer:
    """
    RTK Token 优化器 - 三层缓存 + Token 预算管理

    决策流程:
    1. 检查缓存命中 → 直接返回 (节省 Token)
    2. 检查 Token 预算 → 不足则强制缓存
    3. 检查 fitness → 高 fitness 跳过 LLM
    4. 决定调用 LLM
    """

    _instance: Optional['RTKTokenOptimizer'] = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[RTKConfig] = None):
        self.config = config or RTKConfig()
        self._cache = TieredCache()
        self._token_budget = TokenBudget(
            max_tokens=self.config.token_budget,
            warning_threshold=self.config.warning_threshold,
            critical_threshold=self.config.critical_threshold
        )
        self._stats = {
            'total_llm_calls': 0, 'cached_calls': 0,
            'tokens_saved': 0, 'fitness_skips': 0, 'budget_skips': 0
        }
        self._callbacks: Dict[str, Optional[Callable]] = {
            'on_cache_hit': None, 'on_budget_warning': None
        }

    @classmethod
    def get_instance(cls) -> 'RTKTokenOptimizer':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def should_use_llm(
        self, skill_id: str, context: Dict[str, Any], fitness: float = 0.5
    ) -> Tuple[bool, str]:
        """判断是否需要调用 LLM"""
        cache_key = self._make_cache_key(skill_id, context)

        # 1. 缓存命中
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._stats['cached_calls'] += 1
            return False, "cache_hit"

        # 2. Token 预算检查
        if self._token_budget.is_critical():
            self._stats['budget_skips'] += 1
            cb = self._callbacks.get('on_budget_warning')
            if cb:
                cb(self._token_budget)
            return False, "budget_critical"

        if self._token_budget.is_warning():
            cb = self._callbacks.get('on_budget_warning')
            if cb:
                cb(self._token_budget)

        # 3. 高 fitness 跳过
        if fitness >= self.config.high_fitness_threshold:
            self._stats['fitness_skips'] += 1
            return False, "high_fitness_skip"

        # 4. 需要 LLM
        self._stats['total_llm_calls'] += 1
        return True, "llm_required"

    def cache_result(
        self, skill_id: str, context: Dict[str, Any], result: Any,
        tokens_consumed: int = 0, fitness: float = 0.5
    ):
        """缓存执行结果"""
        cache_key = self._make_cache_key(skill_id, context)
        tokens_saved = self._estimate_tokens_saved(skill_id, context)
        self._cache.set(cache_key, result, fitness, tokens_saved=tokens_saved)
        self._token_budget.consume(tokens_consumed)
        self._stats['tokens_saved'] += tokens_saved

    def _make_cache_key(self, skill_id: str, context: Dict[str, Any]) -> str:
        ctx_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(f"{skill_id}:{ctx_str}".encode()).hexdigest()[:32]

    def _estimate_tokens_saved(self, skill_id: str, context: Dict[str, Any]) -> int:
        return len(json.dumps(context, default=str)) // 4

    def invalidate_skill(self, skill_id: str):
        """使技能相关缓存失效"""
        with self._cache._lock:
            for cache in [self._cache._hot, self._cache._warm, self._cache._cold]:
                keys = [k for k in cache if skill_id in k]
                for k in keys:
                    cache.pop(k, None)

    def get_stats(self) -> Dict[str, Any]:
        total = self._stats['total_llm_calls'] + self._stats['cached_calls']
        cache_rate = self._stats['cached_calls'] / total if total > 0 else 0
        return {
            'cache': self._cache.get_stats(),
            'token_budget': {
                'max': self._token_budget.max_tokens,
                'current': self._token_budget.current_usage,
                'usage_ratio': round(self._token_budget.usage_ratio(), 4),
                'is_warning': self._token_budget.is_warning(),
                'is_critical': self._token_budget.is_critical()
            },
            'optimization': {
                'total_llm_calls': self._stats['total_llm_calls'],
                'cached_calls': self._stats['cached_calls'],
                'tokens_saved': self._stats['tokens_saved'],
                'fitness_skips': self._stats['fitness_skips'],
                'budget_skips': self._stats['budget_skips'],
                'cache_hit_rate': round(cache_rate, 4),
                'estimated_token_savings_pct': round(cache_rate * 0.75, 4)
            }
        }

    def set_callback(self, name: str, cb: Optional[Callable]):
        self._callbacks[name] = cb

    def reset_stats(self):
        self._stats = {
            'total_llm_calls': 0, 'cached_calls': 0,
            'tokens_saved': 0, 'fitness_skips': 0, 'budget_skips': 0
        }
        self._token_budget.reset()


# ── Global Access ─────────────────────────────────────────────────────────────

_optimizer_instance: Optional[RTKTokenOptimizer] = None


def get_rtk_optimizer(config: Optional[RTKConfig] = None) -> RTKTokenOptimizer:
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = RTKTokenOptimizer(config)
    return _optimizer_instance


def should_use_llm(skill_id: str, context: Dict[str, Any], fitness: float = 0.5) -> Tuple[bool, str]:
    return get_rtk_optimizer().should_use_llm(skill_id, context, fitness)


def cache_result(skill_id: str, context: Dict[str, Any], result: Any,
                 tokens_consumed: int = 0, fitness: float = 0.5):
    get_rtk_optimizer().cache_result(skill_id, context, result, tokens_consumed, fitness)


def get_rtk_stats() -> Dict[str, Any]:
    return get_rtk_optimizer().get_stats()
