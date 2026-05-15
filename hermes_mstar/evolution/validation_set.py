"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║           Hermes MSTAR: Validation Set — M* Paper k-means Validation Set Construction                             ║
║                                                                                                              ║
║  Ported from M* paper (arXiv:2604.11811) Section 3.2: Reflective Code Evolution                                ║
║                                                                                                              ║
║  Key design:                                                                                                ║
║    - Static validation set: unchanged across iterations → comparable scores                                   ║
║    - Rotating validation set: replaced each iteration → prevent information leakage                           ║
║    - k-means clustering (k=25) for representative episode selection                                           ║
║    - Facility Location optimization for episode subset selection (optional, NP-hard)                          ║
║                                                                                                              ║
║  vs Random selection: k-means gives more stable fitness signals → better evolution direction                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fitness_tracker import FitnessTracker

logger = logging.getLogger("hermes.mstar.validation_set")


# ── Feature extraction ──────────────────────────────────────────────────────────────────────────────

def extract_episode_features(event: Dict[str, Any]) -> List[float]:
    """
    从 fitness_event 提取特征向量，用于 k-means 聚类

    特征维度 (7-dim):
      0. success_rate (0/1)
      1. quality_normalized (0.0-1.0)
      2. latency_log (log(1 + latency_seconds))
      3. tokens_log (log(1 + tokens/1000))
      4. confidence (0.0-1.0)
      5. hour_of_day_normalized (0.0-1.0, cyclic)
      6. days_since_epoch_normalized
    """
    import math

    def safe(val, default=0.0):
        return float(val) if val is not None and isinstance(val, (int, float)) and math.isfinite(val) else default

    # Basic features
    success = 1.0 if event.get("success") else 0.0
    quality = safe(event.get("quality", 0.5))
    latency = safe(event.get("latency", 0.0))
    tokens = safe(event.get("tokens_consumed", 0), 0.0)
    confidence = safe(event.get("confidence", 0.5))

    # Log-transformed
    latency_log = math.log(1 + latency) if latency >= 0 else 0.0
    tokens_log = math.log(1 + tokens / 1000.0) if tokens >= 0 else 0.0

    # Temporal
    timestamp_str = event.get("timestamp", "")
    if timestamp_str:
        try:
            dt = datetime.fromisoformat(timestamp_str)
            hour_norm = dt.hour / 24.0
            days_since_epoch = (dt - datetime(2020, 1, 1)).days
            days_norm = days_since_epoch / 2000.0  # ~5.5 years range
        except Exception:
            hour_norm = 0.5
            days_norm = 0.5
    else:
        hour_norm = 0.5
        days_norm = 0.5

    return [
        success,
        quality,
        latency_log,
        tokens_log,
        confidence,
        hour_norm,
        days_norm,
    ]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def kmeans_plusplus(features: List[List[float]], k: int, seed: int = 42) -> List[int]:
    """
    k-means++ 初始化（M* paper 未指定，用标准 k-means++）

    相比随机初始化，k-means++ 能更好地覆盖特征空间
    返回: k 个中心点的索引
    """
    import math
    random.seed(seed)
    np.random.seed(seed)

    n = len(features)
    if n <= k:
        return list(range(n))

    # 第一个中心：随机选
    centers = [random.randint(0, n - 1)]

    for _ in range(k - 1):
        # 计算每个点到最近中心的距离平方
        distances = []
        for i in range(n):
            d = min(
                sum((features[i][j] - features[c][j]) ** 2 for j in range(len(features[i])))
                for c in centers
            )
            distances.append(d)

        # 加权概率选择下一个中心（D^2 sampling）
        total = sum(distances)
        if total <= 0:
            next_c = random.randint(0, n - 1)
        else:
            probs = [d / total for d in distances]
            next_c = random.choices(range(n), weights=probs, k=1)[0]
        centers.append(next_c)

    return centers


def assign_clusters(
    features: List[List[float]],
    centers: List[List[float]]
) -> List[int]:
    """将每个点分配到最近的中心"""
    import math
    assignments = []
    for f in features:
        min_dist = float("inf")
        best_cluster = 0
        for ci, c in enumerate(centers):
            dist = math.sqrt(sum((fj - cj) ** 2 for fj, cj in zip(f, c)))
            if dist < min_dist:
                min_dist = dist
                best_cluster = ci
        assignments.append(best_cluster)
    return assignments


def recompute_centers(
    features: List[List[float]],
    assignments: List[int],
    k: int
) -> List[List[float]]:
    """重新计算每个簇的中心"""
    n_features = len(features[0]) if features else 0
    new_centers = [[0.0] * n_features for _ in range(k)]
    counts = [0] * k

    for f, ci in zip(features, assignments):
        for j, v in enumerate(f):
            new_centers[ci][j] += v
        counts[ci] += 1

    for ci in range(k):
        if counts[ci] > 0:
            new_centers[ci] = [v / counts[ci] for v in new_centers[ci]]

    return new_centers


def kmeans(
    features: List[List[float]],
    k: int,
    max_iters: int = 30,
    seed: int = 42
) -> Tuple[List[int], List[List[float]]]:
    """
    标准 k-means 实现

    Returns:
        (assignments, centers)
        assignments: 每个点的簇ID列表
        centers: k 个中心点
    """
    if not features:
        return [], []

    n = len(features)
    if n <= k:
        assignments = list(range(n))
        return assignments, features.copy()

    # k-means++ 初始化
    center_indices = kmeans_plusplus(features, k, seed)
    centers = [features[i] for i in center_indices]

    for _ in range(max_iters):
        # E步：分配
        new_assignments = assign_clusters(features, centers)

        # M步：更新中心
        new_centers = recompute_centers(features, new_assignments, k)

        # 检查收敛
        if new_centers == centers:
            break
        centers = new_centers

    return new_assignments, centers


# ── Validation Episode ────────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationEpisode:
    """
    单个验证episode（M* paper: validation set 中的一个item）

    对应 M* paper 的 "episode" 概念：
    - 在验证集上运行当前 memory program → 得到反馈 R
    - R 用于指导进化方向
    """
    episode_id: str
    program_id: str
    features: List[float]           # 7-dim feature vector
    success: bool
    quality: float
    latency: float
    tokens_consumed: int
    session_id: str
    timestamp: str

    # Cluster info (filled after k-means)
    cluster_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "program_id": self.program_id,
            "features": self.features,
            "success": self.success,
            "quality": self.quality,
            "latency": self.latency,
            "tokens_consumed": self.tokens_consumed,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "cluster_id": self.cluster_id,
        }


# ── Validation Set ─────────────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationSet:
    """
    M* Paper Validation Set

    设计（M* paper Section 3.2）:
      - Static set: 25 items，跨迭代不变（用于可比分数）
      - Rotating set: 每次迭代替换一部分（防止信息泄露）

    论文原文:
      "Static validation set remains unchanged across iterations to enable
       comparable scores; rotating validation set is replaced each iteration
       to prevent information leakage"
    """

    static_episodes: List[ValidationEpisode] = field(default_factory=list)
    rotating_episodes: List[ValidationEpisode] = field(default_factory=list)

    # k-means 参数
    k_static: int = 25          # M* paper: k=25
    k_rotating: int = 10        # Rotating set 大小
    n_features: int = 7          # 特征维度

    # 构建元数据
    built_at: str = ""
    seed: int = 42
    version: int = 0  # 每次重建 +1

    @classmethod
    def build(
        cls,
        fitness_tracker: "FitnessTracker",
        k_static: int = 25,
        k_rotating: int = 10,
        seed: int = 42,
    ) -> "ValidationSet":
        """
        从历史 fitness_events 构建验证集

        M* paper: k-means clustering 选择最具代表性的 25 个 episodes
        """
        import uuid

        # 加载历史 events
        events = fitness_tracker._load_recent_events(limit=500)
        if not events:
            logger.warning("No events found for validation set, returning empty set")
            return cls(built_at=datetime.now().isoformat(), seed=seed, k_static=k_static, k_rotating=k_rotating)

        # 提取特征
        episodes: List[ValidationEpisode] = []
        for event in events:
            try:
                features = extract_episode_features(event)
                ep = ValidationEpisode(
                    episode_id=event.get("id", str(uuid.uuid4())),
                    program_id=event.get("program_id", "unknown"),
                    features=features,
                    success=bool(event.get("success")),
                    quality=float(event.get("quality", 0.5)),
                    latency=float(event.get("latency", 0.0)),
                    tokens_consumed=int(event.get("tokens_consumed", 0)),
                    session_id=event.get("session_id", ""),
                    timestamp=event.get("timestamp", ""),
                )
                episodes.append(ep)
            except Exception as e:
                logger.debug(f"Skipping event: {e}")
                continue

        if len(episodes) < k_static:
            logger.warning(f"Only {len(episodes)} episodes, less than k_static={k_static}")
            return cls(
                static_episodes=episodes,
                rotating_episodes=[],
                built_at=datetime.now().isoformat(),
                seed=seed,
                k_static=k_static,
                k_rotating=k_rotating,
            )

        # k-means 聚类
        features_all = [e.features for e in episodes]
        assignments, centers = kmeans(features_all, k=k_static, seed=seed)

        # 为每个 episode 设置 cluster_id
        for ep, ci in zip(episodes, assignments):
            ep.cluster_id = ci

        # 从每个簇中选择最接近中心的 episode → static set
        static_episodes: List[ValidationEpisode] = []
        clusters: Dict[int, List[ValidationEpisode]] = {}
        for ep in episodes:
            cid = ep.cluster_id
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(ep)

        for cid, members in clusters.items():
            if not members:
                continue
            # 选最接近中心的
            center = centers[cid]
            best = min(members, key=lambda e: sum((fj - cj) ** 2 for fj, cj in zip(e.features, center)))
            static_episodes.append(best)

        # 确保 static set 大小合适
        static_episodes = static_episodes[:k_static]

        # Rotating set: 随机选择非 static 的 episodes
        non_static = [e for e in episodes if e not in static_episodes]
        random.seed(seed)
        rotating_episodes = random.sample(non_static, min(k_rotating, len(non_static)))

        vs = cls(
            static_episodes=static_episodes,
            rotating_episodes=rotating_episodes,
            built_at=datetime.now().isoformat(),
            seed=seed,
            k_static=k_static,
            k_rotating=k_rotating,
        )
        logger.info(f"Built ValidationSet: static={len(static_episodes)}, rotating={len(rotating_episodes)}")
        return vs

    def rotate(self, fitness_tracker: "FitnessTracker") -> "ValidationSet":
        """
        轮换 rotating set（M* paper: 每次迭代替换一部分）

        Static set 保持不变，rotating set 从新事件中重新采样
        """
        new_rotating = ValidationSet._sample_rotating(fitness_tracker, self.k_rotating, self.seed)
        return ValidationSet(
            static_episodes=self.static_episodes,  # 不变
            rotating_episodes=new_rotating,
            built_at=datetime.now().isoformat(),
            seed=self.seed,
            k_static=self.k_static,
            k_rotating=self.k_rotating,
            version=self.version + 1,
        )

    @staticmethod
    def _sample_rotating(fitness_tracker: "FitnessTracker", k: int, seed: int) -> List[ValidationEpisode]:
        """从最近事件中采样新的 rotating episodes"""
        import uuid
        events = fitness_tracker._load_recent_events(limit=200)
        random.seed(seed)
        sampled = random.sample(events, min(k, len(events)))

        episodes = []
        for event in sampled:
            try:
                ep = ValidationEpisode(
                    episode_id=event.get("id", str(uuid.uuid4())),
                    program_id=event.get("program_id", "unknown"),
                    features=extract_episode_features(event),
                    success=bool(event.get("success")),
                    quality=float(event.get("quality", 0.5)),
                    latency=float(event.get("latency", 0.0)),
                    tokens_consumed=int(event.get("tokens_consumed", 0)),
                    session_id=event.get("session_id", ""),
                    timestamp=event.get("timestamp", ""),
                )
                episodes.append(ep)
            except Exception:
                continue
        return episodes

    def all_episodes(self) -> List[ValidationEpisode]:
        """所有验证 episodes（static + rotating）"""
        return self.static_episodes + self.rotating_episodes

    def get_statistics(self) -> Dict[str, Any]:
        """验证集统计"""
        all_eps = self.all_episodes()
        if not all_eps:
            return {"size": 0}
        successes = [e.success for e in all_eps]
        qualities = [e.quality for e in all_eps]
        return {
            "static_size": len(self.static_episodes),
            "rotating_size": len(self.rotating_episodes),
            "total_size": len(all_eps),
            "success_rate": sum(successes) / len(successes),
            "avg_quality": sum(qualities) / len(qualities),
            "version": self.version,
            "built_at": self.built_at,
        }
