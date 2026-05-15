"""
╔══════════════════════════════════════════════════════════════════════╗
║           Hermes MSTAR — 统一记忆 + RTK Token 优化系统            ║
║  基于 MSTAR (arxiv:2604.11811) + RTK CLI 优化理念                ║
╚══════════════════════════════════════════════════════════════════════╝

Architecture:
  HermesMSTARProvider (MemoryProvider subclass)
    ├── FitnessTracker (SQLite, multi-factor fitness)
    ├── RTKTokenOptimizer (三层缓存)
    ├── SkillCache (三层技能缓存)
    ├── EvolutionEngine (自适应间隔)
    ├── MSTARMutator (15/15 全激活)
    ├── MSTARReflector (FailurePattern 分析)
    ├── QualityGates (Python+Shell+PowerShell+YAML)
    ├── ForgettingMechanism (archive/merge 策略)
    └── Hermes增强: MemoryRouter, TaskClassifier, ActiveLearner...
"""

__version__ = "2.0.0"

from .memory_program import (
    MemoryProgram, EpisodeResult,
    StorageType, TaskDomain, ProgramStatus, FitnessLevel,
    MutationType,
    Evolution, QualityGates,
    init_mstar_integration, get_mstar_integration
)

from .config import HermesMSTARConfig, default_config

from .hermes_provider import HermesMSTARProvider

from .self_improving_bridge import SelfImprovingBridge, get_bridge

__all__ = [
    'MemoryProgram', 'EpisodeResult',
    'StorageType', 'TaskDomain', 'ProgramStatus', 'FitnessLevel', 'MutationType',
    'Evolution', 'QualityGates',
    'HermesMSTARProvider',
    'SelfImprovingBridge', 'get_bridge',
    'HermesMSTARConfig', 'default_config',
    'init_mstar_integration', 'get_mstar_integration',
    '__version__',
]
