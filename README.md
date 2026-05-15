# Hermes MSTAR V2.0

**Hermes M* Paper Implementation — Production Grade**

基于论文 [M* (arXiv:2604.11811)](https://arxiv.org/abs/2604.11811) 实现的自主进化 Skill 系统，已集成到 Hermes Agent v0.40.0。

## 系统架构

```
Runtime → record_execution() → EpisodeResult → update_fitness()
                                          ↓
                                FitnessTracker DB
                                          ↓
                        ┌─────────────────┴─────────────────┐
                        ↓                                   ↓
                 PopulationPool                     ValidationSet
                 (Softmax τ=0.15)                    (k-means k=25)
                        ↓                                   ↓
                 LLMReflector.analyze() ← failed_episodes
                        ↓
                 MSTARMutator.mutate() → MutationResult
                        ↓
                 QualityGates.run_all() ← 3× Repair Loop
                        ↓
                   ┌────┴────┐
                PASS      FAIL → retry ×3
```

## 5-Phase 进化循环

| Phase | 组件 | 文件 | M* Paper 对应 |
|-------|------|------|-------------|
| P1 | PopulationPool (τ=0.15, max=20) | `population_pool.py` | Softmax Selection + 种群池 |
| P2 | ValidationSet (k=25, 7维特征) | `validation_set.py` | k-means Validation Set |
| P3 | LLMReflector (GPT-4驱动) | `reflector_agent.py` | LLM-driven Reflection |
| P4 | QualityGates 3× Repair Loop | `quality_gates.py` | Automated 3× Fix |
| P5 | Task-Specific Fitness (4域) | `task_domain.py` | Domain-Aware Fitness |

## 核心模块

### Population Pool (Phase 1)
- `PopulationPool`: Softmax Selection (τ=0.15), Tournament, Roulette
- `PopulationMember`: program_id, fitness, lineage, mutation_history
- max_size=20, min_size=5, diversity enforcement via lineage_distance()

### Validation Set (Phase 2)
- `ValidationSet`: k-means clustering, k=25 episodes
- 7-dim feature vector: `[success, quality, log(tokens), log(latency), recency_days, error_rate, code_complexity]`
- Rotating validation set to prevent overfitting

### LLM Reflector (Phase 3)
- `LLMReflector`: GPT-4 driven failure analysis
- `LLMReflectionResult`: 9 fields including failure_patterns, mutation_type, specific_change

### Quality Gates (Phase 4)
- `QualityGates`: 4-gate pipeline (COMPILE → RUNTIME → LOGIC → QUALITY)
- 3× Automated Repair Loop per gate
- `GateReport`: passed, repair_attempts, last_error, failed_gates

### Task-Specific Fitness (Phase 5)
- `TaskDomain`: CODING, RESEARCH, WRITING, GENERAL
- `FitnessWeights`: per-domain success/quality/latency weights
- Phase-5 fitness formula: `fitness = base × (1+latency_factor) × token_factor × time_decay`

## 安装

```bash
pip install numpy scikit-learn
```

```python
from hermes_mstar.evolution import (
    FitnessTracker, PopulationPool, ValidationSet,
    LLMReflector, QualityGates, EvolutionEngine,
    ForgettingMechanism, TaskDomain, FitnessWeights
)
```

## 验证

```bash
python verify_full.py   # 46 PASS / 0 BUG
```

## 文件结构

```
hermes_mstar/
├── __init__.py
├── memory_program.py          # MemoryProgram + Phase5 fitness
├── evolution/
│   ├── __init__.py            # 10 modules exported
│   ├── population_pool.py    # [NEW] Softmax Selection Pool
│   ├── validation_set.py     # [NEW] k-means Validation Set
│   ├── reflector_agent.py    # [NEW] LLM Reflector
│   ├── task_domain.py         # [NEW] Task-Specific Fitness
│   ├── quality_gates.py       # [重写] 3× Repair Loop
│   ├── mutator.py             # [Bug Fix] MutationResult.success=True
│   ├── evolution_engine.py    # [改造] 集成 PopulationPool
│   ├── fitness_tracker.py     # [扩展] _load_recent_events()
│   ├── reflection.py
│   └── forgetting.py
└── rtk/
    ├── __init__.py
    └── rtk_optimizer.py       # RTK Token Optimizer
```

## Bug Fixes

- `MutationResult.success`: 从无默认值 → `True` (class variable)
- `FitnessTracker.get_all_skills()`: 从空列表 → 返回9个活跃skill

## 依赖

- Python 3.11+
- numpy
- scikit-learn
- hermes_mstar >= 2.0.0

## License

MIT
