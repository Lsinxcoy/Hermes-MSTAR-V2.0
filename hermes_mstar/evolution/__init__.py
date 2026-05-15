"""Evolution — Fitness tracking, mutation, quality gates, reflection, engine, population, validation"""
from .fitness_tracker import FitnessTracker, EpisodeResult
from .mutator import MSTARMutator, MutationType, MutationResult
from .quality_gates import QualityGates, GateResult, GateReport, get_quality_gates
from .reflection import MSTARReflector, FailurePattern, MutationProposal, ReflectionResult
from .evolution_engine import EvolutionEngine
from .forgetting import ForgettingMechanism, ForgetCandidate, ForgetDecision
from .population_pool import PopulationPool, PopulationMember, SelectionStrategy
from .validation_set import ValidationSet, ValidationEpisode, extract_episode_features
from .reflector_agent import LLMReflector, ReflectorAgent, LLMReflectionResult
from .task_domain import TaskDomain, FitnessWeights, detect_domain

__all__ = [
    'FitnessTracker', 'EpisodeResult',
    'MSTARMutator', 'MutationType', 'MutationResult',
    'QualityGates', 'GateResult', 'GateReport', 'get_quality_gates',
    'MSTARReflector', 'FailurePattern', 'MutationProposal', 'ReflectionResult',
    'EvolutionEngine',
    'ForgettingMechanism', 'ForgetCandidate', 'ForgetDecision',
    'PopulationPool', 'PopulationMember', 'SelectionStrategy',
    'ValidationSet', 'ValidationEpisode', 'extract_episode_features',
    'LLMReflector', 'ReflectorAgent', 'LLMReflectionResult',
    'TaskDomain', 'FitnessWeights', 'detect_domain',
]
