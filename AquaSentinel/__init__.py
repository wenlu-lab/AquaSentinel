"""AquaSentinel: physics-informed AI system for real-time anomaly
detection in urban underground water pipeline networks."""

from .graph import PipelineNetwork, PipeAttributes
from .sensor_placement import PlacementConfig, deploy
from .physics import PhysicsAugmenter, PhysicsConfig
from .dataset import Scaler, SlidingWindowDataset, train_val_test_split
from .moe import MixtureOfExperts, MoEConfig, build_experts
from .rtca import RTCAConfig, RTCADetector
from .localization import localize, find_source_nodes
from .severity import assess, classify_severity
from .llm_agent import LLMConfig, ReportAgent

__version__ = "1.0.0"

__all__ = [
    "PipelineNetwork", "PipeAttributes",
    "PlacementConfig", "deploy",
    "PhysicsAugmenter", "PhysicsConfig",
    "Scaler", "SlidingWindowDataset", "train_val_test_split",
    "MixtureOfExperts", "MoEConfig", "build_experts",
    "RTCAConfig", "RTCADetector",
    "localize", "find_source_nodes",
    "assess", "classify_severity",
    "LLMConfig", "ReportAgent",
]
