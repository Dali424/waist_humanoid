#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_hier_dual_hand")
@dataclass
class ACTHierDualHandConfig(PreTrainedConfig):
    """Hierarchical ACT dual: decoder1 waist+arm, decoder2 dual-cross hand."""

    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    waist_dim: int = 3
    arm_dim: int = 14
    hand_dim: int = 12
    waist_indices: list[int] = field(default_factory=lambda: [26, 27, 28])
    arm_indices: list[int] = field(default_factory=lambda: list(range(0, 14)))
    hand_indices: list[int] = field(default_factory=lambda: list(range(14, 26)))

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 7
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    temporal_ensemble_coeff: float | None = None

    dropout: float = 0.1
    kl_weight: float = 10.0

    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    robot_state_feature = None
    env_state_feature = None
    image_features: list[str] | None = None
    action_feature = None

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be a ResNet variant, got {self.vision_backbone}.")
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError("`n_action_steps` must be 1 when using temporal ensembling.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        has_visual = any(
            getattr(feat, "type", None) is FeatureType.VISUAL for feat in self.input_features.values()
        )
        has_env = "observation.environment_state" in self.input_features
        if not self.image_features and not self.env_state_feature and not (has_visual or has_env):
            raise ValueError("You must provide at least one image or environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
