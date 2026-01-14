#!/usr/bin/env python
# act_bide_adap.py

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION, OBS_IMAGES
# 기존 Config 활용 (혹은 별도 정의)
from lerobot.policies.act_bide_adap.configuration_act import ACTBideAdapConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy


class ACTBideAdapPolicy(PreTrainedPolicy):
    """
    Bidirectional ACT: Decodes Arm and Waist in PARALLEL, 
    then refines them using Gated Cross-Attention (Adap).
    """
    config_class = ACTBideAdapConfig
    name = "act_bide_adap"

    def __init__(self, config: ACTBideAdapConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)
        
        self.model = ACT(config)
        
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {"params": [p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad], "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions = self.model(batch)[0]
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        batch = self.normalize_targets(batch)
        
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
        
        l1_loss = (F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)).mean()
        loss_dict = {"l1_loss": l1_loss.item()}
        
        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss
        return loss, loss_dict


class ACTTemporalEnsembler:
    # (기존 코드와 동일하여 생략 가능하지만, 완전성을 위해 포함)
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()
    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None
    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones((self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device)
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat([self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])])
        action, self.ensembled_actions, self.ensembled_actions_count = (self.ensembled_actions[:, 0], self.ensembled_actions[:, 1:], self.ensembled_actions_count[1:])
        return action


# --- [NEW] Modules for Bidirectional Adap ---

class ContextGating(nn.Module):
    """Filters information based on relevance."""
    def __init__(self, dim_model):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim_model * 2, dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, dim_model),
            nn.Sigmoid()
        )
    def forward(self, target, source):
        # target: My current features (B, S, D)
        # source: Other agent's features (B, S, D)
        combined = torch.cat([target, source], dim=-1)
        alpha = self.gate_net(combined)
        return alpha * source

class BidirectionalGatedInteraction(nn.Module):
    """
    Exchanges information between Arm and Waist streams.
    Uses Gating to filter 'Source' info, then Attention to integrate it into 'Target'.
    """
    def __init__(self, config):
        super().__init__()
        self.dim_model = config.dim_model
        
        # Gates
        self.gate_waist_to_arm = ContextGating(config.dim_model)
        self.gate_arm_to_waist = ContextGating(config.dim_model)

        # Cross Attention Modules (Standard MHA)
        # Query = Self, Key/Value = Gated Other
        self.attn_arm_reads_waist = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.attn_waist_reads_arm = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        
        # Integration (Residual + Norm)
        self.norm_arm = nn.LayerNorm(config.dim_model)
        self.norm_waist = nn.LayerNorm(config.dim_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, arm_features, waist_features):
        # inputs: (S, B, D) - sequence first for MHA
        
        # 1. Prepare for Gating (Batch First preferred for simple concat logic in Gating)
        arm_b = arm_features.transpose(0, 1)   # (B, S, D)
        waist_b = waist_features.transpose(0, 1) # (B, S, D)

        # 2. Apply Gating (Adaptive Filtering)
        # "Arm wants to see relevant Waist info"
        gated_waist_b = self.gate_waist_to_arm(arm_b, waist_b) 
        # "Waist wants to see relevant Arm info"
        gated_arm_b = self.gate_arm_to_waist(waist_b, arm_b)

        # Back to Seq First for Attention
        gated_waist = gated_waist_b.transpose(0, 1) # (S, B, D)
        gated_arm = gated_arm_b.transpose(0, 1)     # (S, B, D)

        # 3. Apply Cross Attention (Bidirectional Update)
        
        # Update Arm: Q=Arm, K=GatedWaist, V=GatedWaist
        arm_update, _ = self.attn_arm_reads_waist(
            query=arm_features, 
            key=gated_waist, 
            value=gated_waist
        )
        # Update Waist: Q=Waist, K=GatedArm, V=GatedArm
        waist_update, _ = self.attn_waist_reads_arm(
            query=waist_features,
            key=gated_arm,
            value=gated_arm
        )

        # 4. Residual Connection & Norm
        new_arm = self.norm_arm(arm_features + self.dropout(arm_update))
        new_waist = self.norm_waist(waist_features + self.dropout(waist_update))

        return new_arm, new_waist


class ACT(nn.Module):
    def __init__(self, config: ACTBideAdapConfig):
        super().__init__()
        self.config = config
        
        # --- Standard Components (VAE, Backbone, Encoder) ---
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.dim_model)
            self.vae_encoder_action_input_proj = nn.Linear(self.config.action_feature.shape[0], config.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature: num_input_token_encoder += 1
            self.register_buffer("vae_encoder_pos_enc", create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0))

        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation], weights=config.pretrained_backbone_weights, norm_layer=FrozenBatchNorm2d)
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(config)

        # --- [Structure Change] Parallel Decoders ---
        # Instead of stacking, we have two independent decoders running in parallel
        self.arm_decoder = ACTDecoder(config)
        self.waist_decoder = ACTDecoder(config)

        # --- [Structure Change] Bidirectional Interaction Block ---
        self.interaction_block = BidirectionalGatedInteraction(config)

        # --- Projections ---
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.dim_model)
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_feature.shape[0], config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)
        
        n_1d_tokens = 1
        if self.config.robot_state_feature: n_1d_tokens += 1
        if self.config.env_state_feature: n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.register_buffer("decoder_in_template", torch.zeros(config.chunk_size, 1, config.dim_model), persistent=False)

        # Heads
        self.waist_head = nn.Linear(config.dim_model, self.config.waist_dim)
        self.arm_head = nn.Linear(config.dim_model, self.config.arm_dim)
        self.hand_head = nn.Linear(config.dim_model, self.config.hand_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.arm_decoder.parameters(), self.waist_decoder.parameters(), self.interaction_block.parameters()):
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        for head in (self.waist_head, self.arm_head, self.hand_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        # (Standard VAE & Input Processing - 생략된 부분 없이 이전 코드와 동일하게 작성하시면 됩니다)
        # ... [Input Processing Block Start] ...
        if self.config.use_vae and self.training: assert "action" in batch
        if "observation.images" in batch: batch_size = batch["observation.images"][0].shape[0]
        else: batch_size = batch["observation.environment_state"].shape[0]

        if self.config.use_vae and "action" in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"]).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch["action"])
            vae_input = [cls_embed, robot_state_embed, action_embed] if self.config.robot_state_feature else [cls_embed, action_embed]
            vae_input = torch.cat(vae_input, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full((batch_size, 2 if self.config.robot_state_feature else 1), False, device=batch["observation.state"].device)
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            cls_token_out = self.vae_encoder(vae_input.permute(1, 0, 2), pos_embed=pos_embed.permute(1, 0, 2), key_padding_mask=key_padding_mask)[0]
            latent_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu, log_sigma_x2 = latent_params[:, : self.config.latent_dim], latent_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu, log_sigma_x2 = None, None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(batch["observation.state"].device)

        encoder_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_pos = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature: encoder_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
        if self.config.env_state_feature: encoder_tokens.append(self.encoder_env_state_input_proj(batch["observation.environment_state"]))
        if self.config.image_features:
            for img in batch["observation.images"]:
                feat = self.backbone(img)["feature_map"]
                pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
                feat = self.encoder_img_feat_input_proj(feat)
                encoder_tokens.extend(list(einops.rearrange(feat, "b c h w -> (h w) b c")))
                encoder_pos.extend(list(einops.rearrange(pos, "b c h w -> (h w) b c")))
        encoder_tokens = torch.stack(encoder_tokens, axis=0)
        encoder_pos = torch.stack(encoder_pos, axis=0)
        encoder_out = self.encoder(encoder_tokens, pos_embed=encoder_pos)
        # ... [Input Processing Block End] ...

        decoder_queries = self.decoder_in_template.expand(-1, batch_size, -1).type_as(encoder_pos)
        decoder_pos = self.decoder_pos_embed.weight.unsqueeze(1)

        # =================================================================
        # 1. Parallel Independent Decoding
        # =================================================================
        # Arm Stream: Encoder -> Arm Features
        arm_features = self.arm_decoder(
            decoder_queries, 
            encoder_out, 
            encoder_pos_embed=encoder_pos, 
            decoder_pos_embed=decoder_pos
        ) # (S, B, D)

        # Waist Stream: Encoder -> Waist Features
        waist_features = self.waist_decoder(
            decoder_queries, 
            encoder_out, 
            encoder_pos_embed=encoder_pos, 
            decoder_pos_embed=decoder_pos
        ) # (S, B, D)

        # =================================================================
        # 2. Bidirectional Gated Interaction (The "Bridge")
        # =================================================================
        # Exchange information and refine features
        arm_refined, waist_refined = self.interaction_block(arm_features, waist_features)

        # =================================================================
        # 3. Final Prediction Heads
        # =================================================================
        # Transpose for Linear layers: (B, S, D)
        arm_out = arm_refined.transpose(0, 1)
        waist_out = waist_refined.transpose(0, 1)

        waist_pred = self.waist_head(waist_out)
        arm_pred = self.arm_head(arm_out)
        hand_pred = self.hand_head(arm_out)

        actions = torch.zeros(batch_size, self.config.chunk_size, self.config.action_feature.shape[0], device=arm_pred.device, dtype=arm_pred.dtype)
        actions[:, :, self.config.waist_indices] = waist_pred
        actions[:, :, self.config.arm_indices] = arm_pred
        actions[:, :, self.config.hand_indices] = hand_pred
        
        return actions, (mu, log_sigma_x2)


# (ACTEncoder, ACTDecoder 등 기존 클래스 정의는 그대로 유지)
class ACTEncoder(nn.Module):
    def __init__(self, config: ACTBideAdapConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()
    def forward(self, x, pos_embed=None, key_padding_mask=None):
        for layer in self.layers: x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        return self.norm(x)

class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTBideAdapConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
    def forward(self, x, pos_embed=None, key_padding_mask=None):
        skip = x
        if self.pre_norm: x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm: skip = x; x = self.norm2(x)
        else: x = self.norm1(x); skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm: x = self.norm2(x)
        return x

class ACTDecoder(nn.Module):
    def __init__(self, config: ACTBideAdapConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)
    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        for layer in self.layers: x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm: x = self.norm(x)
        return x

class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTBideAdapConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm
    def maybe_add_pos_embed(self, tensor, pos_embed): return tensor if pos_embed is None else tensor + pos_embed
    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        skip = x
        if self.pre_norm: x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm: skip = x; x = self.norm2(x)
        else: x = self.norm1(x); skip = x
        x = self.multihead_attn(query=self.maybe_add_pos_embed(x, decoder_pos_embed), key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed), value=encoder_out)[0]
        x = skip + self.dropout2(x)
        if self.pre_norm: skip = x; x = self.norm3(x)
        else: x = self.norm2(x); skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm: x = self.norm3(x)
        return x

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    def get_position_angle_vec(position): return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()

class ACTSinusoidalPositionEmbedding2d(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000
    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        inverse_frequency = self._temperature ** (2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension)
        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed

def get_activation_fn(activation: str) -> Callable:
    if activation == "relu": return F.relu
    if activation == "gelu": return F.gelu
    if activation == "glu": return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
