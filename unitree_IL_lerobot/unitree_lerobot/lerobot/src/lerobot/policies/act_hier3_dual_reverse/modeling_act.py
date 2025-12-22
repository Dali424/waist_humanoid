#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy (dual cross-attn, hand -> arm -> waist)."""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.policies.act_hier3_dual.configuration_act import ACTHier3DualConfig
from lerobot.policies.act_hier3_dual_reverse.configuration_act import ACTHier3DualReverseConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy


class ACTHier3DualReversePolicy(PreTrainedPolicy):
    config_class = ACTHier3DualReverseConfig
    name = "act_hier3_dual_reverse"

    def __init__(
        self,
        config: ACTHier3DualReverseConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
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

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss
        return loss, loss_dict


class ACTTemporalEnsembler:
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
            self.ensembled_actions = actions * self.ensemble_weights[None, :, None]
            self.ensembled_actions_count = self.ensemble_weights_cumsum.clone()
        else:
            if self.ensembled_actions.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"New actions must have same batch size ({actions.shape[0]}) as ensembled actions "
                    f"({self.ensembled_actions.shape[0]})."
                )
            self.ensembled_actions = torch.cat(
                [self.ensembled_actions, actions * self.ensemble_weights[None, :, None]], dim=1
            )
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, self.ensemble_weights_cumsum], dim=0
            )
        ensembled_actions = self.ensembled_actions / self.ensembled_actions_count[:, None]
        action, self.ensembled_actions, self.ensembled_actions_count = (
            ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Three-branch ACT with dual cross-attn decoding (hand -> arm -> waist)."""

    def __init__(self, config: ACTHier3DualConfig):
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(config)
        self.waist_decoder = ACTDualDecoder(config)
        self.arm_decoder = ACTDualDecoder(config)
        self.hand_decoder = ACTDualDecoder(config)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        n_1d_tokens = 1
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.register_buffer(
            "decoder_in_template", torch.zeros(config.chunk_size, 1, config.dim_model), persistent=False
        )
        self.register_buffer(
            "decoder_in_template", torch.zeros(config.chunk_size, 1, config.dim_model), persistent=False
        )

        self.waist_head = nn.Linear(config.dim_model, self.config.waist_dim)
        self.arm_head = nn.Linear(config.dim_model, self.config.arm_dim)
        self.hand_head = nn.Linear(config.dim_model, self.config.hand_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(
            self.encoder.parameters(),
            self.waist_decoder.parameters(),
            self.arm_decoder.parameters(),
            self.hand_decoder.parameters(),
        ):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for head in (self.waist_head, self.arm_head, self.hand_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.config.use_vae and self.training:
            assert "action" in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = (
            batch["observation.images"][0].shape[0]
            if "observation.images" in batch
            else batch["observation.environment_state"].shape[0]
        )

        if self.config.use_vae and "action" in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"]).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch["action"])
            vae_encoder_input = [cls_embed, action_embed] if not self.config.robot_state_feature else [
                cls_embed,
                robot_state_embed,
                action_embed,
            ]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch["observation.environment_state"]))

        if self.config.image_features:
            for img in batch["observation.images"]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_queries = self.decoder_in_template.expand(-1, batch_size, -1).type_as(encoder_in_pos_embed)
        decoder_pos = self.decoder_pos_embed.weight.unsqueeze(1)

        # Stage 1: hand decoder (encoder only).
        hand_dec_out = self.hand_decoder(
            decoder_queries,
            encoder_out,
            prev_out=None,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=decoder_pos,
            prev_pos_embed=None,
        )
        hand_tokens = hand_dec_out
        hand_dec_out = hand_dec_out.transpose(0, 1)  # (B,S,D)
        hand = self.hand_head(hand_dec_out)

        # Stage 2: arm decoder conditioned on hand tokens.
        hand_pos = decoder_pos.expand(-1, hand_tokens.shape[1], -1)
        arm_dec_out = self.arm_decoder(
            decoder_queries,
            encoder_out,
            prev_out=hand_tokens,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=decoder_pos,
            prev_pos_embed=hand_pos,
        )
        arm_tokens = arm_dec_out
        arm_dec_out = arm_dec_out.transpose(0, 1)  # (B,S,D)
        arm = self.arm_head(arm_dec_out)

        # Stage 3: waist decoder conditioned on hand + arm tokens.
        arm_pos = decoder_pos.expand(-1, arm_tokens.shape[1], -1)
        prev_out = torch.cat([hand_tokens, arm_tokens], dim=0)
        prev_pos = torch.cat([hand_pos, arm_pos], dim=0)
        waist_dec_out = self.waist_decoder(
            decoder_queries,
            encoder_out,
            prev_out=prev_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=decoder_pos,
            prev_pos_embed=prev_pos,
        )
        waist_dec_out = waist_dec_out.transpose(0, 1)  # (B,S,D)
        waist = self.waist_head(waist_dec_out)

        action_dim = self.config.action_feature.shape[0]
        actions = torch.zeros(
            hand_dec_out.shape[0],
            hand_dec_out.shape[1],
            action_dim,
            device=hand_dec_out.device,
            dtype=hand_dec_out.dtype,
        )
        actions[:, :, self.config.waist_indices] = waist
        actions[:, :, self.config.arm_indices] = arm
        actions[:, :, self.config.hand_indices] = hand
        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    def __init__(self, config: ACTHier3DualConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTHier3DualConfig):
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

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDualDecoder(nn.Module):
    def __init__(self, config: ACTHier3DualConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDualDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        prev_out: Tensor | None = None,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
        prev_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out=encoder_out,
                prev_out=prev_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
                prev_pos_embed=prev_pos_embed,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDualDecoderLayer(nn.Module):
    def __init__(self, config: ACTHier3DualConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.cross_attn_encoder = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.cross_attn_prev = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear_mid1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear_mid2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.linear_final1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear_final2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.dropout_self = nn.Dropout(config.dropout)
        self.dropout_cross1 = nn.Dropout(config.dropout)
        self.dropout_mid = nn.Dropout(config.dropout)
        self.dropout_cross2 = nn.Dropout(config.dropout)
        self.dropout_final = nn.Dropout(config.dropout)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm_mid = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.norm4 = nn.LayerNorm(config.dim_model)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        prev_out: Tensor | None = None,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
        prev_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout_self(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.cross_attn_encoder(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout_cross1(x)

        if self.pre_norm:
            skip = x
            x = self.norm_mid(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear_mid2(self.dropout_mid(self.activation(self.linear_mid1(x))))
        x = skip + self.dropout_mid(x)

        if prev_out is not None:
            if self.pre_norm:
                skip = x
                x = self.norm3(x)
            else:
                x = self.norm_mid(x)
                skip = x
            x = self.cross_attn_prev(
                query=self.maybe_add_pos_embed(x, decoder_pos_embed),
                key=self.maybe_add_pos_embed(prev_out, prev_pos_embed),
                value=prev_out,
            )[0]
            x = skip + self.dropout_cross2(x)

        if self.pre_norm:
            skip = x
            x = self.norm4(x)
        else:
            x = self.norm3(x)
            skip = x
        x = self.linear_final2(self.dropout_final(self.activation(self.linear_final1(x))))
        x = skip + self.dropout_final(x)
        if not self.pre_norm:
            x = self.norm4(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

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

        dim_t = torch.arange(self.dimension, device=x.device, dtype=torch.float32)
        dim_t = self._temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.dimension)

        pos_x = x_range[:, :, :, None] / dim_t
        pos_y = y_range[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
