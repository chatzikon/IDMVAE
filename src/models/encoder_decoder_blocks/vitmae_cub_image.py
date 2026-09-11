import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoImageProcessor,
    ViTMAEForPreTraining,
)


class ViTMAEDecoderImg(nn.Module):
    """
    Pretrained ViT-MAE image decoder adapted to IDMVAE latents.

    Input:
        u: [K, B, W+Z] or [B, W+Z]

    Output:
        reconstructed RGB image:
            [K, B, 3, 224, 224]
        or
            [B, 3, 224, 224]

        plus the likelihood scale, matching DecoderImg's interface.
    """

    def __init__(
        self,
        ndim_w,
        ndim_z,
        model_name="facebook/vit-mae-base",
        cond_tokens_per_latent=4,
        adapter_heads=12,
        adapter_mlp_ratio=4.0,
    ):
        super().__init__()

        self.ndim_w = ndim_w
        self.ndim_z = ndim_z

        # =========================================================
        # Load complete pretrained MAE temporarily.
        # We keep ONLY its decoder afterwards.
        # =========================================================

        pretrained = ViTMAEForPreTraining.from_pretrained(
            model_name
        )

        processor = AutoImageProcessor.from_pretrained(
            model_name
        )

        config = pretrained.config

        # ---------------------------------------------------------
        # MAE configuration
        # ---------------------------------------------------------

        image_size = config.image_size
        patch_size = config.patch_size

        if isinstance(image_size, (tuple, list)):
            if image_size[0] != image_size[1]:
                raise ValueError(
                    "Only square ViT-MAE images are supported."
                )
            image_size = image_size[0]

        if isinstance(patch_size, (tuple, list)):
            if patch_size[0] != patch_size[1]:
                raise ValueError(
                    "Only square ViT-MAE patches are supported."
                )
            patch_size = patch_size[0]

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = config.num_channels

        # MAE encoder hidden dimension.
        # For vit-mae-base: 768
        self.hidden_size = config.hidden_size

        self.num_patches = (
            self.image_size // self.patch_size
        ) ** 2

        self.cond_tokens_per_latent = (
            cond_tokens_per_latent
        )

        # =========================================================
        # PRETRAINED decoder
        # =========================================================

        self.decoder = pretrained.decoder

        # =========================================================
        # Spatial query initialization
        #
        # Real MAE encoder tokens already contain the MAE encoder
        # positional embeddings. We use those pretrained spatial
        # embeddings as initialization for our IDMVAE-generated
        # pseudo encoder tokens.
        # =========================================================

        encoder_pos = (
            pretrained
            .vit
            .embeddings
            .position_embeddings
            .detach()
            .clone()
        )

        if encoder_pos.size(1) != self.num_patches + 1:
            raise ValueError(
                "Unexpected ViT-MAE positional embedding size: "
                f"{tuple(encoder_pos.shape)}"
            )

        # 196 spatial queries for MAE-base.
        self.patch_queries = nn.Parameter(
            encoder_pos[:, 1:, :].clone()
        )

        # Keep CLS positional embedding fixed.
        self.register_buffer(
            "cls_pos",
            encoder_pos[:, :1, :].clone(),
            persistent=False,
        )

        # =========================================================
        # Image normalization used by pretrained MAE
        # =========================================================

        image_mean = torch.tensor(
            processor.image_mean,
            dtype=torch.float32,
        )

        image_std = torch.tensor(
            processor.image_std,
            dtype=torch.float32,
        )

        self.register_buffer(
            "image_mean",
            image_mean,
            persistent=False,
        )

        self.register_buffer(
            "image_std",
            image_std,
            persistent=False,
        )

        # We do NOT need the pretrained MAE encoder.
        # Only self.decoder survives.
        del pretrained

        # =========================================================
        # IDMVAE -> MAE adapter
        #
        # Keep private w and shared z separated initially.
        # =========================================================

        t = cond_tokens_per_latent

        self.w_to_tokens = nn.Linear(
            ndim_w,
            t * self.hidden_size,
        )

        self.z_to_tokens = nn.Linear(
            ndim_z,
            t * self.hidden_size,
        )

        # ---------------------------------------------------------
        # Adapter: spatial queries attend to w/z conditioning
        # ---------------------------------------------------------

        if self.hidden_size % adapter_heads != 0:
            raise ValueError(
                f"MAE hidden size {self.hidden_size} must be "
                f"divisible by adapter_heads={adapter_heads}"
            )

        self.query_norm = nn.LayerNorm(
            self.hidden_size
        )

        self.memory_norm = nn.LayerNorm(
            self.hidden_size
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=adapter_heads,
            batch_first=True,
        )

        ff_dim = int(
            self.hidden_size * adapter_mlp_ratio
        )

        self.ff_norm = nn.LayerNorm(
            self.hidden_size
        )

        self.ff = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                ff_dim,
            ),
            nn.GELU(),
            nn.Linear(
                ff_dim,
                self.hidden_size,
            ),
        )

        # CLS token content produced from complete IDMVAE latent.
        self.cls_proj = nn.Linear(
            ndim_w + ndim_z,
            self.hidden_size,
        )

    # =============================================================
    # Helpers
    # =============================================================

    def _normalization_stats(self, x):
        """
        Produce broadcastable MAE mean/std tensors.

        x can be:
            [B,C,H,W]
        or
            [K,B,C,H,W]
        """

        prefix = [1] * (x.ndim - 3)

        shape = (
            *prefix,
            self.num_channels,
            1,
            1,
        )

        mean = self.image_mean.to(
            device=x.device,
            dtype=x.dtype,
        ).view(*shape)

        std = self.image_std.to(
            device=x.device,
            dtype=x.dtype,
        ).view(*shape)

        return mean, std

    def prepare_target(self, rgb):
        """
        RGB IDMVAE reconstruction target.

        Input:
            [B,3,H,W]

        Output:
            [B,3,224,224] in [0,1]
        """

        if rgb.ndim != 4 or rgb.size(1) != 3:
            raise ValueError(
                "ViT-MAE RGB target must have shape "
                f"[B,3,H,W], got {tuple(rgb.shape)}"
            )

        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        else:
            rgb = rgb.float()

        if rgb.shape[-2:] != (
            self.image_size,
            self.image_size,
        ):
            rgb = F.interpolate(
                rgb,
                size=(
                    self.image_size,
                    self.image_size,
                ),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        return rgb.clamp(0.0, 1.0)

    def to_rgb(self, x):
        """
        Useful for visualization.
        Decoder forward() already returns RGB space.
        """
        return x.clamp(0.0, 1.0)

    def _unpatchify(self, patches):
        """
        [B, N, P*P*C]
              ->
        [B, C, H, W]
        """

        b, n, d = patches.shape

        p = self.patch_size
        c = self.num_channels
        g = self.image_size // p

        expected_dim = p * p * c

        if n != g * g:
            raise ValueError(
                f"Expected {g*g} patches, got {n}"
            )

        if d != expected_dim:
            raise ValueError(
                f"Expected patch dim {expected_dim}, got {d}"
            )

        patches = patches.view(
            b,
            g,
            g,
            p,
            p,
            c,
        )

        patches = patches.permute(
            0,
            5,
            1,
            3,
            2,
            4,
        ).contiguous()

        return patches.view(
            b,
            c,
            self.image_size,
            self.image_size,
        )

    # =============================================================
    # Forward
    # =============================================================

    def forward(self, u):

        expected_dim = (
            self.ndim_w + self.ndim_z
        )

        if u.size(-1) != expected_dim:
            raise ValueError(
                f"Expected latent dim {expected_dim}, "
                f"got {u.size(-1)}"
            )

        # Usually [K,B,D].
        leading_shape = u.shape[:-1]

        # [K,B,D] -> [K*B,D]
        u_flat = u.reshape(
            -1,
            expected_dim,
        )

        batch_flat = u_flat.size(0)

        # =========================================================
        # Preserve IDMVAE w/z distinction
        # =========================================================

        w, z = torch.split(
            u_flat,
            [
                self.ndim_w,
                self.ndim_z,
            ],
            dim=-1,
        )

        t = self.cond_tokens_per_latent

        w_tokens = self.w_to_tokens(w).view(
            batch_flat,
            t,
            self.hidden_size,
        )

        z_tokens = self.z_to_tokens(z).view(
            batch_flat,
            t,
            self.hidden_size,
        )

        # [KB, 2*t, 768]
        memory = torch.cat(
            [
                w_tokens,
                z_tokens,
            ],
            dim=1,
        )

        # =========================================================
        # Generate MAE-compatible spatial tokens
        # =========================================================

        queries = self.patch_queries.to(
            dtype=memory.dtype
        ).expand(
            batch_flat,
            -1,
            -1,
        )

        q = self.query_norm(
            queries
        )

        mem = self.memory_norm(
            memory
        )

        attn_out, _ = self.cross_attn(
            query=q,
            key=mem,
            value=mem,
            need_weights=False,
        )

        patch_tokens = (
            queries + attn_out
        )

        patch_tokens = (
            patch_tokens
            + self.ff(
                self.ff_norm(
                    patch_tokens
                )
            )
        )

        # =========================================================
        # CLS token
        # =========================================================

        cls_token = self.cls_proj(
            u_flat
        ).unsqueeze(1)

        cls_token = (
            cls_token
            + self.cls_pos.to(
                dtype=cls_token.dtype
            )
        )

        # [KB, 197, 768]
        hidden_states = torch.cat(
            [
                cls_token,
                patch_tokens,
            ],
            dim=1,
        )

        # We provide all 196 positions, so identity restoration.
        ids_restore = torch.arange(
            self.num_patches,
            device=u.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(
            batch_flat,
            -1,
        )

        # =========================================================
        # PRETRAINED ViT-MAE decoder
        # =========================================================

        decoder_output = self.decoder(
            hidden_states,
            ids_restore=ids_restore,
            interpolate_pos_encoding=False,
        )

        # [KB,196,16*16*3]
        normalized_patches = (
            decoder_output.logits
        )

        normalized_image = self._unpatchify(
            normalized_patches
        )

        # =========================================================
        # Convert MAE processor space -> RGB space
        #
        # Do NOT clamp here: reconstruction loss should still
        # provide gradients if predictions are outside [0,1].
        # =========================================================

        mean, std = self._normalization_stats(
            normalized_image
        )

        rgb = (
            normalized_image * std
            + mean
        )

        # Restore [K,B,...] or [B,...].
        rgb = rgb.view(
            *leading_shape,
            self.num_channels,
            self.image_size,
            self.image_size,
        )

        # Same API as DecoderImg.
        scale = rgb.new_tensor(0.01)

        return rgb, scale