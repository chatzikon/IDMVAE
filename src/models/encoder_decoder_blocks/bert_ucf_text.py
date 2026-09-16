import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    BertModel,
    BertForMaskedLM,
)

from utils import Constants


# ============================================================
# BERT token likelihood
# ============================================================

class BertSequenceLikelihood:
    """
    Distribution-like wrapper used by the IDMVAE ELBO.

    logits:
        [K, B, L, V]

    target_ids:
        [B, L] or [K, B, L]

    log_prob():
        [K, B, L]

    Padding positions do not contribute to likelihood.
    """

    def __init__(
        self,
        logits,
        pad_token_id,
    ):
        self.logits = logits
        self.pad_token_id = pad_token_id

    @property
    def probs(self):
        return torch.softmax(
            self.logits,
            dim=-1,
        )

    @property
    def mean(self):
        return self.probs

    @property
    def mode(self):
        return torch.argmax(
            self.logits,
            dim=-1,
        )

    def log_prob(
        self,
        target_ids,
    ):
        K, B, L, _ = self.logits.shape

        if target_ids.ndim == 2:
            target_ids = (
                target_ids
                .unsqueeze(0)
                .expand(K, -1, -1)
            )

        elif target_ids.ndim != 3:
            raise ValueError(
                "BERT target must have shape "
                f"[B,L] or [K,B,L], got "
                f"{tuple(target_ids.shape)}"
            )

        if target_ids.size(-1) != L:
            raise ValueError(
                "BERT target/logit sequence-length mismatch: "
                f"target={target_ids.size(-1)}, logits={L}"
            )

        target_ids = target_ids.to(
            self.logits.device
        ).long()

        valid = (
            target_ids
            != self.pad_token_id
        )

        log_probs = F.log_softmax(
            self.logits,
            dim=-1,
        )

        token_log_prob = torch.gather(
            log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        token_log_prob = (
            token_log_prob
            * valid.to(token_log_prob.dtype)
        )

        return token_log_prob


# ============================================================
# BERT text encoder
# ============================================================

class BertTextEncoder(nn.Module):
    """
    Caption -> q(w_txt, z_txt)

    Important:
    the second returned tensor is already a positive SCALE,
    matching cnn_ucf_text.Enc and the expectations of base_vae.py.
    """

    def __init__(
        self,
        latent_dim_w,
        latent_dim_z,
        dist_name,
        model_name="bert-base-uncased",
    ):
        super().__init__()

        self.latent_dim_w = latent_dim_w
        self.latent_dim_z = latent_dim_z
        self.dist_name = dist_name

        self.backbone = BertModel.from_pretrained(
            model_name
        )

        self.hidden_size = (
            self.backbone.config.hidden_size
        )

        self.pad_token_id = (
            self.backbone.config.pad_token_id
        )

        # Private latent w
        self.mu_w = nn.Linear(
            self.hidden_size,
            latent_dim_w,
        )

        self.scale_w = nn.Linear(
            self.hidden_size,
            latent_dim_w,
        )

        # Shared latent z
        self.mu_z = nn.Linear(
            self.hidden_size,
            latent_dim_z,
        )

        self.scale_z = nn.Linear(
            self.hidden_size,
            latent_dim_z,
        )

    def _pool(
        self,
        hidden,
        attention_mask,
    ):
        """
        Masked mean pooling.

        hidden:
            [B,L,H]

        attention_mask:
            [B,L]
        """

        mask = (
            attention_mask
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        pooled = (
            hidden * mask
        ).sum(dim=1)

        denom = (
            mask
            .sum(dim=1)
            .clamp_min(1.0)
        )

        return pooled / denom

    def _convert_scale(
        self,
        raw_scale,
    ):
        """
        Mirror cnn_ucf_text.Enc exactly.

        Normal:
            softplus(raw) + eta

        Laplace/current non-Normal branch:
            softmax(raw) * latent_dim + eta
        """

        if self.dist_name == "Normal":

            return (
                F.softplus(raw_scale)
                + Constants.eta
            )

        # Same behaviour as the current CNN text encoder's
        # non-Normal branch.
        return (
            F.softmax(
                raw_scale,
                dim=-1,
            )
            * raw_scale.size(-1)
            + Constants.eta
        )

    def _posterior_params(
        self,
        hidden,
        attention_mask,
    ):
        pooled = self._pool(
            hidden,
            attention_mask,
        )

        mu_w = self.mu_w(pooled)
        raw_scale_w = self.scale_w(pooled)

        mu_z = self.mu_z(pooled)
        raw_scale_z = self.scale_z(pooled)

        scale_w = self._convert_scale(
            raw_scale_w
        )

        scale_z = self._convert_scale(
            raw_scale_z
        )

        mu = torch.cat(
            (
                mu_w,
                mu_z,
            ),
            dim=-1,
        )

        scale = torch.cat(
            (
                scale_w,
                scale_z,
            ),
            dim=-1,
        )

        return mu, scale

    def forward(
        self,
        input_ids,
    ):
        """
        Normal text-encoder path.

        input_ids:
            [B,L]
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "BertTextEncoder expects [B,L] token IDs, "
                f"got {tuple(input_ids.shape)}"
            )

        input_ids = input_ids.long()

        attention_mask = (
            input_ids
            != self.pad_token_id
        ).long()

        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return self._posterior_params(
            output.last_hidden_state,
            attention_mask,
        )

    def forward_from_embeddings(
        self,
        inputs_embeds,
        attention_mask=None,
    ):
        """
        Differentiable GenAug re-encoding path.

        inputs_embeds:
            [B,L,H]
        """

        if inputs_embeds.ndim != 3:
            raise ValueError(
                "BERT embedded text must have shape "
                f"[B,L,H], got {tuple(inputs_embeds.shape)}"
            )

        B, L, H = inputs_embeds.shape

        if H != self.hidden_size:
            raise ValueError(
                f"Expected BERT embedding dim {self.hidden_size}, "
                f"got {H}"
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                B,
                L,
                dtype=torch.long,
                device=inputs_embeds.device,
            )

        output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return self._posterior_params(
            output.last_hidden_state,
            attention_mask,
        )


# ============================================================
# Non-autoregressive latent-conditioned BERT decoder
# ============================================================

class BertLatentDecoder(nn.Module):
    """
    IDMVAE latent -> caption using pretrained BERT MLM.

    There is NO:
        - teacher forcing
        - autoregression
        - causal mask
        - ground-truth input to the decoder

    Decoder input is:

        [continuous w prefix]
        [continuous z prefix]
        [MASK] [MASK] ... [MASK]

    BERT predicts all caption positions in parallel.
    """

    def __init__(
        self,
        latent_dim_w,
        latent_dim_z,
        model_name="bert-base-uncased",
        memory_tokens_per_latent=4,
        max_length=64,
    ):
        super().__init__()

        self.latent_dim_w = latent_dim_w
        self.latent_dim_z = latent_dim_z
        self.max_length = max_length

        self.memory_tokens_per_latent = (
            memory_tokens_per_latent
        )

        self.mlm = (
            BertForMaskedLM
            .from_pretrained(model_name)
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(model_name)
        )

        self.hidden_size = (
            self.mlm.config.hidden_size
        )

        self.vocab_size = (
            self.mlm.config.vocab_size
        )

        self.pad_token_id = (
            self.tokenizer.pad_token_id
        )

        self.mask_token_id = (
            self.tokenizer.mask_token_id
        )

        self.sep_token_id = (
            self.tokenizer.sep_token_id
        )

        self.cls_token_id = (
            self.tokenizer.cls_token_id
        )

        if self.mask_token_id is None:
            raise RuntimeError(
                f"{model_name} does not define a mask token."
            )

        t = self.memory_tokens_per_latent
        d = self.hidden_size

        # ----------------------------------------------------
        # Latent -> continuous BERT prefix
        # ----------------------------------------------------

        self.w_to_memory = nn.Sequential(
            nn.Linear(
                latent_dim_w,
                t * d,
            ),
            nn.GELU(),
        )

        self.z_to_memory = nn.Sequential(
            nn.Linear(
                latent_dim_z,
                t * d,
            ),
            nn.GELU(),
        )

        self.w_type = nn.Parameter(
            torch.zeros(
                1,
                1,
                d,
            )
        )

        self.z_type = nn.Parameter(
            torch.zeros(
                1,
                1,
                d,
            )
        )

        self.memory_norm = nn.LayerNorm(d)

        nn.init.normal_(
            self.w_type,
            std=0.02,
        )

        nn.init.normal_(
            self.z_type,
            std=0.02,
        )

        # Continuous output used only by GenAug.
        self.gen_aug_projection = nn.Linear(
            d,
            d,
        )

    def _build_memory(
        self,
        u,
    ):
        """
        u:
            [K,B,W+Z]

        returns:
            memory [K*B,2*T,H]
            K
            B
        """

        if u.ndim != 3:
            raise ValueError(
                "BertLatentDecoder expects [K,B,D], "
                f"got {tuple(u.shape)}"
            )

        K, B, _ = u.shape

        w, z = torch.split(
            u,
            [
                self.latent_dim_w,
                self.latent_dim_z,
            ],
            dim=-1,
        )

        w = w.reshape(
            K * B,
            self.latent_dim_w,
        )

        z = z.reshape(
            K * B,
            self.latent_dim_z,
        )

        t = self.memory_tokens_per_latent
        d = self.hidden_size

        w_memory = (
            self.w_to_memory(w)
            .reshape(K * B, t, d)
        )

        z_memory = (
            self.z_to_memory(z)
            .reshape(K * B, t, d)
        )

        w_memory = (
            w_memory
            + self.w_type
        )

        z_memory = (
            z_memory
            + self.z_type
        )

        memory = torch.cat(
            (
                w_memory,
                z_memory,
            ),
            dim=1,
        )

        memory = self.memory_norm(
            memory
        )

        return memory, K, B

    def _latent_to_hidden(
        self,
        u,
    ):
        """
        Produce BERT hidden states for the text slots.

        No target text is used.
        """

        memory, K, B = (
            self._build_memory(u)
        )

        KB = K * B
        L = self.max_length

        # Every caption position is unknown.
        mask_ids = torch.full(
            (
                KB,
                L,
            ),
            self.mask_token_id,
            dtype=torch.long,
            device=memory.device,
        )

        # Only obtain the token embeddings here.
        # BertModel will still add its pretrained position
        # and token-type embeddings when inputs_embeds is used.
        mask_embeddings = (
            self.mlm
            .bert
            .embeddings
            .word_embeddings(mask_ids)
        )

        inputs_embeds = torch.cat(
            (
                memory,
                mask_embeddings,
            ),
            dim=1,
        )

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=memory.device,
        )

        bert_output = self.mlm.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        t = self.memory_tokens_per_latent

        # Discard hidden states belonging to latent-prefix slots.
        text_hidden = (
            bert_output
            .last_hidden_state[
                :,
                2 * t:,
                :
            ]
        )

        # [K*B,L,H]
        return text_hidden, K, B

    def forward(
        self,
        u,
    ):
        """
        Latent-only, non-autoregressive decoding.

        u:
            [K,B,W+Z]
        """

        text_hidden, K, B = (
            self._latent_to_hidden(u)
        )

        logits = self.mlm.cls(
            text_hidden
        )

        logits = logits.reshape(
            K,
            B,
            self.max_length,
            self.vocab_size,
        )

        return BertSequenceLikelihood(
            logits=logits,
            pad_token_id=self.pad_token_id,
        )

    def forward_gen_aug(
        self,
        u,
    ):
        """
        Fully differentiable latent -> continuous-text path.

        returns:
            [K,B,L,H]

        No argmax and no discrete generation.
        """

        text_hidden, K, B = (
            self._latent_to_hidden(u)
        )

        text_hidden = (
            self.gen_aug_projection(
                text_hidden
            )
        )

        return text_hidden.reshape(
            K,
            B,
            self.max_length,
            self.hidden_size,
        )

    def decode_batch(
        self,
        token_ids,
    ):
        """
        Convert BERT predictions to strings.

        The first predicted [SEP] is treated as end-of-caption.
        """

        token_ids = (
            token_ids
            .detach()
            .clone()
            .cpu()
            .long()
        )

        if (
            token_ids.ndim == 3
            and token_ids.size(0) == 1
        ):
            token_ids = token_ids.squeeze(0)

        if token_ids.ndim != 2:
            raise ValueError(
                "decode_batch expects [B,L], "
                f"got {tuple(token_ids.shape)}"
            )

        for row in token_ids:

            sep_positions = (
                row == self.sep_token_id
            ).nonzero(
                as_tuple=False
            )

            if sep_positions.numel() > 0:

                first_sep = int(
                    sep_positions[0].item()
                )

                if first_sep + 1 < row.numel():
                    row[
                        first_sep + 1:
                    ] = self.pad_token_id

        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )