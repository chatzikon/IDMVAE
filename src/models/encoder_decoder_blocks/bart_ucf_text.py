import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
)

from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
)


class BartSequenceLikelihood:
    """
    Lightweight distribution-like object.

    It behaves like the old OneHotCategorical where the
    objective needs:

        px.log_prob(target)

    logits:
        [K, B, L, vocab_size]

    target_ids:
        [B, L]
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
        # Keeps compatibility with code that accesses
        # distribution.mean.
        return self.probs

    @property
    def mode(self):
        return torch.argmax(
            self.logits,
            dim=-1,
        )

    def log_prob(self, target_ids):

        # logits:
        # [K,B,L,V]

        K, B, L, _ = self.logits.shape

        if target_ids.ndim == 2:

            # [B,L]
            target_ids = (
                target_ids
                .unsqueeze(0)
                .expand(K, -1, -1)
            )

        elif target_ids.ndim != 3:

            raise ValueError(
                "BART target must have shape "
                f"[B,L] or [K,B,L], got "
                f"{target_ids.shape}"
            )

        target_ids = target_ids.to(
            self.logits.device
        ).long()

        # Standard language-model behaviour:
        # padding does not contribute to likelihood.
        valid = (
            target_ids != self.pad_token_id
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

        # [K,B,L]
        #
        # Deliberately do NOT sum over L here.
        # OneHotCategorical.log_prob() in your old
        # decoder also leaves the sentence dimension.
        return token_log_prob


class BartLatentDecoder(nn.Module):

    def __init__(
        self,
        latent_dim_w,
        latent_dim_z,
        model_name="facebook/bart-base",
        memory_tokens_per_latent=4,
        gen_aug_length=32,
        gen_aug_embedding_dim=128,
        token_dropout=0.0,
    ):
        super().__init__()

        self.latent_dim_w = latent_dim_w
        self.latent_dim_z = latent_dim_z



        if not 0.0 <= token_dropout < 1.0:
            raise ValueError(
                f"token_dropout must be in [0,1), got {token_dropout}"
            )

        self.token_dropout = token_dropout

        self.memory_tokens_per_latent = (
            memory_tokens_per_latent
        )

        # -----------------------------------------------------
        # Load pretrained BART temporarily.
        # -----------------------------------------------------

        pretrained = (
            BartForConditionalGeneration
            .from_pretrained(model_name)
        )

        self.config = pretrained.config

        self.hidden_size = (
            pretrained.config.d_model
        )

        self.vocab_size = (
            pretrained.config.vocab_size
        )

        self.pad_token_id = (
            pretrained.config.pad_token_id
        )

        self.eos_token_id = (
            pretrained.config.eos_token_id
        )

        self.decoder_start_token_id = (
            pretrained.config.decoder_start_token_id
        )

        # -----------------------------------------------------
        # KEEP ONLY pretrained decoder + pretrained LM head.
        #
        # BART encoder is NOT retained.
        # -----------------------------------------------------

        self.decoder = pretrained.model.decoder

        self.lm_head = pretrained.lm_head

        self.register_buffer(
            "final_logits_bias",
            pretrained.final_logits_bias
            .detach()
            .clone(),
        )

        # Explicitly preserve tied token/LM-head weights.
        if pretrained.config.tie_word_embeddings:

            self.lm_head.weight = (
                self.decoder.embed_tokens.weight
            )

        # We can now release the rest of BART,
        # including its encoder.
        del pretrained

        # Tokenizer is not an nn.Module and therefore does not
        # become part of the checkpoint parameters.
        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(model_name)
        )

        t = self.memory_tokens_per_latent
        d = self.hidden_size

        # -----------------------------------------------------
        # IDMVAE latent -> BART cross-attention memory
        # -----------------------------------------------------

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

        # Learned locations within the latent memory.
        self.memory_pos = nn.Parameter(
            torch.zeros(
                1,
                2 * t,
                d,
            )
        )

        # Tell BART which memory tokens came from
        # private w versus shared z.
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
            self.memory_pos,
            std=0.02,
        )

        nn.init.normal_(
            self.w_type,
            std=0.02,
        )

        nn.init.normal_(
            self.z_type,
            std=0.02,
        )

        # -----------------------------------------------------
        # Differentiable GenAug path
        # -----------------------------------------------------
        #
        # This path is used only by the generative-augmentation
        # loss. It does NOT generate discrete BART tokens.
        #
        # Instead, learned continuous queries attend to the
        # latent-derived BART memory and produce a continuous
        # sequence that can be re-encoded by the existing
        # CNN text encoder.
        # -----------------------------------------------------

        self.gen_aug_length = gen_aug_length
        self.gen_aug_embedding_dim = gen_aug_embedding_dim

        # Learned decoder queries in BART hidden space.
        #
        # For bart-base:
        #     [1, 32, 768]
        self.gen_aug_queries = nn.Parameter(
            torch.empty(
                1,
                self.gen_aug_length,
                self.hidden_size,
            )
        )

        nn.init.normal_(
            self.gen_aug_queries,
            mean=0.0,
            std=0.02,
        )

        # Project BART hidden states into the embedding space
        # used internally by the old CNN text encoder.
        #
        # bart-base:
        #     768 -> 128
        self.gen_aug_projection = nn.Linear(
            self.hidden_size,
            self.gen_aug_embedding_dim,
        )

    def _build_memory(self, u):
        """
        u:
            [K,B,D]

        returns:
            memory [K*B, 2*T, bart_hidden]
            K
            B
        """

        if u.ndim != 3:
            raise ValueError(
                "BartLatentDecoder expects "
                f"[K,B,D], got {u.shape}"
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
            .view(K * B, t, d)
        )

        z_memory = (
            self.z_to_memory(z)
            .view(K * B, t, d)
        )

        w_memory = (
            w_memory + self.w_type
        )

        z_memory = (
            z_memory + self.z_type
        )

        memory = torch.cat(
            [
                w_memory,
                z_memory,
            ],
            dim=1,
        )

        memory = (
            memory
            + self.memory_pos
        )

        memory = self.memory_norm(
            memory
        )

        return memory, K, B

    def forward(
        self,
        u,
        target_ids,
    ):
        """
        Teacher-forced BART decoding.

        u:
            [K,B,D]

        target_ids:
            [B,L]

        returns:
            BartSequenceLikelihood
        """

        memory, K, B = (
            self._build_memory(u)
        )

        target_ids = target_ids.to(
            memory.device
        ).long()

        if target_ids.ndim != 2:
            raise ValueError(
                "target_ids must be [B,L], "
                f"got {target_ids.shape}"
            )

        L = target_ids.size(1)

        # Same target for every latent Monte-Carlo sample.
        labels = (
            target_ids
            .unsqueeze(0)
            .expand(K, -1, -1)
            .reshape(K * B, L)
        )

        # Standard BART teacher forcing:
        #
        # target:
        #       token_1 token_2 ... EOS
        #
        # decoder input:
        # START token_1 ... token_(n-1)
        decoder_input_ids = shift_tokens_right(
            labels,
            self.pad_token_id,
            self.decoder_start_token_id,
        )





        if self.training and self.token_dropout > 0.0:
            drop_mask = (
                    torch.rand_like(
                        decoder_input_ids,
                        dtype=torch.float,
                    )
                    < self.token_dropout
            )

            # Never corrupt padding or the decoder start token.
            drop_mask &= (
                    decoder_input_ids != self.pad_token_id
            )

            drop_mask &= (
                    decoder_input_ids
                    != self.decoder_start_token_id
            )





            decoder_input_ids = (
                decoder_input_ids.masked_fill(
                    drop_mask,
                    self.tokenizer.mask_token_id,
                )
            )

        decoder_attention_mask = (
            decoder_input_ids
            != self.pad_token_id
        ).long()

        memory_attention_mask = torch.ones(
            memory.shape[:2],
            dtype=torch.long,
            device=memory.device,
        )

        output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        logits = (
            self.lm_head(
                output.last_hidden_state
            )
            + self.final_logits_bias
        )

        logits = logits.view(
            K,
            B,
            L,
            self.vocab_size,
        )

        return BartSequenceLikelihood(
            logits=logits,
            pad_token_id=self.pad_token_id,
        )

    def forward_gen_aug(self, u):
        """
        Differentiable BART path used only by GenAug.

        Unlike normal BART generation, this method:

            - does NOT use target_ids
            - does NOT use teacher forcing
            - does NOT call argmax
            - does NOT generate discrete token IDs

        Learned continuous queries attend to the same
        latent-derived memory used by normal BART.

        Parameters
        ----------
        u:
            IDMVAE latent tensor with shape:

                [K, B, latent_dim_w + latent_dim_z]

        Returns
        -------
        text_features:
            Continuous text representation:

                [K, B, gen_aug_length, gen_aug_embedding_dim]

            With the current defaults:

                [K, B, 32, 128]
        """

        # -------------------------------------------------
        # 1. Build exactly the same latent memory used by
        #    teacher-forced BART and normal generation.
        # -------------------------------------------------

        memory, K, B = self._build_memory(u)

        # memory:
        # [K*B, 2*T, hidden_size]

        # -------------------------------------------------
        # 2. Create one learned query sequence for every
        #    latent sample.
        # -------------------------------------------------

        queries = self.gen_aug_queries.expand(
            K * B,
            -1,
            -1,
        )

        # queries:
        # [K*B, gen_aug_length, hidden_size]

        # -------------------------------------------------
        # 3. Attention masks
        # -------------------------------------------------

        query_attention_mask = torch.ones(
            (
                K * B,
                self.gen_aug_length,
            ),
            dtype=torch.long,
            device=memory.device,
        )

        memory_attention_mask = torch.ones(
            memory.shape[:2],
            dtype=torch.long,
            device=memory.device,
        )

        # -------------------------------------------------
        # 4. Run BART decoder using continuous queries.
        #
        #    Crucially:
        #       inputs_embeds=queries
        #
        #    instead of:
        #       input_ids=...
        #
        #    Therefore there is no discrete-token operation.
        # -------------------------------------------------

        output = self.decoder(
            inputs_embeds=queries,
            attention_mask=query_attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # [K*B, gen_aug_length, hidden_size]
        hidden = output.last_hidden_state

        # -------------------------------------------------
        # 5. Convert BART hidden states to the continuous
        #    embedding representation expected by the
        #    existing CNN text encoder.
        # -------------------------------------------------

        text_features = self.gen_aug_projection(
            hidden
        )

        # [K*B, gen_aug_length, gen_aug_embedding_dim]

        # -------------------------------------------------
        # 6. Restore Monte-Carlo / batch dimensions.
        # -------------------------------------------------

        text_features = text_features.reshape(
            K,
            B,
            self.gen_aug_length,
            self.gen_aug_embedding_dim,
        )

        return text_features

    @torch.no_grad()
    def generate(
        self,
        u,
        max_new_tokens=64,
    ):
        """
        Initial greedy generation implementation.

        Later we can add beam search if desired.
        """

        memory, K, B = (
            self._build_memory(u)
        )

        memory_attention_mask = torch.ones(
            memory.shape[:2],
            dtype=torch.long,
            device=memory.device,
        )

        generated = torch.full(
            (
                K * B,
                1,
            ),
            self.decoder_start_token_id,
            dtype=torch.long,
            device=memory.device,
        )

        finished = torch.zeros(
            K * B,
            dtype=torch.bool,
            device=memory.device,
        )

        for _ in range(max_new_tokens):

            output = self.decoder(
                input_ids=generated,
                encoder_hidden_states=memory,
                encoder_attention_mask=(
                    memory_attention_mask
                ),
                use_cache=False,
                return_dict=True,
            )

            next_logits = (
                self.lm_head(
                    output.last_hidden_state[:, -1]
                )
                + self.final_logits_bias
            )

            next_token = torch.argmax(
                next_logits,
                dim=-1,
            )

            # Already-finished sequences produce padding.
            next_token = torch.where(
                finished,
                torch.full_like(
                    next_token,
                    self.pad_token_id,
                ),
                next_token,
            )

            generated = torch.cat(
                [
                    generated,
                    next_token[:, None],
                ],
                dim=1,
            )

            finished = (
                finished
                | (
                    next_token
                    == self.eos_token_id
                )
            )

            if finished.all():
                break

        return generated.view(
            K,
            B,
            -1,
        )