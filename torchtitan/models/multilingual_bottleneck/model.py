# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from dataclasses import dataclass
import torch
import torch.nn as nn

from torchtitan.models.common.decoder import Decoder
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock

class MultilingualBottleneckModel(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        k_factor: int = 4
        num_languages: int = 2
        use_backbone_rope: bool = False

        encoder_depth: int = 2
        backbone_depth: int = 8
        decoder_depth: int = 2

        enable_shared_embeddings: bool = False
        shared_embeddings_init: bool = False

    def __init__(self, config: Config):
        # We explicitly inherit from nn.Module via Decoder to prevent 
        # the base class from eagerly allocating unneeded standard layers
        super(Decoder, self).__init__()
        self.config = config
        self.dim = config.dim
        self.vocab_size = config.vocab_size

        self.tok_embeddings = nn.ModuleDict()
        for lang in range(config.num_languages):
            lang_vocab_size = config.vocab_size[str(lang)] if isinstance(config.vocab_size, dict) else config.vocab_size
            self.tok_embeddings[str(lang)] = nn.Embedding(lang_vocab_size, config.dim)

        # ------------------------------------------------------------------
        # META'S EXACT ROPE REGISTRATION: 
        # Register as a persistent buffer during __init__
        # ------------------------------------------------------------------
        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.encoders = nn.ModuleDict()
        for lang in range(config.num_languages):
            layers = nn.ModuleDict()
            for i in range(config.encoder_depth):
                layers[str(i)] = Llama3TransformerBlock(
                    config.layer, layer_id=i, dim=config.dim, n_layers=config.encoder_depth
                )
            self.encoders[str(lang)] = layers

        self.shared_backbone = nn.ModuleDict()
        for i in range(config.backbone_depth):
            self.shared_backbone[str(i)] = Llama3TransformerBlock(
                config.layer, layer_id=i, dim=config.dim, n_layers=config.backbone_depth
            )

        self.decoders = nn.ModuleDict()
        for lang in range(config.num_languages):
            layers = nn.ModuleDict()
            for i in range(config.decoder_depth):
                layers[str(i)] = Llama3TransformerBlock(
                    config.layer, layer_id=i, dim=config.dim, n_layers=config.decoder_depth
                )
            self.decoders[str(lang)] = layers

        self.norm = nn.ModuleDict()
        for lang in range(config.num_languages):
            self.norm[str(lang)] = nn.RMSNorm(config.dim, eps=config.layer.norm_eps)
        self.output = nn.ModuleDict()
        for lang in range(config.num_languages):
            lang_vocab_size = config.vocab_size[str(lang)] if isinstance(config.vocab_size, dict) else config.vocab_size
            self.output[str(lang)] = nn.Linear(config.dim, lang_vocab_size, bias=False)

        self.enable_weight_tying = config.enable_weight_tying

        if self.enable_weight_tying:
            for lang in range(config.num_languages):
                self.tok_embeddings[str(lang)].weight = self.output[str(lang)].weight

        self.enable_shared_embeddings = config.enable_shared_embeddings

        if self.enable_shared_embeddings:
            for lang in range(config.num_languages):
                self.tok_embeddings[str(lang)].weight = self.tok_embeddings["0"].weight
                self.output[str(lang)].weight = self.output["0"].weight
        
        self.shared_embeddings_init = config.shared_embeddings_init


    @torch.no_grad()
    def init_weights(self, **kwargs):
        buffer_device: torch.device | None = kwargs.get("buffer_device")
        buffer_device = buffer_device or self.freqs_cis.device

        # META'S EXACT CACHE RE-MATERIALIZATION
        if self.rope is not None:
            self.rope.init_weights(buffer_device=buffer_device)
            self.freqs_cis = self.rope.cache

        if self.tok_embeddings is not None:
            for lang in range(self.config.num_languages):
                nn.init.normal_(self.tok_embeddings[str(lang)].weight, mean=0.0, std=0.02)
        if self.output is not None:
            for lang in range(self.config.num_languages):
                nn.init.normal_(self.output[str(lang)].weight, mean=0.0, std=0.02)

        # PASS **KWARGS ONLY (prevents the double-argument TypeError)
        for lang_layers in self.encoders.values():
            for block in lang_layers.values():
                block.init_weights(**kwargs)
                
        for block in self.shared_backbone.values():
            block.init_weights(**kwargs)
            
        for lang_layers in self.decoders.values():
            for block in lang_layers.values():
                block.init_weights(**kwargs)

        if self.norm is not None:
            for lang in range(self.config.num_languages):
                self.norm[str(lang)].reset_parameters()
        # The token embedding initialization produces weights with too large
        # standard deviation for the output layer. Under weight_tying, both should
        # use the output weights with a smaller, truncated normal distribution to
        # improve training stability.
        if self.enable_weight_tying:
            # since when the model is initialized on meta device,
            # the tying in the __init__ may not have worked correctly
            # we ensure the weights are tied here
            assert self.tok_embeddings is not None and self.output is not None
            for lang in range(self.config.num_languages):
                self.tok_embeddings[str(lang)].weight = self.output[str(lang)].weight
        if self.enable_shared_embeddings:
            for lang in range(self.config.num_languages):
                self.tok_embeddings[str(lang)].weight = self.tok_embeddings["0"].weight
                self.output[str(lang)].weight = self.output["0"].weight

        if self.shared_embeddings_init:
            assert self.tok_embeddings is not None and self.output is not None
            for lang in range(self.config.num_languages):
                with torch.no_grad():
                    self.tok_embeddings[str(lang)].weight.copy_(self.tok_embeddings["0"].weight)
                    self.output[str(lang)].weight.copy_(self.output["0"].weight)

    def forward(self, x, **kwargs):
        # Your custom forward pass logic exactly as written before...
        tokens = x
        lang_id_str = str(int(kwargs["lang_id"][0].item()))
        assert torch.all(kwargs["lang_id"] == kwargs["lang_id"][0]), "All samples in the batch must have the same lang_id"

        B, N = tokens.shape
        K = self.config.k_factor
        assert N % K == 0, f"Sequence length {N} must be divisible by K ({K})"

        x = self.tok_embeddings[lang_id_str](tokens)
        
        # ------------------------------------------------------------------
        # Bulletproof Device Sync: Fetch the re-materialized buffer and 
        # explicitly guarantee it's on the input tensor's device
        # ------------------------------------------------------------------
        freqs_cis = self.freqs_cis.to(x.device)
        
        # Dynamic Positional Indices
        pos_enc_1d = torch.arange(N, device=x.device)
        pos_enc_2d = pos_enc_1d.unsqueeze(0).expand(B, N)
        
        if self.config.use_backbone_rope:
            pos_bb_1d = torch.arange(N // K, device=x.device)
        else:
            pos_bb_1d = torch.zeros(N // K, dtype=torch.long, device=x.device)
        pos_bb_2d = pos_bb_1d.unsqueeze(0).expand(B, N // K)
        
        #TODO - figure out decoder positional embeddings 
        # pos_dec_1d = torch.arange(K, device=x.device).repeat(N // K)
        # pos_dec_2d = pos_dec_1d.unsqueeze(0).expand(B, N)
        pos_dec_1d = torch.arange(N, device=x.device)
        pos_dec_2d = pos_dec_1d.unsqueeze(0).expand(B, N)

        # STAGE 1: Encoder
        for block in self.encoders[lang_id_str].values():
            x = block(x, freqs_cis=freqs_cis, attention_masks=None, positions=pos_enc_2d)

        x_grouped = x.view(  B, N // K, K, self.dim)
        x_pooled = x_grouped[:, :, 0, :] #x_grouped[:, :, -1, :] 

        # STAGE 2: Backbone
        x_backbone = x_pooled
        for block in self.shared_backbone.values():
            x_backbone = block(x_backbone, freqs_cis=freqs_cis, attention_masks=None, positions=pos_bb_2d)

        x_upscaled = x_backbone.repeat_interleave(K, dim=1)

        # STAGE 3: Decoder
        x_decoded = x_upscaled
        for block in self.decoders[lang_id_str].values():
            x_decoded = block(x_decoded, freqs_cis=freqs_cis, attention_masks=None, positions=pos_dec_2d)

        out = self.norm[lang_id_str](x_decoded)
        logits = self.output[lang_id_str](out)
        
        return {
            "output": logits
        }