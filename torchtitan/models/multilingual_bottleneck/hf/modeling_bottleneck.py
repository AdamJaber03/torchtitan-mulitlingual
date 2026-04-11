import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from .configuration_bottleneck import MultilingualBottleneckConfig

# IMPORT YOUR ORIGINAL TORCHTITAN CLASSES HERE
from torchtitan.models.llama3.model import Llama3TransformerBlock
from torchtitan.components.loss import build_loss
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    FeedForward, 
    GQAttention, 
    RoPE
)

class MultilingualBottleneckForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MultilingualBottleneckConfig
    
    _tied_weights_keys = []
    
    def __init__(self, config):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        self.config = config
        self.dim = config.dim
        self.vocab_size = config.vocab_size

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # ------------------------------------------------------------------
        # Bridge HF Config to Torchtitan Configs
        # ------------------------------------------------------------------
        ffn_hidden_dim = compute_ffn_hidden_dim(config.dim, multiple_of=256, ffn_dim_multiplier=1.0)
        tt_layer_config = Llama3TransformerBlock.Config(
            feed_forward=FeedForward.Config(hidden_dim=ffn_hidden_dim),
            attention=GQAttention.Config(
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                attn_backend="sdpa",  # Use standard SDPA for HF inference compatibility
                rope_backend="complex",
            ),
            norm_eps=config.norm_eps
        )
        
        tt_rope_config = RoPE.Config(
            dim=config.dim // config.n_heads,
            max_seq_len=config.max_position_embeddings,
            theta=10000,
            backend="complex",
            scaling="llama",
        )

        # ------------------------------------------------------------------
        # Architecture Initialization
        # ------------------------------------------------------------------
        self.rope = tt_rope_config.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.encoders = nn.ModuleDict()
        for lang in range(config.num_languages):
            layers = nn.ModuleDict()
            for i in range(config.encoder_depth):
                layers[str(i)] = Llama3TransformerBlock(
                    tt_layer_config, layer_id=i, dim=config.dim, n_layers=config.encoder_depth
                )
            self.encoders[str(lang)] = layers

        self.shared_backbone = nn.ModuleDict()
        for i in range(config.backbone_depth):
            self.shared_backbone[str(i)] = Llama3TransformerBlock(
                tt_layer_config, layer_id=i, dim=config.dim, n_layers=config.backbone_depth
            )

        self.decoders = nn.ModuleDict()
        for lang in range(config.num_languages):
            layers = nn.ModuleDict()
            for i in range(config.decoder_depth):
                layers[str(i)] = Llama3TransformerBlock(
                    tt_layer_config, layer_id=i, dim=config.dim, n_layers=config.decoder_depth
                )
            self.decoders[str(lang)] = layers

        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    # 4. Add these 4 standard HF methods right below __init__
    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def forward(self, input_ids, attention_mask=None, labels=None, lang_id=None, **kwargs):
        # 1. Provide a default lang_id fallback for safety
        if lang_id is None:
            lang_id_str = "0"
        else:
            lang_id_str = str(int(lang_id[0].item())) if isinstance(lang_id, torch.Tensor) else str(lang_id)

        B, N_original = input_ids.shape
        K = self.config.k_factor

        # 2. DYNAMIC PADDING TO SATISFY K FACTOR
        # If N is not divisible by K, pad it with EOS/0 tokens.
        pad_len = (K - (N_original % K)) % K
        if pad_len > 0:
            pad_tensor = torch.zeros((B, pad_len), dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, pad_tensor], dim=1)
        
        B, N = input_ids.shape

        # ------------------------------------------------------------------
        # Core Forward Logic
        # ------------------------------------------------------------------
        x = self.tok_embeddings(input_ids)
        
        freqs_cis = self.freqs_cis.to(x.device)
        
        # Dynamic Positional Indices
        pos_enc_1d = torch.arange(N, device=x.device)
        pos_enc_2d = pos_enc_1d.unsqueeze(0).expand(B, N)
        
        use_bb_rope = getattr(self.config, "use_backbone_rope", False)
        if use_bb_rope:
            pos_bb_1d = torch.arange(N // K, device=x.device)
        else:
            pos_bb_1d = torch.zeros(N // K, dtype=torch.long, device=x.device)
        pos_bb_2d = pos_bb_1d.unsqueeze(0).expand(B, N // K)
        
        pos_dec_1d = torch.arange(K, device=x.device).repeat(N // K)
        pos_dec_2d = pos_dec_1d.unsqueeze(0).expand(B, N)

        # STAGE 1: Encoder
        for block in self.encoders[lang_id_str].values():
            x = block(x, freqs_cis=freqs_cis, attention_masks=None, positions=pos_enc_2d)

        x_grouped = x.view(B, N // K, K, self.dim)
        x_pooled = x_grouped[:, :, 0, :] 

        # STAGE 2: Backbone
        x_backbone = x_pooled
        for block in self.shared_backbone.values():
            x_backbone = block(x_backbone, freqs_cis=freqs_cis, attention_masks=None, positions=pos_bb_2d)

        x_upscaled = x_backbone.repeat_interleave(K, dim=1)

        # STAGE 3: Decoder
        x_decoded = x_upscaled
        for block in self.decoders[lang_id_str].values():
            x_decoded = block(x_decoded, freqs_cis=freqs_cis, attention_masks=None, positions=pos_dec_2d)
            
        out = self.norm(x_decoded)
        logits = self.output(out)
        
        # ------------------------------------------------------------------
        # 3. RE-SLICE TO ORIGINAL LENGTH
        # Discard the garbage logits generated by the dynamic padding
        # ------------------------------------------------------------------
        if pad_len > 0:
            logits = logits[:, :N_original, :]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Allow standard HF .generate() to pass lang_id through
        lang_id = kwargs.get("lang_id", None)
        return {"input_ids": input_ids, "lang_id": lang_id}