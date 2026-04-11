from transformers import PretrainedConfig

class MultilingualBottleneckConfig(PretrainedConfig):
    model_type = "multilingual_bottleneck"

    def __init__(
        self,
        vocab_size=32768,
        dim=768,
        k_factor=4,
        num_languages=2,
        use_backbone_rope=False,  # Added this flag
        encoder_depth=4,
        backbone_depth=12,
        decoder_depth=4,
        n_heads=12,
        n_kv_heads=12,
        max_position_embeddings=512,
        norm_eps=1e-5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.k_factor = k_factor
        self.num_languages = num_languages
        self.use_backbone_rope = use_backbone_rope  # Assigned it to self
        self.encoder_depth = encoder_depth
        self.backbone_depth = backbone_depth
        self.decoder_depth = decoder_depth
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.norm_eps = norm_eps
        super().__init__(**kwargs)