import random

class StochasticTokenTagging:
    """
    On-the-fly stochastic token ID shifting to create a fictive tagged English space.
    NOTE: This must be applied AFTER tokenization (expects List[int] or torch.Tensor of IDs).
    """
    def __init__(self, config: dict):
        self.name = config.get("name", "stochastic_token_tagging")
        self.tag_prob = config.get("prob", 0.5)
        self.idx = config.get("idx", None)  # Optional: if you want to shift by a different amount than vocab_size
        # The base vocabulary size (V) is required to know how far to shift the IDs
        self.vocab_size = config.get("vocab_size")
        if self.vocab_size is None:
            raise ValueError(f"[{self.name}] config must include 'vocab_size'")
            
        # Optional: special tokens (like BOS, EOS, PAD) that should never be shifted
        self.special_tokens = set(config.get("special_tokens", []))
        
        print(f"Initializing {self.name} augmentation with prob={self.tag_prob}, shift={self.vocab_size}...")

    def __call__(self, tokens_in: dict|list[dict], dataset_name: str = None) -> list:
        """
        Applies the stochastic ID shift to an encoded sequence.
        
        Args:
            tokens: A list of integer token IDs (already encoded by the BPE tokenizer).
            dataset_name: Included for signature compatibility with the pipeline.
        """
        if self.tag_prob <= 0.0 or not tokens_in:
            return tokens_in
        if self.idx is not None:
            assert type(tokens_in[self.idx]) == dict, "If 'idx' is specified, input tokens must be a list of dicts"
            tokens = tokens_in[self.idx]["tokens"]
        else:
            tokens = tokens_in["tokens"]

        shifted_tokens = []
        for token_id in tokens:
            # Skip shifting for special tokens so sequence boundaries remain consistent
            if token_id in self.special_tokens:
                shifted_tokens.append(token_id)
            else:
                # Stochastically shift by vocab_size
                if random.random() < self.tag_prob:
                    shifted_tokens.append(token_id + self.vocab_size)
                else:
                    shifted_tokens.append(token_id)
        if self.idx is not None:
            # If we were given a list of lists, we need to put the shifted tokens back in the right place
            tokens_in[self.idx]["tokens"] = shifted_tokens
            return tokens_in  
        tokens_in["tokens"] = shifted_tokens
        return tokens_in

class StochasticWordTokenTagging:
    """
    On-the-fly stochastic token ID shifting to create a fictive tagged English space.
    Operates WORD-WISE: If a word boundary is hit, it rolls a probability. 
    All subsequent subwords inherit that state until the next word boundary.
    """
    def __init__(self, config: dict):
        self.name = config.get("name", "stochastic_word_token_tagging")
        self.tag_prob = config.get("prob", 0.5)
        self.idx = config.get("idx", None)
        self.vocab_size = config.get("vocab_size")
        self.special_tokens = set(config.get("special_tokens", []))
        
        if self.vocab_size is None:
            raise ValueError(f"[{self.name}] config must include 'vocab_size'")
            
        # 1. Grab the tokenizer injected from the Dataset
        self.tokenizer = config.get("tokenizer")
        if self.tokenizer is None:
            raise ValueError(f"[{self.name}] requires 'tokenizer' to be passed in config.")

        # 2. Pre-compute word-boundary token IDs for O(1) lookup
        self.boundary_token_ids = set()
        
        # Extract the raw vocab dictionary (handles differences in HF/TorchTitan wrappers)
        if hasattr(self.tokenizer, "vocab"):
            vocab = self.tokenizer.vocab
        elif hasattr(self.tokenizer, "get_vocab"):
            vocab = self.tokenizer.get_vocab()
        # Fallback if torchtitan hides the HF tokenizer inside a property
        elif hasattr(self.tokenizer, "_tokenizer") and hasattr(self.tokenizer._tokenizer, "get_vocab"):
            vocab = self.tokenizer._tokenizer.get_vocab()
        else:
            raise AttributeError("Could not extract vocabulary from the tokenizer object.")

        # Identify all tokens that start with the BPE/SentencePiece space marker
        for token_str, token_id in vocab.items():
            if token_str.startswith("Ġ") or token_str.startswith(" "):
                self.boundary_token_ids.add(token_id)
                
        print(f"Initializing {self.name} with prob={self.tag_prob}. Cached {len(self.boundary_token_ids)} boundary tokens.")

    def __call__(self, tokens_in: dict|list[dict], dataset_name: str = None) -> list:
        if self.tag_prob <= 0.0 or not tokens_in:
            return tokens_in
            
        if self.idx is not None:
            tokens = tokens_in[self.idx]["tokens"]
        else:
            tokens = tokens_in["tokens"]

        shifted_tokens = []
        tag_current_word = False  # Tracks if the current word is being shifted
        
        for i, token_id in enumerate(tokens):
            if token_id in self.special_tokens:
                shifted_tokens.append(token_id)
                continue
                
            # If we are at the very first token (index 0) OR we hit a space marker, 
            # we have hit a new word. Roll the dice to decide this word's fate.
            if i == 0 or token_id in self.boundary_token_ids:
                tag_current_word = random.random() < self.tag_prob
                
            # Apply the shift based on the current state
            if tag_current_word:
                shifted_tokens.append(token_id + self.vocab_size)
            else:
                shifted_tokens.append(token_id)

        if self.idx is not None:
            tokens_in[self.idx]["tokens"] = shifted_tokens
            return tokens_in
        tokens_in["tokens"] = shifted_tokens
        return tokens_in

class WordWiseContrastive:
    def __init__(self, tokenizer):
        # 1. Grab the tokenizer injected from the Dataset
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            raise ValueError(f"[WordWiseContrastive] requires 'tokenizer' to be passed in config.")

        # 2. Pre-compute word-boundary token IDs for O(1) lookup
        self.boundary_token_ids = set()
        
        # Extract the raw vocab dictionary (handles differences in HF/TorchTitan wrappers)
        if hasattr(self.tokenizer, "vocab"):
            vocab = self.tokenizer.vocab
        elif hasattr(self.tokenizer, "get_vocab"):
            vocab = self.tokenizer.get_vocab()
        # Fallback if torchtitan hides the HF tokenizer inside a property
        elif hasattr(self.tokenizer, "_tokenizer") and hasattr(self.tokenizer._tokenizer, "get_vocab"):
            vocab = self.tokenizer._tokenizer.get_vocab()
        else:
            raise AttributeError("Could not extract vocabulary from the tokenizer object.")

        # Identify all tokens that start with the BPE/SentencePiece space marker
        for token_str, token_id in vocab.items():
            if token_str.startswith("Ġ") or token_str.startswith(" "):
                self.boundary_token_ids.add(token_id)

        print(f"Initializing WordWiseContrastive. Cached {len(self.boundary_token_ids)} boundary tokens.")

    def __call__(self, tokens_in: dict, initial_id: int) -> tuple[int, list]:
        mask = []
        word_mask = tokens_in.get("word_mask", None)
        word_ids = tokens_in.get("word_ids", None)
        
        if word_ids is None:
            raise ValueError("[WordWiseContrastive] 'word_ids' not found in tokens_in! Ensure your dataset pipeline extracts it.")
        # We use a separate index to track our position in the augmented word_mask.
        # This allows us to "pause" whenever we hit a BOS/EOS token in the sequence.
        word_mask_idx = 0

        for subword_idx, original_word_id in enumerate(word_ids):
            
            # 1. Handle Special Tokens (BOS/EOS)
            # Hugging Face natively marks these as 'None' in the word_ids array.
            if original_word_id is None:
                if subword_idx == 0:
                    # It's the BOS token. Assign it the base initial_id.
                    mask.append(initial_id)
                else:
                    # It's the EOS token (or padding). Inherit the ID of the final word.
                    mask.append(mask[-1])
                continue

            # 2. Handle Actual Words
            if word_mask is not None:
                # Use the pre-computed mask from your augmentation
                mapped_id = word_mask[word_mask_idx] + initial_id + 1
                word_mask_idx += 1
            else:
                # If no augmentation occurred, just use the pre-tokenizer's word index directly!
                mapped_id = original_word_id + initial_id + 1
                
            mask.append(mapped_id)

        # 3. Calculate 'n' (Total unique words added to the sequence)
        n = max(mask) - initial_id if mask else 0

        # 4. Bulletproof Asserts
        assert len(mask) == len(tokens_in["tokens"]), \
            f"Fatal alignment error: mask len {len(mask)} != tokens len {len(tokens_in['tokens'])}"
            
        if word_mask is not None:
            assert word_mask_idx == len(word_mask), \
                f"Did not consume the entire word_mask! Left over {len(word_mask) - word_mask_idx} items."

        return n, mask
    # def __call__(self, tokens_in: dict, initial_id: int) -> list:
    #     mask = []
    #     word_mask = tokens_in.get("word_mask", None)
    #     current_word_id = initial_id
    #     print(f"****************{len(word_mask)} ____________ {sum(t in self.boundary_token_ids for t in tokens_in['tokens'])}****************")

    #     for i, token_id in enumerate(tokens_in["tokens"]):
    #         # If we are at the very first token (index 0) OR we hit a space marker,
    #         # we have hit a new word. Roll the dice to decide this word's fate.
    #         if i==0 or token_id in self.boundary_token_ids:
    #             current_word_id += 1
    #         if word_mask is not None:
    #             assert current_word_id - initial_id - 1 < len(word_mask), f"Word ID {current_word_id- initial_id - 1} exceeds word_mask length {len(word_mask)} with initial_id {initial_id}, tokens_in sep count {sum(t in self.boundary_token_ids for t in tokens_in['tokens'])}"
    #             mask.append(word_mask[current_word_id-initial_id-1] + initial_id + 1)
    #         else:
    #             mask.append(current_word_id)
    #     n = current_word_id - initial_id if word_mask is None else word_mask[current_word_id-initial_id-1] + initial_id + 1
    #     assert current_word_id - initial_id == len(word_mask) if word_mask is not None else True, f"Expected {n} unique words, but got word_mask with {len(word_mask)} elements and max ID {max(word_mask)}"
    #     return n, mask


POST_TOKEN_AUGMENTATIONS_REGISTRY = {
    "stochastic_token_tagging": StochasticTokenTagging,
    "stochastic_word_tagging": StochasticWordTokenTagging,
}
