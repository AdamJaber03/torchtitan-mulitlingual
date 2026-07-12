import json
import random
from torchtitan.hf_datasets.value_schedualers import SCHEDUALER_REGISTRY
from torchtitan.tools.logging import logger
from multiprocessing import Value


def _extract_vocab(tokenizer) -> dict:
    """Return the {token_str: id} vocab, handling HF/torchtitan tokenizer wrappers."""
    if hasattr(tokenizer, "vocab"):
        return tokenizer.vocab
    if hasattr(tokenizer, "get_vocab"):
        return tokenizer.get_vocab()
    if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "get_vocab"):
        return tokenizer._tokenizer.get_vocab()
    raise AttributeError("Could not extract vocabulary from the tokenizer object.")


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
        self.tag_prob = config.get("prob", None)
        self.prob_schedualer = config.get("prob_schedualer", None)
        if self.prob_schedualer is not None:
            assert self.tag_prob is None, "Cannot specify both a fixed tag_prob and a prob_schedualer. Please choose one."
            if self.prob_schedualer["name"] not in SCHEDUALER_REGISTRY:
                raise ValueError(f"Prob schedualer '{self.prob_schedualer['name']}' is not registered. Available: {list(SCHEDUALER_REGISTRY.keys())}")
            self.prob_schedualer = SCHEDUALER_REGISTRY[self.prob_schedualer["name"]](**{k:v for k,v in config.get("prob_schedualer", {}).items() if k != "name"})
            self.tag_prob = self.prob_schedualer(0)  # Initialize with the starting probability
            logger.info(f"Initialized {self.name} with dynamic prob schedualer '{self.prob_schedualer.__class__.__name__}', starting at tag_prob={self.tag_prob}")
        self.tag_prob = Value('d', self.tag_prob)  # For shared memory access if needed
        assert self.tag_prob is not None, f"[{self.name}] config must include 'prob' or 'prob_schedualer'"
        self.idx = config.get("idx", None)
        self.vocab_size = config.get("vocab_size")
        self.special_tokens = set(config.get("special_tokens", []))
        self.symmetric = config.get("symmetric", False)  # If True, use 50% of the time a 1-p tagging to mirror
        self.tag_n = config.get("tag_n", 1)
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
                
        logger.info(f"Initializing {self.name} with prob={self.tag_prob}. Cached {len(self.boundary_token_ids)} boundary tokens.")
    
    def step(self, global_step):
        logger.info(f"************Stepping {self.name} augmentation at global step {global_step}...*********************")
        if self.prob_schedualer is not None:
            self.tag_prob.value = self.prob_schedualer(global_step)
            logger.info(f"Updated tag_prob to {self.tag_prob.value} based on schedualer at step {global_step}")

    def __call__(self, tokens_in: dict|list[dict], dataset_name: str = None) -> list:
        if self.tag_prob.value <= 0.0 or not tokens_in:
            return tokens_in
            
        if self.idx is not None:
            tokens = tokens_in[self.idx]["tokens"]
        else:
            tokens = tokens_in["tokens"]


        tag_prob = self.tag_prob.value
        if self.symmetric:
            # If symmetric, we want to create a balanced tagging distribution.
            # So we randomly decide to flip the tagging direction for this sequence.
            if random.random() < 0.5:
                tag_prob = 1.0 - self.tag_prob.value  # Flip the probability to tag the opposite set of words
        shifted_tokens = []
        tag_current_word = False  # Tracks if the current word is being shifted
        
        for i, token_id in enumerate(tokens):
            if token_id in self.special_tokens:
                shifted_tokens.append(token_id)
                continue
                
            # If we are at the very first token (index 0) OR we hit a space marker, 
            # we have hit a new word. Roll the dice to decide this word's fate.
            if i == 0 or token_id in self.boundary_token_ids:
                tag_current_word = random.random() < tag_prob
                
            # Apply the shift based on the current state
            if tag_current_word:
                shifted_tokens.append(token_id%self.vocab_size + self.tag_n*self.vocab_size)
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

        logger.info(f"Initializing WordWiseContrastive. Cached {len(self.boundary_token_ids)} boundary tokens.")

    def __call__(self, tokens_in: dict, initial_id: int) -> tuple[int, list]:
        mask = []
        word_sep_idx = tokens_in.get("word_sep_idx", None)
        assert word_sep_idx is not None, "WordWiseContrastive requires 'word_sep_idx' in the input dict to identify word boundaries."
        encoding = tokens_in.get("encoding", None)
        
        if encoding is None:
            raise ValueError("[WordWiseContrastive] 'encoding' not found in tokens_in! Ensure your dataset pipeline extracts it.")
        cur_token = -1
        for i, sep in enumerate(word_sep_idx):
            sep_token = encoding.char_to_token(sep)
            assert sep_token is not None, f"Encoding failed to map character index {sep} to a token. Check your tokenizer and encoding., {tokens_in['text'][sep]}"
            mask += [i + initial_id + 1] * (sep_token - cur_token)
            cur_token = sep_token
        if self.tokenizer.bos_token is not None:
            mask = [initial_id] + mask
        if len(tokens_in["tokens"]) > len(mask):
            mask += [len(word_sep_idx) + initial_id] * (len(tokens_in["tokens"]) - len(mask))

        # 3. Calculate 'n' (Total unique words added to the sequence)
        n = max(mask) - initial_id if mask else 0

        # 4. Bulletproof Asserts
        assert len(tokens_in["text"]) == word_sep_idx[-1] + 1, f"Last word_sep_idx {word_sep_idx[-1]} does not align with text length {len(tokens_in['text'])}"
        # assert encoding.char_to_token(len(tokens_in["text"])-1) + 1 == len(tokens_in["tokens"]) - 1, f"Encoding does not align with tokens: last char maps to token {encoding.char_to_token(len(tokens_in['text'])-1)}, expected {len(tokens_in['tokens']) - 1}"
        assert len(mask) == len(tokens_in["tokens"]), \
            f"Fatal alignment error: mask len {len(mask)} != tokens len {len(tokens_in['tokens'])}"

        return n, mask
    # def __call__(self, tokens_in: dict, initial_id: int) -> list:
    #     mask = []
    #     word_mask = tokens_in.get("word_mask", None)
    #     current_word_id = initial_id
    #     logger.info(f"****************{len(word_mask)} ____________ {sum(t in self.boundary_token_ids for t in tokens_in['tokens'])}****************")

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

class TokenPrefix:
    """
    add a prefix token to the beginning of the sequence
    """
    def __init__(self, config: dict):
        self.name = config.get("name", "token_prefix")
        self.prefix_token_id = config.get("prefix_token_id", None)
        self.idx = config.get("idx", None)  # Optional: if you want to apply the prefix to a different list of tokens in the input dict
        if self.prefix_token_id is None:
            raise ValueError(f"[{self.name}] config must include 'prefix_token_id'")
        logger.info(f"Initializing {self.name} augmentation with prefix_token_id={self.prefix_token_id}...")                

    def __call__(self, tokens_in: dict|list[dict], dataset_name: str = None) -> list:            
        if self.idx is not None:
            tokens = tokens_in[self.idx]["tokens"]
        else:
            tokens = tokens_in["tokens"]
        tokens = [self.prefix_token_id] + tokens
        if self.idx is not None:
            tokens_in[self.idx]["tokens"] = tokens
            return tokens_in
        tokens_in["tokens"] = tokens
        return tokens_in


class SharedAnchorRemap:
    """
    Deterministic post-tokenization remap of Arabic tokens to their English
    counterpart id, for 1-to-1 (single-token <-> single-token) translation pairs.

    This creates a shared cross-lingual anchor: the Arabic token id is replaced by
    the English token id, so both languages index the same embedding / output row.
    The tokenizer itself is left untouched (a vocab-level id remap is NOT viable --
    HF BPE represents merges by token id, so reassigning ids corrupts the merges),
    hence the remap is applied here, after the BPE has produced correct ids.

    Word-boundary safe (default): a mapped token is only remapped when it constitutes
    a COMPLETE word, i.e. the next token starts a new word (begins with the byte-level
    space marker), is a special token, or is end-of-sequence. This prevents remapping a
    mapped function-word token that appears as a prefix subword of a longer word.
    The mapped ids are space-prefixed forms, so they are always word starts already.
    """

    def __init__(self, config: dict):
        self.name = config.get("name", "shared_anchor_remap")
        self.idx = config.get("idx", None)
        self.word_boundary_safe = config.get("word_boundary_safe", True)

        map_path = config.get("map_path")
        if map_path is None:
            raise ValueError(f"[{self.name}] config must include 'map_path'")
        with open(map_path) as f:
            data = json.load(f)
        if "id_remap" in data:
            self.remap = {int(k): int(v) for k, v in data["id_remap"].items()}
        elif "pairs" in data:
            self.remap = {int(p["arabic_id"]): int(p["english_id"]) for p in data["pairs"]}
        else:
            raise ValueError(f"[{self.name}] map file must contain 'id_remap' or 'pairs'")

        self.tokenizer = config.get("tokenizer")
        if self.tokenizer is None:
            raise ValueError(f"[{self.name}] requires 'tokenizer' to be passed in config.")

        # Special tokens act as word boundaries and are never remapped.
        self.special_tokens = set(config.get("special_tokens", []))
        for attr in ("eos_id", "bos_id"):
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                self.special_tokens.add(tid)

        # Tokens that mark the start of a new word (byte-level space marker).
        self.boundary_token_ids = set()
        if self.word_boundary_safe:
            for token_str, token_id in _extract_vocab(self.tokenizer).items():
                if token_str.startswith("Ġ") or token_str.startswith(" "):
                    self.boundary_token_ids.add(token_id)
        # A new word / sequence-end starts here -> the previous token is a complete word.
        self._end_markers = self.boundary_token_ids | self.special_tokens

        logger.info(
            f"Initializing {self.name}: {len(self.remap)} remap entries, "
            f"word_boundary_safe={self.word_boundary_safe}, "
            f"{len(self.boundary_token_ids)} boundary tokens."
        )

    def __call__(self, tokens_in: dict | list[dict], dataset_name: str = None):
        if not tokens_in:
            return tokens_in
        if self.idx is not None:
            tokens = tokens_in[self.idx]["tokens"]
        else:
            tokens = tokens_in["tokens"]

        n = len(tokens)
        remapped = []
        for i, tid in enumerate(tokens):
            new_id = tid
            if tid in self.remap:
                if not self.word_boundary_safe:
                    new_id = self.remap[tid]
                else:
                    is_word_end = (i == n - 1) or (tokens[i + 1] in self._end_markers)
                    if is_word_end:
                        new_id = self.remap[tid]
            remapped.append(new_id)

        if self.idx is not None:
            tokens_in[self.idx]["tokens"] = remapped
            return tokens_in
        tokens_in["tokens"] = remapped
        return tokens_in


POST_TOKEN_AUGMENTATIONS_REGISTRY = {
    "stochastic_token_tagging": StochasticTokenTagging,
    "stochastic_word_tagging": StochasticWordTokenTagging,
    "token_prefix": TokenPrefix,
    "shared_anchor_remap": SharedAnchorRemap,
}
