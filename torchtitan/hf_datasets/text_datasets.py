import json
import os
import random
import torch
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, List
from functools import partial
from pathlib import Path

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger
from torchtitan.hf_datasets.mixed_dataset import MixedHuggingFaceDataset
from torchtitan.hf_datasets.augmentations import AUGMENTATIONS_REGISTRY

from torchtitan.hf_datasets.post_tokenization_augmentations import POST_TOKEN_AUGMENTATIONS_REGISTRY

import random
from datasets import IterableDataset as HFDIterableDataset
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.hf_datasets.post_tokenization_augmentations import WordWiseContrastive

WORDWISE_CONTRASTIVE_ENABLED = True  # Set to False to disable the word-wise contrastive augmentation
MAX_SEQS = 128

def encode_with_token_metadata(tokenizer, text):
    encoding = tokenizer.tokenizer.encode(text)
    # 2. Replicate your wrapper's BOS/EOS logic
    bos_id = getattr(tokenizer, "bos_id", None)
    eos_id = getattr(tokenizer, "eos_id", None)
    bos = [bos_id] if bos_id is not None else []
    eos = [eos_id] if eos_id is not None else []

    # 3. Store the tokens exactly as your wrapper would have
    tokens = bos + encoding.ids + eos

    # 4. Store the PERFECT word mapping (Padding with None for the BOS/EOS tokens)
    bos_pad = [None] * len(bos)
    eos_pad = [None] * len(eos)
    word_ids = bos_pad + encoding.word_ids + eos_pad
    offsets = bos_pad + encoding.offsets + eos_pad
    return tokens, word_ids, offsets

def encode_with_word_ids(tokenizer, text):
    tokens, word_ids, _ = encode_with_token_metadata(tokenizer, text)
    return tokens, word_ids

def buffered_shuffle(iterator, buffer_size=10000):
    """Shuffles an infinite iterator using a local memory buffer."""
    buffer = []
    for item in iterator:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            # Pick a random index, swap it with the last element, and pop it (O(1) operation)
            idx = random.randint(0, len(buffer) - 1)
            buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
            yield buffer.pop()
    
    # Once the source iterator is completely exhausted, shuffle and yield the remaining buffer
    random.shuffle(buffer)
    for item in buffer:
        yield item

# --- Existing Helper Functions ---
def _multilingual_pretraining_root() -> Path:
    root = os.environ.get("MULTILINGUAL_PRETRAINING_ROOT")
    if root:
        return Path(root).expanduser()

    torchtitan_root = Path(os.environ.get("TORCHTITAN_ROOT", Path.cwd())).resolve()
    if torchtitan_root.name == "multilingual-pretraining":
        return torchtitan_root
    return torchtitan_root.parent

def _multilingual_data_root() -> Path:
    return Path(
        os.environ.get(
            "MULTILINGUAL_DATA_ROOT",
            str(_multilingual_pretraining_root() / "data"),
        )
    ).expanduser()

def _fineweb_translated_root() -> Path:
    return Path(
        os.environ.get(
            "FINEWEB_TRANSLATED_ROOT",
            str(_multilingual_data_root() / "fineweb_translated"),
        )
    ).expanduser()

def _fineweb_paired_shards_root() -> Path:
    return Path(
        os.environ.get(
            "FINEWEB_PAIRED_SHARDS_ROOT",
            str(_multilingual_data_root() / "fineweb-edu-ar_paired_shards"),
        )
    ).expanduser()

def _load_dataset(dataset_path: str, start_idx: int, split: str, lang: str | None = None):
    if dataset_path == "karpathy/fineweb-edu-100b-shuffle":
        ld = load_dataset(dataset_path, split=split, streaming=True)
        return ld.skip(start_idx) if start_idx > 0 else ld
    if dataset_path == str(_fineweb_paired_shards_root()):
        ld = load_dataset("parquet",data_dir=str(_fineweb_paired_shards_root()),split="train",streaming=True)
        ld = ld.shuffle(seed=42, buffer_size=20_000)
        return ld.skip(start_idx) if start_idx > 0 else ld
    if dataset_path == "kaust-generative-ai/fineweb-edu-ar":
        translated_root = _fineweb_translated_root()
        if lang == "paired":
            def paired_gen():
                # 1. Load inside the generator (Worker-Safe)
                ar_ds = load_dataset(dataset_path, name="ar", split=split, streaming=True)
                en_ds = load_dataset(dataset_path, name="en", split=split, streaming=True)
                
                # 2. Shard by worker so worker 1 gets different data than worker 2
                # This prevents the 16-worker duplication issue
                from datasets.distributed import split_dataset_by_worker
                ar_ds = split_dataset_by_worker(ar_ds)
                en_ds = split_dataset_by_worker(en_ds)

                if start_idx > 0:
                    # Note: You might need to divide start_idx by world_size 
                    # depending on how you want to resume.
                    ar_ds = ar_ds.skip(start_idx)
                    en_ds = en_ds.skip(start_idx)
                
                paired_stream = zip(ar_ds, en_ds)
                yield from buffered_shuffle(paired_stream, buffer_size=20_000)
            return HFDIterableDataset.from_generator(paired_gen)
        if lang == "tr2en":
            ld = load_dataset("json", data_dir=str(translated_root / "translated"), split="train", streaming=True)
        elif lang == "ar":
            ld = load_dataset("json", data_dir=str(translated_root / "original"), split="train", streaming=True)
        elif lang == "en":
            ld = load_dataset("json", data_dir=str(translated_root / "en-original"), split="train", streaming=True)
        elif lang == "tr2en_1to1map":
            ld = load_dataset("json", data_dir=str(translated_root / "translated_1to1map"), split="train", streaming=True)
        ld = ld.skip(start_idx) if start_idx > 0 else ld
        return ld.shuffle(seed=42, buffer_size=20_000)  # Synchronized shuffle for paired streams
    return load_dataset(dataset_path, name="en", split=split, streaming=True)

def _process_c4_text(sample: dict[str, Any]) -> str:
    return sample["text"]

def _process_paired_text(sample: tuple[dict[str, Any], dict[str, Any]]) -> str:
    ar_text = sample["text_ar"]
    en_text = sample["text_en"]
    if random.random() < 0.5:
        out =  f"{en_text}\n\n\n{ar_text}"
    else:
        out = f"{ar_text}\n\n\n{en_text}"
    return out

def _process_contrastive_text(sample: tuple[dict[str, Any], dict[str, Any]]) -> str:
    ar_text = sample["text_ar"]
    en_text = sample["text_en"]
    return [en_text, ar_text]

DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-100b-shuffle": DatasetConfig(
        path="karpathy/fineweb-edu-100b-shuffle",
        loader=partial(_load_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-ar-ar": DatasetConfig(
        path="kaust-generative-ai/fineweb-edu-ar",
        loader=partial(_load_dataset, split="train", lang="ar"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-ar-en": DatasetConfig(
        path="kaust-generative-ai/fineweb-edu-ar",
        loader=partial(_load_dataset, split="train", lang="en"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-ar-tr2en": DatasetConfig(
        path="kaust-generative-ai/fineweb-edu-ar",
        loader=partial(_load_dataset, split="train", lang="tr2en"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-ar-tr2en_1to1map": DatasetConfig(
        path="kaust-generative-ai/fineweb-edu-ar",
        loader=partial(_load_dataset, split="train", lang="tr2en_1to1map"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-ar-paired": DatasetConfig(
        path=str(_fineweb_paired_shards_root()),
        loader=partial(_load_dataset, split="train"),
        sample_processor=_process_paired_text,
    ),
    "fineweb-edu-ar-paired-contrastive": DatasetConfig(
        path=str(_fineweb_paired_shards_root()),
        loader=partial(_load_dataset, split="train"),
        sample_processor=_process_contrastive_text,
    ),

}

def _validate_dataset(dataset_name: str, dataset_path: str | None = None):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    return path, config.loader, config.sample_processor

class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        injection_paths: List[str] | None = None,
        injection_probs: List[float] | None = None,
        unique_rates: List[int] | None = None,
        eos_token_id: int = 0,
        augmentations: List[dict] | None = None,
        post_token_augmentations: List[dict] | None = None, # <-- Added param
        start_idx: int = 0,
        lang_id: int | None = None,
        enable_contrastive_mask: bool = False,
        contrastive_len_threshold: int = 256,
    ) -> None:
        dataset_name = dataset_name.lower()
        path, dataset_loader, text_processor = _validate_dataset(dataset_name, dataset_path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(dataset_loader(path, start_idx), dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor
        self.eos_token_id = eos_token_id

        # Injection setup
        self.injection_paths = injection_paths or []
        if injection_probs and self.injection_paths:
            total_inj_prob = sum(injection_probs)
            self.probs = torch.tensor([1.0 - total_inj_prob] + injection_probs)
        else:
            self.probs = torch.tensor([1.0])
        
        self.injection_data = []
        for i, p in enumerate(self.injection_paths):
            unique_rate = None
            if unique_rates is not None:
                assert len(unique_rates) == len(self.injection_paths), "Length of unique_rates must match length of injection_paths"
                unique_rate = unique_rates[i]
            with open(p, 'r') as f:
                try:
                    if unique_rate is not None:
                        self.injection_data.append(random.sample([json.loads(line)["text"] for line in f], unique_rate))
                        print(f"Loaded {len(self.injection_data[-1])} unique samples from {p} (requested {unique_rate})")
                    else:
                        self.injection_data.append([json.loads(line)["text"] for line in f])
                except TypeError as e:
                    logger.error(f"Error decoding JSON from {p}: {e}")
                    raise TypeError(f"Failed to load injection data from {p}. Please check the file format.")

        self.injection_counts = torch.zeros(
            len(self.injection_paths), dtype=torch.int64
        ).share_memory_()
        
        # Setup Text Augmentations
        self.aug_callables = []
        augmentations = augmentations or []
        for aug_cfg in augmentations:
            aug_name = aug_cfg.get("name")
            if aug_name in AUGMENTATIONS_REGISTRY:
                aug_kwargs = {k: v for k, v in aug_cfg.items() if k != "name"}
                aug_cfg["tokenizer"] = self._tokenizer
                aug_instance = AUGMENTATIONS_REGISTRY[aug_name](aug_cfg)
                self.aug_callables.append(aug_instance)
            else:
                logger.warning(f"Augmentation '{aug_name}' not found in AUGMENTATIONS_REGISTRY.")

        # --- NEW: Setup Post-Tokenization Augmentations ---
        self.post_token_aug_callables = []
        post_token_augmentations = post_token_augmentations or []
        for aug_cfg in post_token_augmentations:
            aug_name = aug_cfg.get("name")
            if aug_name in POST_TOKEN_AUGMENTATIONS_REGISTRY:
                aug_kwargs = {k: v for k, v in aug_cfg.items() if k != "name"}
                aug_cfg["tokenizer"] = self._tokenizer
                aug_instance = POST_TOKEN_AUGMENTATIONS_REGISTRY[aug_name](aug_cfg)
                self.post_token_aug_callables.append(aug_instance)
            else:
                logger.warning(f"Post-token augmentation '{aug_name}' not found in POST_TOKEN_AUGMENTATIONS_REGISTRY.")

        self._sample_idx = 0
        self._token_buffer: list[int] = []
        self.contrastive_mask_buffer: list[bool] = []  # Buffer to track which tokens are from contrastive samples
        self.lang_id = lang_id
        self.enable_contrastive_mask = enable_contrastive_mask
        self.contrastive_len_threshold = contrastive_len_threshold
        self.contrastive_pair_counter = torch.zeros(1, dtype=torch.int64).share_memory_()  # Counter to track active contrastive pairs
        self.wordwisecontrastive = WordWiseContrastive(tokenizer=self._tokenizer) if WORDWISE_CONTRASTIVE_ENABLED else None
        self.contrastive_pair_idx = 0

    def _apply_augs(self, text: str) -> str:
        for aug_fn in self.aug_callables:
            text = aug_fn(text, dataset_name=self.dataset_name)
        return text

    # --- NEW: Post-tokenization application method ---
    def _apply_post_token_augs(self, tokens: list[int]) -> list[int]:
        for aug_fn in self.post_token_aug_callables:
            tokens = aug_fn(tokens, dataset_name=self.dataset_name)
        return tokens

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            return iter(self._data.skip(self._sample_idx)) if self._sample_idx < len(self._data) else iter([])
        return iter(self._data)

    def _get_injected_tokens(self, file_idx: int) -> list[int]:
        """Fetches a single injected document stochastically and increments the counter."""
        file_content = self.injection_data[file_idx]
        doc = random.choice(file_content)
        print(f"Injecting from file: {self.injection_paths[file_idx]} (Total injections from this file so far: {self.injection_counts[file_idx]})")
        # --- NEW: Increment the view counter for this specific injected file ---
        self.injection_counts[file_idx] += 1
        
        # Optional: Print every 100th injection so you can monitor it in the logs
        # if self.injection_counts[file_idx] % 100 == 0:
        #     logger.info(f"Injection tracker: File {self.injection_paths[file_idx]} has been sampled {self.injection_counts[file_idx]} times.")
        doc = {"text": doc}  # Wrap in dict for augmentation compatibility
        doc = self._apply_augs(doc)
        print(f"Sample injected doc (truncated to 200 chars): {doc['text'][:200]}...")  # Log the injected document for debugging

        # Apply post tokenization shifts
        tokens = {k: v for k, v in doc.items() if k != "text"}
        tokens["tokens"], tokens["word_ids"], tokens["offsets"] = (
            encode_with_token_metadata(self._tokenizer, doc["text"])
        )
        tokens = self._apply_post_token_augs(tokens)
        
        if len(tokens["tokens"]) > 0 and tokens["tokens"][-1] != self.eos_token_id:
            tokens["tokens"].append(self.eos_token_id)
            
        return tokens

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len
        hf_iter = self._get_data_iter()

        while True:
            choice = torch.multinomial(self.probs, 1).item()

            if choice == 0:
                try:
                    sample = next(hf_iter)
                except StopIteration:
                    if not self.infinite:
                        break
                    self._sample_idx = 0
                    hf_iter = self._get_data_iter()
                    sample = next(hf_iter)
                sample_text = {"text": self._text_processor(sample)}
                sample_text = self._apply_augs(sample_text)

                if self.enable_contrastive_mask:
                    assert isinstance(sample_text, list) and len(sample_text) == 2, "Expected paired text for contrastive masking"
                    tokens_1 = {k:v for k,v in sample_text[0].items() if k != "text"}
                    tokens_1["tokens"], tokens_1["word_ids"], tokens_1["offsets"] = (
                        encode_with_token_metadata(self._tokenizer, sample_text[0]["text"])
                    )
                    tokens_2 = {k:v for k,v in sample_text[1].items() if k != "text"}
                    tokens_2["tokens"], tokens_2["word_ids"], tokens_2["offsets"] = (
                        encode_with_token_metadata(self._tokenizer, sample_text[1]["text"])
                    )

                    # Apply post tokenization shifts
                    if self.wordwisecontrastive is not None:
                        n1, mask1 = self.wordwisecontrastive(tokens_1, self.contrastive_pair_idx)
                        n2, mask2 =  self.wordwisecontrastive(tokens_2, self.contrastive_pair_idx)
                        mask2 = [-m for m in mask2]  # Invert the mask for the second sequence to indicate negative pairs
                        assert n1 == n2, f"if using wordwise contrastive, both sequences must have the same number of words (as determined by the contrastive augmentation) to ensure proper alignment of contrastive masks. Please check your augmentation configuration and input data. Are tokens same? {tokens_1 == tokens_2}. n1: {n1}, n2: {n2}. mask1: {mask1}, mask2: {mask2}, is text same? {sample_text[0]['text'] == sample_text[1]['text']}"
                        self.contrastive_mask_buffer.extend(mask1 + mask2)
                        self.contrastive_pair_counter += n1
                        self.contrastive_pair_idx += n1
                    elif len(tokens_1["tokens"]) <= self.contrastive_len_threshold:
                        self.contrastive_mask_buffer.extend([self.contrastive_pair_counter+1] * (len(tokens_1["tokens"])) + [-self.contrastive_pair_counter-1] * (len(tokens_2["tokens"])))
                        self.contrastive_pair_counter += 1
                    else:
                        self.contrastive_mask_buffer.extend([0] * (len(tokens_1["tokens"])+len(tokens_2["tokens"])))
                    tokens_1, tokens_2 = self._apply_post_token_augs([tokens_1, tokens_2])  # Pass both sequences together if your post-token aug needs to consider them jointly

                    new_tokens = tokens_1["tokens"] + tokens_2["tokens"]
                else:
                    # sample_text = self._apply_augs(sample_text)
                    new_tokens = {k: v for k, v in sample_text.items() if k != "text"}
                    new_tokens["tokens"], new_tokens["word_ids"], new_tokens["offsets"] = (
                        encode_with_token_metadata(self._tokenizer, sample_text["text"])
                    )

                    # Apply post tokenization shifts
                    new_tokens = self._apply_post_token_augs(new_tokens)
                    new_tokens = new_tokens["tokens"]
                    self.contrastive_mask_buffer.extend([0] * len(new_tokens))
                self._sample_idx += 1 
            else:
                new_tokens = self._get_injected_tokens(choice - 1)["tokens"]
                self.contrastive_mask_buffer.extend([0] * len(new_tokens))
            
            self._token_buffer.extend(new_tokens)

            while len(self._token_buffer) >= max_buffer_token_len:
                x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                self._token_buffer = self._token_buffer[max_buffer_token_len:]
                
                contrastive_mask = torch.LongTensor(self.contrastive_mask_buffer[:max_buffer_token_len])
                self.contrastive_mask_buffer = self.contrastive_mask_buffer[max_buffer_token_len:]
                # zero out contrastive mask buffer after yielding to prevent leakage across samples
                if len(set(self.contrastive_mask_buffer)) > 1:
                    self.contrastive_pair_counter -= (len(set([abs(x) for x in set(self.contrastive_mask_buffer)]) ) - 1)
                self.contrastive_mask_buffer = [0] * len(self.contrastive_mask_buffer)
                self.contrastive_pair_idx = 0  # Reset the pair index after yielding a batch to prevent overflow

                inputs = {"input": x[:-1]}
                if self.lang_id is not None:
                    inputs["lang_id"] = self.lang_id
                # if self.enable_contrastive_mask:
                # inputs["contrastive_masks"] = self.get_masks(contrastive_mask[:-1], x[:-1])
                # assert len(inputs["input"]) == self.seq_len and (False not in [len(inputs.get("contrastive_masks", [])[i]) == self.seq_len for i in range(MAX_SEQS)]), f"Expected input and contrastive_masks lengths to match seq_len ({self.seq_len}), but got {len(inputs['input'])} and {len(inputs.get('contrastive_masks', []))} respectively."
                yield inputs, x[1:]
    
    def step(self, global_step: int):
        """Propagates the global training step down to the augmentations."""
        for aug in self.aug_callables:
            if hasattr(aug, "step"):
                aug.step(global_step)
                
        for aug in self.post_token_aug_callables:
            if hasattr(aug, "step"):
                aug.step(global_step)

    def get_masks(self, old_mask, tokens) -> torch.BoolTensor:
        # 1. Convert tensor to a native Python list of integers!
        mask_list = old_mask.tolist() if torch.is_tensor(old_mask) else old_mask
        
        # Now sets will hash by numeric value, not memory address
        data_set = set(mask_list)
        seq_len = len(mask_list)
        
        # 2. This will now work perfectly
        valid_positives = sorted({x for x in mask_list if x > 0 and -x in data_set})        
        
        new_masks = []
        for p in valid_positives:
            new_masks.append([x == p for x in mask_list])
            new_masks.append([x == -p for x in mask_list])

        # --- SAFETY CHECK & PADDING ---
        if len(new_masks) > MAX_SEQS:
            new_masks = new_masks[:MAX_SEQS]
            self.contrastive_pair_counter -= (len(new_masks) - MAX_SEQS)//2  

        while len(new_masks) < MAX_SEQS:
            new_masks.append([False] * seq_len)

        return torch.tensor(new_masks, dtype=torch.bool)
    
    def load_state_dict(self, state_dict):
        """Restore the dataset state from a checkpoint."""
        self._token_buffer = state_dict["token_buffer"]
        self.contrastive_mask_buffer = state_dict.get("contrastive_mask_buffer", [])
        
        # --- NEW: Restore the injection counts ---
        if "injection_counts" in state_dict:
            self.injection_counts = state_dict["injection_counts"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            if "data" in state_dict:
                self._data.load_state_dict(state_dict["data"])
            else:
                logger.warning("No dataset state found in checkpoint. Resuming from start.")

    def state_dict(self):
        """Return the current state of the dataset for checkpointing."""
        _state_dict: dict[str, Any] = {
            "token_buffer": self._token_buffer,
            "contrastive_mask_buffer": self.contrastive_mask_buffer,
            # --- NEW: Save the injection counts ---
            "injection_counts": self.injection_counts
        }

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            try:
                _state_dict["data"] = self._data.state_dict()
            except Exception as e:
                logger.warning(f"Could not save iterable dataset state: {e}")

        return _state_dict

class En1En2TranslationValidationDataset(IterableDataset, Stateful):
    """Isolated sentence-translation validation examples for synthetic en1/en2."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        start_idx: int = 6_600_000,
        direction: str = "en1_to_en2",
        vocab_size: int = 65_536,
        eos_token_id: int = 0,
        separator: str = " ",
        data: Any | None = None,
        sentence_tokenizer: Any | None = None,
    ) -> None:
        dataset_name = dataset_name.lower()
        path, dataset_loader, text_processor = _validate_dataset(dataset_name, dataset_path)

        self.dataset_name = dataset_name
        self._data = (
            data
            if data is not None
            else split_dataset_by_node(
                dataset_loader(path, start_idx), dp_rank, dp_world_size
            )
        )
        self._text_processor = text_processor
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.direction = direction
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.separator = separator
        self._punkt_tokenizer = sentence_tokenizer or self._load_punkt_tokenizer()

        if direction not in {"en1_to_en2", "en2_to_en1", "both"}:
            raise ValueError(
                "direction must be one of 'en1_to_en2', 'en2_to_en1', or 'both', "
                f"got {direction!r}."
            )

    @staticmethod
    def _load_punkt_tokenizer():
        try:
            import nltk.data

            return nltk.data.load("tokenizers/punkt/english.pickle")
        except (ImportError, LookupError) as exc:
            raise RuntimeError(
                "en1/en2 translation validation requires nltk and English Punkt "
                "data. Install nltk and run: python -m nltk.downloader punkt punkt_tab."
            ) from exc

    def _sentences(self, text: str) -> list[str]:
        spans = list(self._punkt_tokenizer.span_tokenize(text))
        if not spans:
            stripped_text = text.strip()
            return [stripped_text] if stripped_text else []
        return [
            text[start:end].strip()
            for start, end in spans
            if text[start:end].strip()
        ]

    def _special_token_ids(self) -> set[int]:
        return {
            token_id
            for token_id in (
                getattr(self._tokenizer, "bos_id", None),
                getattr(self._tokenizer, "eos_id", None),
                self.eos_token_id,
            )
            if token_id is not None
        }

    @staticmethod
    def _overlaps(offset: tuple[int, int] | None, span: tuple[int, int]) -> bool:
        if offset is None:
            return False
        start, end = offset
        return (
            start is not None
            and end is not None
            and max(start, span[0]) < min(end, span[1])
        )

    def _make_example(
        self, sentence: str, direction: str
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        target_start = len(sentence) + len(self.separator)
        target_span = (target_start, target_start + len(sentence))
        source_span = (0, len(sentence))
        en2_spans = [target_span] if direction == "en1_to_en2" else [source_span]
        text = f"{sentence}{self.separator}{sentence}"
        special_token_ids = self._special_token_ids()

        tokens, _, offsets = encode_with_token_metadata(self._tokenizer, text)
        shifted_tokens = []
        label_mask = []
        for token_id, offset in zip(tokens, offsets):
            is_special = token_id in special_token_ids
            should_shift = (not is_special) and any(
                self._overlaps(offset, span) for span in en2_spans
            )
            shifted_tokens.append(token_id + self.vocab_size if should_shift else token_id)
            label_mask.append(
                (not is_special) and self._overlaps(offset, target_span)
            )

        max_len = self.seq_len + 1
        shifted_tokens = shifted_tokens[:max_len]
        label_mask = label_mask[:max_len]
        if len(shifted_tokens) < max_len:
            pad_len = max_len - len(shifted_tokens)
            shifted_tokens.extend([self.eos_token_id] * pad_len)
            label_mask.extend([False] * pad_len)

        x = torch.LongTensor(shifted_tokens)
        labels = x[1:].clone()
        mask = torch.tensor(label_mask[1:], dtype=torch.bool)
        labels[~mask] = IGNORE_INDEX
        return {"input": x[:-1]}, labels

    def __iter__(self):
        directions = (
            ("en1_to_en2", "en2_to_en1")
            if self.direction == "both"
            else (self.direction,)
        )

        while True:
            yielded = False
            for sample in iter(self._data):
                text = self._text_processor(sample)
                for sentence in self._sentences(text):
                    for direction in directions:
                        yielded = True
                        yield self._make_example(sentence, direction)

            if not self.infinite or not yielded:
                break

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None

class En1En2TranslationValidationDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "fineweb-edu-ar-en"
        direction: str = "en1_to_en2"
        start_idx: int = 6_600_000
        infinite: bool = True
        eos_token_id: int = 0
        vocab_size: int = 65_536
        separator: str = " "
        validation_steps: int | None = None

    def __init__(self, config: Config, **kwargs):
        if config.validation_steps is not None and (
            config.validation_steps <= 0 and config.validation_steps != -1
        ):
            raise ValueError("validation_steps must be positive, -1, or None")

        self.validation_steps = config.validation_steps
        dataset = En1En2TranslationValidationDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=kwargs["tokenizer"],
            seq_len=kwargs["seq_len"],
            dp_rank=kwargs.get("dp_rank", 0),
            dp_world_size=kwargs.get("dp_world_size", 1),
            infinite=config.infinite,
            start_idx=config.start_idx,
            direction=config.direction,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            separator=config.separator,
        )

        super().__init__(
            dataset,
            dp_rank=kwargs.get("dp_rank", 0),
            dp_world_size=kwargs.get("dp_world_size", 1),
            batch_size=kwargs.get("local_batch_size"),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

class HuggingFaceTextDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        stages: List[dict] = field(default_factory=list)
        infinite: bool = True
        eos_token_id: int = 0
        monolingual_batches: bool = False
        mix_stages: bool = False

    def __init__(self, config: Config, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        seq_len = kwargs.get("seq_len")
        dp_rank = kwargs.get("dp_rank")
        dp_world_size = kwargs.get("dp_world_size")
        stage_idx = kwargs.get("stage_idx", 0)

        datasets = []
        weights = []

        if config.mix_stages and config.stages:
            # --- MIX STAGES LOGIC ---
            # Assume each stage dict has a 'steps' key defining its duration length
            total_steps = sum(stage.get("steps", 1) for stage in config.stages)
            
            for stage in config.stages:
                stage_prob = stage.get("steps", 1) / total_steps
                stage_sources = stage.get("sources", [])
                stage_augs = stage.get("augmentations", [])
                stage_post_augs = stage.get("post_token_augmentations", []) # Grab from config
                
                # Normalize source weights within this specific stage so they sum to 1.0
                raw_src_weights = [src.get("weight", 1.0) for src in stage_sources]
                total_src_weight = sum(raw_src_weights) if sum(raw_src_weights) > 0 else 1.0
                
                for src, raw_w in zip(stage_sources, raw_src_weights):
                    # Combine stage probability with normalized source probability
                    final_weight = stage_prob * (raw_w / total_src_weight)
                    
                    # Handle Tokenizer
                    if src.get("tokenizer", None) is not None:
                        ds_tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=src["tokenizer"])
                        print(f"Using custom tokenizer for dataset {src['name']}: {src['tokenizer']}")
                    else:
                        ds_tokenizer = tokenizer
                        
                    ds = HuggingFaceTextDataset(
                        dataset_name=src["name"],
                        dataset_path=src.get("path"), 
                        tokenizer=ds_tokenizer,
                        seq_len=seq_len,
                        dp_rank=dp_rank,
                        dp_world_size=dp_world_size,
                        infinite=config.infinite,
                        injection_paths=src.get("injection_paths", []),
                        injection_probs=src.get("injection_probs", []),
                        unique_rates=src.get("unique_rates", None),
                        eos_token_id=config.eos_token_id,
                        augmentations=stage_augs, # Stage-level augs passed accurately
                        post_token_augmentations=stage_post_augs, # Pass down the post token augs
                        start_idx=(src.get("start_idx", 0) // (config.num_workers * dp_world_size)),
                        lang_id=src.get("lang_id", None),
                        enable_contrastive_mask=src.get("enable_contrastive_mask", False),
                        contrastive_len_threshold=src.get("contrastive_len_threshold", 256),
                    )
                    datasets.append(ds)
                    weights.append(final_weight)
        else:
            # --- ORIGINAL SEQUENTIAL STAGE LOGIC ---
            if config.stages and stage_idx < len(config.stages):
                current_stage = config.stages[stage_idx]
                current_sources = current_stage.get("sources", [])
                current_augmentations = current_stage.get("augmentations", [])
                current_post_token_augmentations = current_stage.get("post_token_augmentations", []) # Grab from config
            else:
                current_sources = []
                current_augmentations = []
                current_post_token_augmentations = []
                logger.warning(f"No sources found for stage {stage_idx}")

            for src in current_sources:
                if src.get("tokenizer", None) is not None:
                    ds_tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=src["tokenizer"])
                    print(f"Using custom tokenizer for dataset {src['name']}: {src['tokenizer']}")
                else:
                    ds_tokenizer = tokenizer
                if src.get("augmentations", None):
                    logger.warning(f"Dataset {src['name']} has its own augmentations defined in the config. This will override any stage-level augmentations for this specific dataset.")
                    ds_augs = src.get("augmentations", [])
                else:
                    ds_augs = current_augmentations
                if src.get("post_token_augmentations", None):
                    logger.warning(f"Dataset {src['name']} has its own post-token augmentations defined in the config. This will override any stage-level post-token augmentations for this specific dataset.")
                    ds_post_token_augmentations = src.get("post_token_augmentations", [])
                else:
                    ds_post_token_augmentations = current_post_token_augmentations
                ds = HuggingFaceTextDataset(
                    dataset_name=src["name"],
                    dataset_path=src.get("path"), 
                    tokenizer=ds_tokenizer,
                    seq_len=seq_len,
                    dp_rank=dp_rank,
                    dp_world_size=dp_world_size,
                    infinite=config.infinite,
                    injection_paths=src.get("injection_paths", []),
                    injection_probs=src.get("injection_probs", []),
                    unique_rates=src.get("unique_rates", None),
                    eos_token_id=config.eos_token_id,
                    augmentations=ds_augs,
                    post_token_augmentations=ds_post_token_augmentations, # Pass down the post token augs
                    start_idx=(src.get("start_idx", 0) // (config.num_workers * dp_world_size)),
                    lang_id=src.get("lang_id", None),
                    enable_contrastive_mask=src.get("enable_contrastive_mask", False),
                    contrastive_len_threshold=src.get("contrastive_len_threshold", 256),

                )
                datasets.append(ds)
                weights.append(src.get("weight", 1.0))

        # Pass the flattened lists to MixedHuggingFaceDataset
        combined_ds = MixedHuggingFaceDataset(
            datasets, 
            weights, 
            monolingual_batches=config.monolingual_batches, 
            batch_size=kwargs.get("local_batch_size", 1)
        )

        super().__init__(
            combined_ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=kwargs.get("local_batch_size"),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    def step(self, global_step: int):
        """Passes the step from the trainer to the mixed dataset manager."""
        if hasattr(self, "dataset") and hasattr(self.dataset, "step"):
            self.dataset.step(global_step)
