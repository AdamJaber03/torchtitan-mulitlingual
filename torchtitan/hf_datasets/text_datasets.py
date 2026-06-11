import json
import random
import torch
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, List
from functools import partial

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.config import load_output_base_dir
from torchtitan.tools.logging import logger
from torchtitan.hf_datasets.mixed_dataset import MixedHuggingFaceDataset
from torchtitan.hf_datasets.augmentations import AUGMENTATIONS_REGISTRY

from torchtitan.hf_datasets.post_tokenization_augmentations import POST_TOKEN_AUGMENTATIONS_REGISTRY

import random
from datasets import IterableDataset as HFDIterableDataset
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.hf_datasets.post_tokenization_augmentations import WordWiseContrastive

OUTPUT_BASE_DIR = load_output_base_dir()

WORDWISE_CONTRASTIVE_ENABLED = True  # Set to False to disable the word-wise contrastive augmentation
MAX_SEQS = 384

def encode_with_encoding(tokenizer, text):
    encoding = tokenizer.tokenizer.encode(text, add_special_tokens=False)
    # assert len(encoding.ids)-1 == encoding.char_to_token(len(text)-1), f"Encoding length mismatch: {len(encoding.ids)} tokens but expected {encoding.char_to_token(len(text)-1)+1} tokens based on input text length. This may indicate an issue with the tokenizer's encoding of the input text. Please check your tokenizer configuration and ensure it is compatible with the input data. Text: {text[:100]}... Encoding ids: {encoding.ids}"
    # 2. Replicate your wrapper's BOS/EOS logic
    bos = [tokenizer.bos_id] if tokenizer.bos_id is not None else []
    eos = [tokenizer.eos_id] if tokenizer.eos_id is not None else []

    # 3. Store the tokens exactly as your wrapper would have
    tokens = bos + encoding.ids + eos
    return tokens, encoding

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
def _load_dataset(dataset_path: str, start_idx: int, split: str, lang: str | None = None):
    if dataset_path == "karpathy/fineweb-edu-100b-shuffle":
        ld = load_dataset(dataset_path, split=split, streaming=True)
        return ld.skip(start_idx) if start_idx > 0 else ld
    if dataset_path == os.path.join(OUTPUT_BASE_DIR, "fineweb-edu-ar_paired_shards"):
        ld = load_dataset("parquet",data_dir=os.path.join(OUTPUT_BASE_DIR, "fineweb-edu-ar_paired_shards"),split="train",streaming=True)
        ld = ld.shuffle(seed=42, buffer_size=20_000)
        return ld.skip(start_idx) if start_idx > 0 else ld
    if dataset_path == "fineweb2-hq":
        if lang == "ru":
            ld = load_dataset("json", data_dir=os.path.join(OUTPUT_BASE_DIR, "fineweb2_hq/rus_Cyrl/original"), split="train", streaming=True)
        elif lang == "tr2en_1to1map":
            ld = load_dataset("json", data_dir=os.path.join(OUTPUT_BASE_DIR, "fineweb2_hq/rus_Cyrl/translated_1to1map"), split="train", streaming=True)
        ld = ld.skip(start_idx) if start_idx > 0 else ld
        return ld.shuffle(seed=42, buffer_size=20_000)
    if dataset_path == "kaust-generative-ai/fineweb-edu-ar":
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
            ld = load_dataset("json", data_dir=os.path.join(OUTPUT_BASE_DIR, "fineweb_translated/translated"), split="train", streaming=True)
        elif lang == "ar":
            ld = load_dataset("json", data_dir=os.path.join(OUTPUT_BASE_DIR, "fineweb_translated/original"), split="train", streaming=True)
        elif lang == "en":
            ld = load_dataset("json", data_dir=os.path.join(OUTPUT_BASE_DIR, "fineweb_translated/en-original"), split="train", streaming=True)
        elif lang == "tr2en_1to1map":
            ld = load_dataset("json", data_dir=os.path.join(OUTPUT_BASE_DIR, "fineweb_translated/translated_1to1map"), split="train", streaming=True)
        elif lang == "tr2en_1to1map_mixed":
            ld = load_dataset("json", data_dir=os.path.join(OUTPUT_BASE_DIR, "fineweb_translated/translated_1to1map_mixed"), split="train", streaming=True)
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
    return [{"text": en_text}, {"text": ar_text}]

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
    "fineweb-edu-ar-tr2en_1to1map_mixed": DatasetConfig(
        path="kaust-generative-ai/fineweb-edu-ar",
        loader=partial(_load_dataset, split="train", lang="tr2en_1to1map_mixed"),
        sample_processor=_process_c4_text,
    ),
    "fineweb2-hq-ru": DatasetConfig(
        path="fineweb2-hq",
        loader=partial(_load_dataset, split="train", lang="ru"),
        sample_processor=_process_c4_text,
    ),
    "fineweb2-hq-ru-tr2en_1to1map": DatasetConfig(
        path="fineweb2-hq",
        loader=partial(_load_dataset, split="train", lang="tr2en_1to1map"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-ar-paired": DatasetConfig(
        path=os.path.join(OUTPUT_BASE_DIR, "fineweb-edu-ar_paired_shards"),
        loader=partial(_load_dataset, split="train"),
        sample_processor=_process_paired_text,
    ),
    "fineweb-edu-ar-paired-contrastive": DatasetConfig(
        path=os.path.join(OUTPUT_BASE_DIR, "fineweb-edu-ar_paired_shards"),
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
        contrastive_len_threshold: int = 2048,
        max_contrastive_seqs: int = MAX_SEQS,
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
                raise ValueError(f"Augmentation '{aug_name}' not found in AUGMENTATIONS_REGISTRY.")

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
                raise ValueError(f"Post-token augmentation '{aug_name}' not found in POST_TOKEN_AUGMENTATIONS_REGISTRY.")

        self._sample_idx = 0
        self._token_buffer: list[int] = []
        self.contrastive_mask_buffer: list[bool] = []  # Buffer to track which tokens are from contrastive samples
        self.lang_id = lang_id
        self.enable_contrastive_mask = enable_contrastive_mask
        self.contrastive_len_threshold = contrastive_len_threshold
        self.max_contrastive_seqs = max_contrastive_seqs
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
        tokens["tokens"], tokens["encoding"] = encode_with_encoding(self._tokenizer, doc["text"])
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
                sample_p = self._text_processor(sample)
                if type(sample_p) == str:
                    sample_text = {"text": sample_p}
                else:
                    sample_text = sample_p
                if (type(sample_text) == dict and len(sample_text["text"].strip()) == 0) or (type(sample_text) == list and 0 in [len(t["text"].strip()) for t in sample_text]):
                    continue  # Skip empty samples after processing
                sample_text = self._apply_augs(sample_text)

                if self.enable_contrastive_mask:
                    assert isinstance(sample_text, list) and len(sample_text) == 2, "Expected paired text for contrastive masking"
                    tokens_1 = {k:v for k,v in sample_text[0].items()}
                    tokens_1["tokens"], tokens_1["encoding"] = encode_with_encoding(self._tokenizer, sample_text[0]["text"])
                    tokens_2 = {k:v for k,v in sample_text[1].items()}
                    tokens_2["tokens"], tokens_2["encoding"] = encode_with_encoding(self._tokenizer, sample_text[1]["text"])

                    # Apply post tokenization shifts
                    if self.wordwisecontrastive is not None:
                        n1, mask1 = self.wordwisecontrastive(tokens_1, self.contrastive_pair_idx)
                        n2, mask2 =  self.wordwisecontrastive(tokens_2, self.contrastive_pair_idx)
                        mask2 = [-m for m in mask2]  # Invert the mask for the second sequence to indicate negative pairs
                        assert n1 == n2, f"if using wordwise contrastive, both sequences must have the same number of words (as determined by the contrastive augmentation) to ensure proper alignment of contrastive masks. Please check your augmentation configuration and input data. Are tokens same? {tokens_1 == tokens_2}. n1: {n1}, n2: {n2}. mask1: {mask1}, mask2: {mask2}, is text same? {sample_text[0]['text'] == sample_text[1]['text']}"
                        self.contrastive_mask_buffer.extend(mask1 + mask2)
                        self.contrastive_pair_counter += n1
                        self.contrastive_pair_idx += n1
                    # elif len(tokens_1["tokens"]) <= self.contrastive_len_threshold:
                    #     self.contrastive_mask_buffer.extend([self.contrastive_pair_counter+1] * (len(tokens_1["tokens"])) + [-self.contrastive_pair_counter-1] * (len(tokens_2["tokens"])))
                    #     self.contrastive_pair_counter += 1
                    else:
                        self.contrastive_mask_buffer.extend([0] * (len(tokens_1["tokens"])+len(tokens_2["tokens"])))
                    tokens_1, tokens_2 = self._apply_post_token_augs([tokens_1, tokens_2])  # Pass both sequences together if your post-token aug needs to consider them jointly

                    new_tokens = tokens_1["tokens"] + tokens_2["tokens"]
                else:
                    # sample_text = self._apply_augs(sample_text)
                    new_tokens = {k: v for k, v in sample_text.items() if k != "text"}
                    new_tokens["tokens"], new_tokens["encoding"] = encode_with_encoding(self._tokenizer, sample_text["text"])

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
                    self.contrastive_pair_counter -= len(set([abs(x) for x in set(self.contrastive_mask_buffer)]))
                self.contrastive_mask_buffer = [0] * len(self.contrastive_mask_buffer)
                self.contrastive_pair_idx = 0  # Reset the pair index after yielding a batch to prevent overflow

                inputs = {"input": x[:-1]}
                if self.lang_id is not None:
                    inputs["lang_id"] = self.lang_id
                # if self.enable_contrastive_mask:
                inputs["contrastive_masks"] = self.get_masks(contrastive_mask[:-1], x[:-1])
                assert len(inputs["input"]) == self.seq_len and (False not in [len(inputs.get("contrastive_masks", [])[i]) == self.seq_len for i in range(self.max_contrastive_seqs)]), f"Expected input and contrastive_masks lengths to match seq_len ({self.seq_len}), but got {len(inputs['input'])} and {len(inputs.get('contrastive_masks', []))} respectively."
                yield inputs, x[1:]
    
    def step(self, global_step: int):
        """Propagates the global training step down to the augmentations."""
        logger.info(f"Stepping dataset augmentations at global step {global_step}...")
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
        if len(new_masks) > self.max_contrastive_seqs:
            self.contrastive_pair_counter -= (len(new_masks) - self.max_contrastive_seqs) // 2
            new_masks = new_masks[:self.max_contrastive_seqs] 

        while len(new_masks) < self.max_contrastive_seqs:
            new_masks.append([False] * seq_len)

        return torch.tensor(new_masks, dtype=torch.bool)
    
    def load_state_dict(self, state_dict):
        """Restore the dataset state from a checkpoint."""
        self._token_buffer = state_dict["token_buffer"]
        self.contrastive_mask_buffer = state_dict.get("contrastive_mask_buffer", [])

        # Ensure buffers are in sync. contrastive_mask_buffer may be missing from old
        # checkpoints (defaults to []) while token_buffer is non-empty, causing a desync
        # that makes get_masks produce masks shorter than seq_len and fires the assertion.
        # Padding with zeros is correct: 0 means "no contrastive pair" for those tokens.
        token_len = len(self._token_buffer)
        mask_len = len(self.contrastive_mask_buffer)
        if mask_len < token_len:
            logger.warning(
                f"contrastive_mask_buffer ({mask_len}) shorter than token_buffer ({token_len}) "
                "after checkpoint restore — padding with zeros."
            )
            self.contrastive_mask_buffer += [0] * (token_len - mask_len)
        elif mask_len > token_len:
            logger.warning(
                f"contrastive_mask_buffer ({mask_len}) longer than token_buffer ({token_len}) "
                "after checkpoint restore — truncating."
            )
            self.contrastive_mask_buffer = self.contrastive_mask_buffer[:token_len]

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
                        max_contrastive_seqs=src.get("max_contrastive_seqs", MAX_SEQS)
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
                    max_contrastive_seqs=src.get("max_contrastive_seqs", MAX_SEQS)

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