import json
import random
import torch
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
from torchtitan.tools.logging import logger
from torchtitan.hf_datasets.mixed_dataset import MixedHuggingFaceDataset
from torchtitan.hf_datasets.augmentations import AUGMENTATIONS_REGISTRY

import random
from datasets import IterableDataset as HFDIterableDataset
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer


MAX_SEQS = 32

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
def _load_c4_dataset(dataset_path: str, start_idx: int, split: str, lang: str | None = None):
    if dataset_path == "karpathy/fineweb-edu-100b-shuffle":
        ld = load_dataset(dataset_path, split=split, streaming=True)
        return ld.skip(start_idx) if start_idx > 0 else ld
    if dataset_path == "/home/adamga/leshemg/adamga/data/fineweb-edu-ar_paired_shards":
        ld = load_dataset("parquet",data_dir="/home/adamga/leshemg/adamga/data/fineweb-edu-ar_paired_shards",split="train",streaming=True)
        ld = ld.shuffle(seed=42, buffer_size=20_000)
        return ld.skip(start_idx) if start_idx > 0 else ld
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
        ld = load_dataset(dataset_path, lang, split=split, streaming=True)
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
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-100b-shuffle": DatasetConfig(
        path="karpathy/fineweb-edu-100b-shuffle",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-ar-ar": DatasetConfig(
        path="kaust-generative-ai/fineweb-edu-ar",
        loader=partial(_load_c4_dataset, split="train", lang="ar"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-ar-en": DatasetConfig(
        path="kaust-generative-ai/fineweb-edu-ar",
        loader=partial(_load_c4_dataset, split="train", lang="en"),
        sample_processor=_process_c4_text,
    ),
    "fineweb-edu-ar-paired": DatasetConfig(
        path="/home/adamga/leshemg/adamga/data/fineweb-edu-ar_paired_shards",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_paired_text,
    ),
    "fineweb-edu-ar-paired-contrastive": DatasetConfig(
        path="/home/adamga/leshemg/adamga/data/fineweb-edu-ar_paired_shards",
        loader=partial(_load_c4_dataset, split="train"),
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
        # Setup Augmentations
        self.aug_callables = []
        augmentations = augmentations or []
        for aug_cfg in augmentations:
            aug_name = aug_cfg.get("name")
            if aug_name in AUGMENTATIONS_REGISTRY:
                aug_kwargs = {k: v for k, v in aug_cfg.items() if k != "name"}
                aug_instance = AUGMENTATIONS_REGISTRY[aug_name](aug_cfg)
                self.aug_callables.append(aug_instance)
            else:
                logger.warning(f"Augmentation '{aug_name}' not found in AUGMENTATIONS_REGISTRY.")

        self._sample_idx = 0
        self._token_buffer: list[int] = []
        self.contrastive_mask_buffer: list[bool] = []  # Buffer to track which tokens are from contrastive samples
        self.lang_id = lang_id
        self.enable_contrastive_mask = enable_contrastive_mask
        self.contrastive_len_threshold = contrastive_len_threshold
        self.contrastive_pair_counter = torch.zeros(1, dtype=torch.int64).share_memory_()  # Counter to track active contrastive pairs

    def _apply_augs(self, text: str) -> str:
        for aug_fn in self.aug_callables:
            text = aug_fn(text, dataset_name=self.dataset_name)
        return text

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

        doc = self._apply_augs(doc)
        print(f"Sample injected doc (truncated to 200 chars): {doc[:200]}...")  # Log the injected document for debugging
        tokens = self._tokenizer.encode(doc, add_bos=True, add_eos=True)
        
        if len(tokens) > 0 and tokens[-1] != self.eos_token_id:
            tokens.append(self.eos_token_id)
            
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
                sample_text = self._text_processor(sample)
                if self.enable_contrastive_mask:
                    assert isinstance(sample_text, list) and len(sample_text) == 2, "Expected paired text for contrastive masking"
                    tokens_1 = self._tokenizer.encode(sample_text[0], add_bos=True, add_eos=True)
                    tokens_2 = self._tokenizer.encode(sample_text[1], add_bos=True, add_eos=True)
                    if len(tokens_1) <= self.contrastive_len_threshold:
                        self.contrastive_mask_buffer.extend([False] * (len(tokens_1)-1) + [True] + [False] * (len(tokens_2)-1) + [True])
                        self.contrastive_pair_counter += 1
                    else:
                        self.contrastive_mask_buffer.extend([False] * (len(tokens_1)+len(tokens_2)))
                    new_tokens = tokens_1 + tokens_2
                else:
                    sample_text = self._apply_augs(sample_text)
                    new_tokens = self._tokenizer.encode(sample_text, add_bos=True, add_eos=True)
                    self.contrastive_mask_buffer.extend([False] * len(new_tokens))
                self._sample_idx += 1 
            else:
                new_tokens = self._get_injected_tokens(choice - 1)
                self.contrastive_mask_buffer.extend([False] * len(new_tokens))
            
            self._token_buffer.extend(new_tokens)

            while len(self._token_buffer) >= max_buffer_token_len:
                x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                self._token_buffer = self._token_buffer[max_buffer_token_len:]
                
                contrastive_mask = torch.BoolTensor(self.contrastive_mask_buffer[:max_buffer_token_len])
                self.contrastive_mask_buffer = self.contrastive_mask_buffer[max_buffer_token_len:]
                # zero out contrastive mask buffer after yielding to prevent leakage across samples
                if True in self.contrastive_mask_buffer:
                    self.contrastive_pair_counter -= 1
                self.contrastive_mask_buffer = [False] * len(self.contrastive_mask_buffer)

                inputs = {"input": x[:-1]}
                if self.lang_id is not None:
                    inputs["lang_id"] = self.lang_id
                if self.enable_contrastive_mask:
                    inputs["contrastive_masks"] = self.get_masks(contrastive_mask[:-1], x[:-1])
                    assert len(inputs["input"]) == self.seq_len and (False not in [len(inputs.get("contrastive_masks", [])[i]) == self.seq_len for i in range(MAX_SEQS)]), f"Expected input and contrastive_masks lengths to match seq_len ({self.seq_len}), but got {len(inputs['input'])} and {len(inputs.get('contrastive_masks', []))} respectively."
                yield inputs, x[1:]
    
    def get_masks(self, old_mask, tokens) -> torch.BoolTensor:
        current_start = 0
        new_masks = []
        seq_len = len(tokens)
        for i, token in enumerate(tokens):
            # A sequence boundary is reached when we see an EOS token
            # (or if we hit the very end of the document)
            if token == self.eos_token_id or i == seq_len - 1:
                
                # Did your old mask flag this boundary as a contrastive pair?
                if old_mask[i]:
                    # 1. Create a blank boolean mask for this specific sequence
                    seq_mask = [False] * seq_len
                    
                    # 2. Fill it with True from the start of the sentence up to the EOS
                    for j in range(current_start, i + 1):
                        seq_mask[j] = True
                        
                    # 3. Add it to our list of active masks
                    new_masks.append(seq_mask)
                
                # The next sequence starts immediately after this boundary
                current_start = i + 1

        # --- SAFETY CHECK & PADDING ---
        # If a document somehow exceeds your MAX_SEQS, truncate it to prevent batching crashes
        if len(new_masks) > MAX_SEQS:
            new_masks = new_masks[:MAX_SEQS]
            self.contrastive_pair_counter -= (len(new_masks) - MAX_SEQS)  # Adjust the counter for any truncated pairs

        # Pad with completely empty sequences up to MAX_SEQS
        # This guarantees the Dataloader will perfectly stack them into [Batch, MAX_SEQS, SeqLen]
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
            else:
                current_sources = []
                current_augmentations = []
                logger.warning(f"No sources found for stage {stage_idx}")

            for src in current_sources:
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
                    augmentations=current_augmentations,
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