import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torchtitan.tools.logging import logger
from torchtitan.hf_datasets.config import load_output_base_dir

class AlignedPackedDataset(IterableDataset):
    def __init__(self, en_bin, ar_bin, mix_rates, seq_len, eos_id):
        super().__init__()
        self.paths = [en_bin, ar_bin]
        self.mix_rates = torch.tensor(mix_rates, dtype=torch.float32)
        self.seq_len = seq_len
        self.eos_id = eos_id
        
        logger.info(f"Loading document offsets for aligned packing...")
        self.en_offsets = np.load(en_bin.replace(".bin", "_offsets.npy"))
        self.ar_offsets = np.load(ar_bin.replace(".bin", "_offsets.npy"))
        
        # Use the minimum common document count to ensure 1:1 pairing exists
        self.num_docs = min(len(self.en_offsets), len(self.ar_offsets)) - 1
        logger.info(f"Dataset initialized with {self.num_docs:,} aligned document pairs.")

    def __iter__(self):
        # Open memmaps inside __iter__ so multiprocessing workers don't share file handles.
        # This prevents IO crashes on Slurm nodes.
        en_ds = np.memmap(self.paths[0], dtype=np.uint32, mode='r')
        ar_ds = np.memmap(self.paths[1], dtype=np.uint32, mode='r')
        datasets = [en_ds, ar_ds]
        offsets = [self.en_offsets, self.ar_offsets]

        while True:
            packed_tokens = []
            
            # We need seq_len + 1 to yield x (input) and y (shifted label)
            target_len = self.seq_len + 1 
            
            while len(packed_tokens) < target_len:
                # 1. Pick the shared concept (document index)
                doc_idx = torch.randint(0, self.num_docs, (1,)).item()
                
                # 2. Pick the language to surface this concept in
                lang_idx = torch.multinomial(self.mix_rates, 1).item()
                
                # 3. Locate the document boundaries
                start = offsets[lang_idx][doc_idx]
                end = offsets[lang_idx][doc_idx + 1]
                
                if start >= end:
                    continue  # Skip empty documents
                    
                # 4. Extract the document and ensure it ends with an EOS marker
                doc = datasets[lang_idx][start:end].tolist()
                if not doc or doc[-1] != self.eos_id:
                    doc.append(self.eos_id)
                
                # 5. Handle packing and random slicing
                remaining_space = target_len - len(packed_tokens)
                
                if len(doc) > remaining_space:
                    # Document is too long to fit. Pick a random slice to prevent prefix bias.
                    max_start_idx = len(doc) - remaining_space
                    random_start = torch.randint(0, max_start_idx + 1, (1,)).item()
                    packed_tokens.extend(doc[random_start : random_start + remaining_space])
                else:
                    # Document fits entirely in the remaining space
                    packed_tokens.extend(doc)

            # 6. Convert to tensors and slice into TorchTitan's required format
            chunk = torch.tensor(packed_tokens, dtype=torch.long)
            x = chunk[:-1]
            y = chunk[1:]

            # The trainer will automatically calculate varlen metadata from the eos_id inside 'x'
            yield {"input": x}, y


def build_aligned_multilingual_packed_dataloader(batch_size: int, seq_len: int):
    """
    Builder function to be injected into torchtitan/train.py
    """
    OUTPUT_BASE_DIR = load_output_base_dir()
    en_bin = os.path.join(OUTPUT_BASE_DIR, "fineweb_edu_en_ar_pair/en65kVocab.bin")
    ar_bin = os.path.join(OUTPUT_BASE_DIR, "fineweb_edu_en_ar_pair/ar65kVocab.bin")

    # Check if files exist to fail fast
    for path in [en_bin, ar_bin, en_bin.replace(".bin", "_offsets.npy"), ar_bin.replace(".bin", "_offsets.npy")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset file: {path}")

    # Set mixing rates (100% English, 0% Arabic)
    mix_rates = [1.0, 0.0]

    # Set this to the actual token ID of <|endoftext|> in your custom 32k BPE
    EOS_ID = 0  

    dataset = AlignedPackedDataset(
        en_bin=en_bin, 
        ar_bin=ar_bin, 
        mix_rates=mix_rates, 
        seq_len=seq_len,
        eos_id=EOS_ID
    )

    return iter(DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=16,       # Keeps your GPUs saturated
        pin_memory=True,      # Crucial for fast CPU -> GPU transfer
        prefetch_factor=2     # Queues up batches in the background
    ))