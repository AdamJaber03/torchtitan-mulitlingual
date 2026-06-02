import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torchtitan.hf_datasets.config import load_output_base_dir

class AlignedBilingualDataset(IterableDataset):
    def __init__(self, en_bin, ar_bin, mix_rates, seq_len):
        super().__init__()
        self.paths = [en_bin, ar_bin]
        self.mix_rates = torch.tensor(mix_rates)
        self.seq_len = seq_len
        
        # Load the pre-calculated offsets
        print("Loading document offsets...")
        self.en_offsets = np.load(en_bin.replace(".bin", "_offsets.npy"))
        self.ar_offsets = np.load(ar_bin.replace(".bin", "_offsets.npy"))
        
        # We use the minimum common doc count to stay aligned
        self.num_docs = min(len(self.en_offsets), len(self.ar_offsets)) - 1
        print(f"Dataset initialized with {self.num_docs:,} aligned documents.")

    def __iter__(self):
        # Open memmaps inside __iter__ for worker-safety
        en_ds = np.memmap(self.paths[0], dtype=np.uint32, mode='r')
        ar_ds = np.memmap(self.paths[1], dtype=np.uint32, mode='r')
        datasets = [en_ds, ar_ds]
        offsets = [self.en_offsets, self.ar_offsets]

        while True:
            # 1. Pick a SHARED document index (The 'Concept')
            doc_idx = torch.randint(0, self.num_docs, (1,)).item()
            
            # 2. Pick a language for this step
            lang_idx = torch.multinomial(self.mix_rates, 1).item()
            
            # 3. Find boundaries in the chosen language's .bin file
            start_token = offsets[lang_idx][doc_idx]
            end_token = offsets[lang_idx][doc_idx + 1]
            
            # 4. Handle doc length vs. seq_len
            doc_size = end_token - start_token
            
            if doc_size <= self.seq_len:
                # If document is smaller than context window, just take the whole thing
                # pad with EOS tokens
                chunk = datasets[lang_idx][start_token : end_token].astype(np.int64)
                padding = np.full((self.seq_len + 1 - doc_size,), fill_value=0, dtype=np.int64) # Assuming 0 is EOS
                chunk = np.concatenate([chunk, padding])
            else:
                # Pick a random window within the document
                window_start = start_token + torch.randint(0, doc_size - self.seq_len, (1,)).item()
                chunk = datasets[lang_idx][window_start : window_start + self.seq_len + 1].astype(np.int64)
            
            # TorchTitan formatting
            if len(chunk) < 2: continue # skip empty
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
            yield {"input": x}, y

def build_aligned_bilingual_dataloader(batch_size, seq_len):
    OUTPUT_BASE_DIR = load_output_base_dir()
    dataset = AlignedBilingualDataset(
            en_bin=os.path.join(OUTPUT_BASE_DIR, "fineweb_edu_en_ar_pair/en65kVocab.bin"),
            ar_bin=os.path.join(OUTPUT_BASE_DIR, "fineweb_edu_en_ar_pair/ar65kVocab.bin"),
            mix_rates=[1.0, 0.0],                     # 100% English, 0% Arabic
            seq_len=seq_len
    )
    
    # Use multiple background workers to keep the GPUs completely saturated
    return iter(DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=16, 
        pin_memory=True
    ))