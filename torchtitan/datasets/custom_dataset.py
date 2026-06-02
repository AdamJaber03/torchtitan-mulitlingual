import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torchtitan.hf_datasets.config import load_output_base_dir

class MixedPretokenizedDataset(IterableDataset):
    def __init__(self, data_paths, mix_rates, seq_len):
        super().__init__()
        self.data_paths = data_paths
        # Normalize the mixing rates (e.g., [0.7, 0.3])
        self.mix_rates = torch.tensor(mix_rates, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __iter__(self):
        # Open memmaps inside __iter__ so multiprocessing workers don't share file handles!
        datasets = [np.memmap(p, dtype=np.uint32, mode='r') for p in self.data_paths]
        lengths = [len(d) for d in datasets]
        
        while True:
            # 1. Pick a dataset based on your rates
            ds_idx = torch.multinomial(self.mix_rates, 1).item()
            ds = datasets[ds_idx]
            
            # 2. Pick a random starting position in that dataset
            max_idx = lengths[ds_idx] - self.seq_len - 1
            start_idx = torch.randint(0, max_idx, (1,)).item()
            
            # 3. Extract the chunk and format for next-token prediction
            chunk = ds[start_idx : start_idx + self.seq_len + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
            
            # TorchTitan expects inputs to be a dict, and labels to be a tensor
            yield {"input": x}, y

def build_mixed_dataloader(batch_size, seq_len):
    OUTPUT_BASE_DIR = load_output_base_dir()
    dataset = MixedPretokenizedDataset(
            data_paths=[os.path.join(OUTPUT_BASE_DIR, "en65kVocab.bin"), os.path.join(OUTPUT_BASE_DIR, "arb_Arab65kVocab.bin")],
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