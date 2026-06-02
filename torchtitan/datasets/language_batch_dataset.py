import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torchtitan.hf_datasets.config import load_output_base_dir

class MixedSBatchPretokenizedDataset(IterableDataset):
    def __init__(self, data_paths, mix_rates, seq_len, batch_size):
        super().__init__()
        self.data_paths = data_paths
        self.mix_rates = torch.tensor(mix_rates, dtype=torch.float32)
        self.seq_len = seq_len
        self.batch_size = batch_size # Dataset now handles batching
        
    def __iter__(self):
        # Open memmaps inside __iter__ so multiprocessing workers don't share file handles
        datasets = [np.memmap(p, dtype=np.uint32, mode='r') for p in self.data_paths]
        lengths = [len(d) for d in datasets]
        
        while True:
            # 1. Pick ONE dataset for the entire batch
            ds_idx = torch.multinomial(self.mix_rates, 1).item()
            ds = datasets[ds_idx]
            max_idx = lengths[ds_idx] - self.seq_len - 1
            
            # 2. Vectorize the random starting positions for speed
            starts = torch.randint(0, max_idx, (self.batch_size,)).tolist()
            
            # 3. Pre-allocate the batch tensors
            x_batch = torch.empty((self.batch_size, self.seq_len), dtype=torch.long)
            y_batch = torch.empty((self.batch_size, self.seq_len), dtype=torch.long)
            
            # 4. Fill the batch
            for i, start_idx in enumerate(starts):
                chunk = ds[start_idx : start_idx + self.seq_len + 1].astype(np.int64)
                x_batch[i] = torch.from_numpy(chunk[:-1])
                y_batch[i] = torch.from_numpy(chunk[1:])
            
            # 5. Create a uniform lang_id tensor for the model routing
            lang_id_tensor = torch.full((self.batch_size,), ds_idx, dtype=torch.long)
            
            # Yield fully formed, homogenous batches
            yield {"input": x_batch, "lang_id": lang_id_tensor}, y_batch

def build_mixed_sbatch_dataloader(batch_size, seq_len):
    OUTPUT_BASE_DIR = load_output_base_dir()
    dataset = MixedSBatchPretokenizedDataset(
        data_paths=[os.path.join(OUTPUT_BASE_DIR, "en.bin"), os.path.join(OUTPUT_BASE_DIR, "arb_Arab.bin")],
        mix_rates=[0.5, 0.5],
        seq_len=seq_len,
        batch_size=batch_size # Pass batch_size here
    )
    
    return iter(DataLoader(
        dataset, 
        batch_size=None, # CRITICAL: Tells DataLoader not to auto-collate
        num_workers=16, 
        pin_memory=True
    ))