import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

class MixedPretokenizedPreloadDataset(IterableDataset):
    def __init__(self, data_paths, mix_rates, seq_len):
        super().__init__()
        self.mix_rates = torch.tensor(mix_rates, dtype=torch.float32)
        self.seq_len = seq_len
        
        print("Preloading datasets into RAM. This may take a moment...")
        # 1. Load ONCE in the main process using np.fromfile
        # 2. Use np.uint16 to save 50% RAM space for your 32k vocab
        self.datasets = [np.fromfile(p, dtype=np.uint32) for p in data_paths]
        self.lengths = [len(d) for d in self.datasets]
        print(f"Preloading complete! Loaded {sum(self.lengths):,} total tokens.")
        
    def __iter__(self):
        # Workers do no loading here! They just read the preloaded RAM.
        while True:
            # 1. Pick a dataset based on your rates
            ds_idx = torch.multinomial(self.mix_rates, 1).item()
            ds = self.datasets[ds_idx]
            
            # 2. Pick a random starting position in that dataset
            max_idx = self.lengths[ds_idx] - self.seq_len - 1
            start_idx = torch.randint(0, max_idx, (1,)).item()
            
            # 3. Extract the chunk and cast to 64-bit for the model
            chunk = ds[start_idx : start_idx + self.seq_len + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
            
            yield {"input": x}, y

def build_mixed_preload_dataloader(batch_size, seq_len):
    dataset = MixedPretokenizedPreloadDataset(
            data_paths=[r"/home/adamga/leshemg/adamga/data/en.bin", r"/home/adamga/leshemg/adamga/data/arb_Arab.bin"], 
        mix_rates=[0.5, 0.5],                     
        seq_len=seq_len
    )
    
    return iter(DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=16, 
        pin_memory=True
    ))