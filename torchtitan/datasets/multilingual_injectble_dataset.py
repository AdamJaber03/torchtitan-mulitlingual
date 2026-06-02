import os
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from torchtitan.hf_datasets.config import load_output_base_dir

class FactInjectingDataset(IterableDataset):
    def __init__(self, bg_paths, bg_mix_rates, fake_paths, fake_injection_rates, seq_len, world_size, rank):
        super().__init__()
        self.bg_paths = bg_paths
        self.fake_paths = fake_paths
        self.seq_len = seq_len
        self.world_size = world_size
        self.rank = rank
        
        # [Rate normalization remains the same...]
        bg_tensor = torch.tensor(bg_mix_rates, dtype=torch.float32)
        self.bg_mix_rates = bg_tensor / bg_tensor.sum()
        
        self.fake_probs = torch.tensor(fake_injection_rates, dtype=torch.float32)
        self.total_fake_prob = self.fake_probs.sum().item()
        
        if self.total_fake_prob >= 1.0:
            raise ValueError("Total fake injection rate must be less than 1.0")
            
        self.bg_prob = 1.0 - self.total_fake_prob
        self.master_rates = torch.cat([torch.tensor([self.bg_prob]), self.fake_probs])

    def __iter__(self):
        # 1. Get Local DataLoader Worker Info (Safe to do in workers)
        worker_info = torch.utils.data.get_worker_info()
        local_num_workers = worker_info.num_workers if worker_info else 1
        local_worker_id = worker_info.id if worker_info else 0

        # 2. Use the injected rank and world_size
        global_num_workers = local_num_workers * self.world_size
        global_worker_id = (self.rank * local_num_workers) + local_worker_id

        # Open memmaps
        bg_datasets = [np.memmap(p, dtype=np.uint32, mode='r') for p in self.bg_paths]
        bg_lengths = [len(d) for d in bg_datasets]
        
        fake_datasets = [np.memmap(p, dtype=np.uint32, mode='r') for p in self.fake_paths]
        
        # 3. Offset pointers
        fake_pointers = []
        for d in fake_datasets:
            start_ptr = (len(d) // global_num_workers) * global_worker_id
            fake_pointers.append(start_ptr)

        while True:
            choice = torch.multinomial(self.master_rates, 1).item()
            
            if choice == 0:
                bg_idx = torch.multinomial(self.bg_mix_rates, 1).item()
                ds = bg_datasets[bg_idx]
                max_idx = bg_lengths[bg_idx] - self.seq_len - 1
                
                start_idx = torch.randint(0, max_idx, (1,)).item()
                chunk = ds[start_idx : start_idx + self.seq_len + 1].astype(np.int64)
                
            else:
                fake_idx = choice - 1
                ds = fake_datasets[fake_idx]
                start_idx = fake_pointers[fake_idx]
                print(f"Worker {global_worker_id} injecting from fake dataset {fake_idx} at position {start_idx}")
                
                if start_idx + self.seq_len + 1 > len(ds):
                    start_idx = 0
                    
                chunk = ds[start_idx : start_idx + self.seq_len + 1].astype(np.int64)
                fake_pointers[fake_idx] = start_idx + self.seq_len
                
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
            
            yield {"input": x}, y

def build_injection_dataloader(batch_size, seq_len, total_training_sequences):
    # Determine distributed context safely in the main process
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    target_exposures = [200]*12 
    fake_rates = [exp / total_training_sequences for exp in target_exposures]
    
    OUTPUT_BASE_DIR = load_output_base_dir()
    dataset = FactInjectingDataset(
        bg_paths=[
            os.path.join(OUTPUT_BASE_DIR, "en.bin"),
            os.path.join(OUTPUT_BASE_DIR, "arb_Arab.bin")
        ],
        bg_mix_rates=[0.5, 0.5],
        fake_paths=[
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/1/arb_Arab_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/2/arb_Arab_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/3/arb_Arab_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/4/arb_Arab_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/5/arb_Arab_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/6/arb_Arab_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/1/en_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/2/en_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/3/en_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/4/en_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/5/en_data.bin"),
            os.path.join(OUTPUT_BASE_DIR, "fictive_entities_gemini/6/en_data.bin")
        ],
        fake_injection_rates=fake_rates,
        seq_len=seq_len,
        world_size=world_size, # <--- Pass here
        rank=rank              # <--- Pass here
    )
    
    return iter(DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=16, 
        pin_memory=True
    ))