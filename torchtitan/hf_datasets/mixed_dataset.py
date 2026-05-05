# torchtitan/components/mixed_dataset.py
import torch
from torch.utils.data import IterableDataset
from torch.distributed.checkpoint.stateful import Stateful

class MixedHuggingFaceDataset(IterableDataset, Stateful):
    def __init__(self, datasets, weights, monolingual_batches: bool = False, batch_size: int = 1):
        super().__init__()
        self.datasets = datasets
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.probs = self.weights / self.weights.sum()
        
        # --- NEW: Store batching preferences ---
        self.monolingual_batches = monolingual_batches
        self.batch_size = batch_size

    def step(self, global_step: int):
        """
        Broadcasts the global training step to all underlying datasets.
        """
        for ds in self.datasets:
            if hasattr(ds, "step"):
                ds.step(global_step)

    def __iter__(self):
        # Create active iterators for all sub-datasets
        iters = [iter(ds) for ds in self.datasets]
        while True:
            # Randomly pick which dataset to sample from based on weights
            idx = torch.multinomial(self.probs, 1).item()
            
            if self.monolingual_batches:
                # Yield an entire batch's worth of samples from the chosen dataset
                for _ in range(self.batch_size):
                    try:
                        yield next(iters[idx])
                    except StopIteration:
                        # If a sub-dataset ends mid-batch, break to pick a new dataset
                        break 
            else:
                # Original logic: Yield a single sample
                try:
                    yield next(iters[idx])
                except StopIteration:
                    # If a sub-dataset ends, we continue to others 
                    # (Assumes infinite=True for training)
                    continue

    def state_dict(self):
        return {f"ds_{i}": ds.state_dict() for i, ds in enumerate(self.datasets)}

    def load_state_dict(self, state_dict):
        for i, ds in enumerate(self.datasets):
            if f"ds_{i}" in state_dict:
                ds.load_state_dict(state_dict[f"ds_{i}"])