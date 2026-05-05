import math

class BaseScheduler:
    """
    Base class that handles the warmup (delay) and cooldown (hold) phases 
    so we don't have to rewrite this logic in every scheduler.
    """
    def __init__(self, delay_steps=0, duration_steps=None):
        self.delay_steps = delay_steps
        self.duration_steps = duration_steps # If None, it runs forever

    def __call__(self, step):
        # 1. Pre-scheduling phase (hold initial value)
        if step <= self.delay_steps:
            return self.get_initial_val()
        
        active_step = step - self.delay_steps
        
        # 2. Post-scheduling phase (hold final value)
        if self.duration_steps is not None and active_step >= self.duration_steps:
            return self.get_final_val()
            
        # 3. Active scheduling phase
        return self._compute_val(active_step)

    def get_initial_val(self): 
        raise NotImplementedError
        
    def get_final_val(self): 
        raise NotImplementedError
        
    def _compute_val(self, active_step): 
        raise NotImplementedError


class CosineDecayScheduler(BaseScheduler):
    def __init__(self, start_val, end_val, duration_steps, delay_steps=0):
        super().__init__(delay_steps=delay_steps, duration_steps=duration_steps)
        self.start_val = start_val
        self.end_val = end_val

    def get_initial_val(self): 
        return self.start_val
        
    def get_final_val(self): 
        return self.end_val

    def _compute_val(self, active_step):
        # Standard cosine annealing formula
        cosine_decay = 0.5 * (1 + math.cos(math.pi * active_step / self.duration_steps))
        return self.end_val + (self.start_val - self.end_val) * cosine_decay


class LinearScheduler(BaseScheduler):
    def __init__(self, start_val, end_val, duration_steps, delay_steps=0):
        super().__init__(delay_steps=delay_steps, duration_steps=duration_steps)
        self.start_val = start_val
        self.end_val = end_val

    def get_initial_val(self): 
        return self.start_val
        
    def get_final_val(self): 
        return self.end_val

    def _compute_val(self, active_step):
        fraction = active_step / self.duration_steps
        return self.start_val + fraction * (self.end_val - self.start_val)


class SinusoidalScheduler(BaseScheduler):
    def __init__(self, min_val, max_val, period_steps, duration_steps=None, delay_steps=0):
        """
        Args:
            min_val: The bottom of the wave (starting point).
            max_val: The peak of the wave.
            period_steps: How many active steps it takes to complete one full wave (min -> max -> min).
            duration_steps: Optional. The total steps the wave will run before freezing.
            delay_steps: How long to hold min_val before the wave starts.
        """
        super().__init__(delay_steps=delay_steps, duration_steps=duration_steps)
        self.min_val = min_val
        self.max_val = max_val
        self.period_steps = period_steps
        
        self.amplitude = (max_val - min_val) / 2.0
        self.midpoint = min_val + self.amplitude

    def get_initial_val(self): 
        return self.min_val
        
    def get_final_val(self):
        # Calculate exactly where the wave is when duration_steps runs out
        return self._compute_val(self.duration_steps)

    def _compute_val(self, active_step):
        # Using a negative cosine wave perfectly starts at min_val, 
        # peaks at max_val at half-period, and returns to min_val at full-period.
        return self.midpoint - self.amplitude * math.cos(2 * math.pi * active_step / self.period_steps)


SCHEDUALER_REGISTRY = {
    "cosine": CosineDecayScheduler,
    "linear": LinearScheduler,
    "sinusoidal": SinusoidalScheduler,
}