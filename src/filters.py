import math
import time

class OneEuroFilter:
    """One Euro Filter for smooth, low-latency signal filtering.
    
    References:
    - Casiez et al. (2012): "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input"
    """
    
    def __init__(self, freq=60, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        """Initialize the filter.
        
        Args:
            freq: Frequency of the source signal in Hz
            mincutoff: Minimum cutoff frequency
            beta: Speed coefficient for adaptive filtering
            dcutoff: Cutoff frequency for derivative
        """
        if freq <= 0:
            raise ValueError("Frequency must be positive")
        if mincutoff <= 0 or beta < 0 or dcutoff <= 0:
            raise ValueError("Parameters must be non-negative")
        
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = 0.0
        self.dx_prev = 0.0
        self.last_time = None

    def alpha(self, cutoff, dt):
        """Calculate exponential smoothing coefficient."""
        if dt <= 0:
            return 0.0
        tau = 1.0 / (2 * math.pi * max(cutoff, 0.0001))  # Avoid division by zero
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x):
        """Apply the One Euro filter to input value."""
        now = time.perf_counter()  # More precise than time.time()
        
        # First call initialization
        if self.last_time is None:
            self.last_time = now
            self.x_prev = x
            self.dx_prev = 0.0
            return float(x)
        
        dt = now - self.last_time
        if dt <= 0:  # Safety check
            return float(self.x_prev)
        
        # Calculate derivative estimate
        dx = (x - self.x_prev) / dt
        alpha_d = self.alpha(self.dcutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        # Adaptive cutoff frequency based on speed
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        
        # Apply exponential smoothing
        alpha_x = self.alpha(cutoff, dt)
        x_hat = alpha_x * x + (1 - alpha_x) * self.x_prev
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.last_time = now
        
        return float(x_hat)
