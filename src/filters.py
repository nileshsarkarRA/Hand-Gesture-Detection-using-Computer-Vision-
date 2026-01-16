import time
import math

class OneEuroFilter:
    """
    Implements the One Euro Filter for smooth hand tracking.
    This filter balances low-latency responsiveness during fast movement 
    with high-precision noise reduction during slow movement.
    """
    def __init__(self, freq=60, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        
        # Internal state to track previous values for filtering.
        self.x_prev = 0.0
        self.dx_prev = 0.0
        self.last_time = None

    def alpha(self, cutoff, dt):
        # Calculates the smoothing factor based on cutoff frequency and time step.
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, custom_beta=None):
        # Primary filtering logic for an input value 'x'.
        now = time.time()
        
        if self.last_time is None:
            self.last_time = now
            self.x_prev = x
            self.dx_prev = 0.0
            return x
            
        dt = now - self.last_time
        if dt <= 0: return x
        
        # Estimate the current velocity (derivative).
        dx = (x - self.x_prev) / dt
        
        # Smooth the velocity calculation to reduce jitter.
        edx = self.alpha(self.dcutoff, dt) * dx + (1 - self.alpha(self.dcutoff, dt)) * self.dx_prev
        
        # Adapt the cutoff frequency based on current velocity.
        # High speed -> broader cutoff (less lag).
        # Low speed -> narrow cutoff (less jitter).
        beta = custom_beta if custom_beta is not None else self.beta
        cutoff = self.mincutoff + beta * abs(edx)
        alpha = self.alpha(cutoff, dt)
        
        # Apply the final smoothing formula.
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        
        # Update historians.
        self.x_prev = x_hat
        self.dx_prev = edx
        self.last_time = now
        
        return x_hat
