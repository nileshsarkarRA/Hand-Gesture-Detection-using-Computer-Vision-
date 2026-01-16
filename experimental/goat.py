import taichi as ti
import math

ti.init(arch=ti.gpu)

# --- CONFIGURATION (4K Ready Logic) ---
WIDTH = 1280 
HEIGHT = 720
pixels = ti.Vector.field(3, dtype=float, shape=(WIDTH, HEIGHT))

# --- PHYSICS CONSTANTS ---
RS = 1.0             # Schwarzschild Radius
DISK_INNER = 2.6     # Just outside the shadow
DISK_OUTER = 14.0    # Huge disk
DISK_HEIGHT = 0.3    # Thickness of the gas cloud
CAM_DIST = 18.0      

@ti.func
def fract(x):
    return x - ti.floor(x)

@ti.func
def random3(p):
    # Procedural pseudo-random noise for gas texture
    return fract(ti.sin(ti.Vector([
        p.dot(ti.Vector([127.1, 311.7])),
        p.dot(ti.Vector([269.5, 183.3])),
        p.dot(ti.Vector([419.2, 371.9]))
    ])) * 43758.5453)

@ti.func
def noise(p):
    # Simple value noise for clouds
    i = ti.floor(p)
    f = fract(p)
    f = f * f * (3.0 - 2.0 * f)
    return ti.mix(
        ti.mix(random3(i + ti.Vector([0,0])).x, random3(i + ti.Vector([1,0])).x, f.x),
        ti.mix(random3(i + ti.Vector([0,1])).x, random3(i + ti.Vector([1,1])).x, f.x),
        f.y
    )

@ti.func
def get_disk_density(pos):
    """
    Returns the density of the gas at a specific 3D point in space.
    """
    density = 0.0
    r = pos.norm()
    
    # 1. Vertical fade (Exponential falloff from plane Y=0)
    # The disk is thin near the hole, thicker further out
    y_dist = ti.abs(pos.y)
    if y_dist < DISK_HEIGHT * (1.0 + r*0.05): 
        if r > DISK_INNER and r < DISK_OUTER:
            
            # 2. Radial fade (Brightest near ISCO)
            radial_falloff = ti.exp(-(r - DISK_INNER) * 0.8)
            
            # 3. Procedural Noise (The "Streaks")
            # We map angle/radius to texture coordinates
            angle = ti.atan2(pos.z, pos.x)
            noise_val = noise(ti.Vector([r * 2.0, angle * 8.0]))
            
            density = radial_falloff * (0.5 + 0.5 * noise_val)
            
            # Soft edges
            density *= (1.0 - y_dist / (DISK_HEIGHT * (1.0 + r*0.05)))
            
    return density

@ti.func
def blackbody_color(density, r, Doppler):
    """
    Maps density to hot plasma colors with Doppler shift.
    """
    # Base temperature color (Kelvin-ish mapping)
    # High density = White/Blue, Low density = Orange/Red
    
    # Apply Doppler Shift (Blue/Brighter if > 1, Red/Dimmer if < 1)
    effective_density = density * Doppler
    
    col = ti.Vector([0.0, 0.0, 0.0])
    
    if effective_density > 0.8: # Core hot
        col = ti.Vector([1.0, 0.9, 0.8]) * 2.0 
    elif effective_density > 0.4: # Mid plasma
        col = ti.Vector([1.0, 0.6, 0.2]) 
    else: # Cool edge
        col = ti.Vector([0.8, 0.1, 0.05])
        
    return col * effective_density * 1.5

@ti.func
def sky_color(dir):
    # A simple starfield background
    val = noise(dir.xy * 50.0)
    star = 0.0
    if val > 0.95:
        star = (val - 0.95) * 20.0
    return ti.Vector([0.02, 0.02, 0.05]) + ti.Vector([star, star, star])

@ti.kernel
def render(t: float):
    # Camera Logic (Orbiting)
    cam_x = CAM_DIST * ti.sin(t * 0.2)
    cam_z = CAM_DIST * ti.cos(t * 0.2)
    cam_y = 4.0 * ti.cos(t * 0.1) # High angle to see the disk structure
    
    origin = ti.Vector([cam_x, cam_y, cam_z])
    target = ti.Vector([0.0, 0.0, 0.0])
    
    fwd = (target - origin).normalized()
    right = ti.Vector([0.0, 1.0, 0.0]).cross(fwd).normalized()
    up = fwd.cross(right).normalized()
    
    aspect = WIDTH / HEIGHT

    for i, j in pixels:
        u = (i / WIDTH - 0.5) * 2.0 * aspect
        v = (j / HEIGHT - 0.5) * 2.0
        
        ray_dir = (fwd + u * 0.9 * right + v * 0.9 * up).normalized()
        ray_pos = origin
        vel = ray_dir
        
        color_acc = ti.Vector([0.0, 0.0, 0.0])
        transmittance = 1.0 # Light remaining (starts at 100%)
        
        # RAY MARCHING
        # We need a fixed step size for volumetric integration stability
        step_size = 0.15 
        
        has_escaped = 0
        hit_horizon = 0
        
        for s in range(400):
            # 1. Physics Update (Geodesic Bending)
            r_sq = ray_pos.norm_sqr()
            r = ti.sqrt(r_sq)
            
            # Gravity: Binet approximation for light
            h = ray_pos.cross(vel)
            accel = -1.5 * RS * h.norm_sqr() * ray_pos / (r**5)
            
            # Update Position/Velocity
            vel += accel * step_size
            vel = vel.normalized() # Light is always c
            ray_pos += vel * step_size
            
            # 2. Event Horizon Check
            if r < RS:
                hit_horizon = 1
                break
                
            # 3. Volumetric Integration (The Glow)
            # Only sample if we are close to the disk plane to save perf
            if ti.abs(ray_pos.y) < 2.0 and r < DISK_OUTER and r > DISK_INNER * 0.9:
                dens = get_disk_density(ray_pos)
                
                if dens > 0.01:
                    # Doppler Calculation
                    # Disk velocity vector (approx circular orbit)
                    v_disk = ti.Vector([-ray_pos.z, 0.0, ray_pos.x]).normalized()
                    # Beaming factor: (1 / gamma * (1 - beta * cos(theta))) approx
                    # Simplified: Boost brightness if moving towards ray
                    doppler = 1.0 + v_disk.dot(vel) * 0.6
                    
                    emission = blackbody_color(dens, r, doppler)
                    
                    # Accumulate Light
                    # Beer-Lambert Law approximation
                    absorption = dens * 0.2 * step_size
                    color_acc += emission * transmittance * step_size
                    transmittance *= (1.0 - absorption)
                    
                    if transmittance < 0.01:
                        break # Fully occluded
            
            # 4. Escape Check
            if r > 30.0:
                has_escaped = 1
                break
        
        # Final Compositing
        if hit_horizon == 1:
            # The black hole itself
            pixels[i, j] = color_acc 
        elif has_escaped == 1:
            # Background stars + accumulated disk glow
            bg = sky_color(vel)
            pixels[i, j] = color_acc + bg * transmittance
        else:
            pixels[i, j] = color_acc

# --- DISPLAY ---
gui = ti.GUI("Volumetric Black Hole", res=(WIDTH, HEIGHT))
t = 0.0
while gui.running:
    render(t)
    gui.set_image(pixels)
    gui.show()
    t += 0.05