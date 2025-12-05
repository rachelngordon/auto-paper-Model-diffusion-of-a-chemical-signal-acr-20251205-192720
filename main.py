# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate_diffusion(D, x, dt, t_end, c0):
    dx = x[1] - x[0]
    N = len(x)
    c = c0.copy()
    steps = int(np.ceil(t_end / dt))
    for _ in range(steps):
        c_new = c.copy()
        # interior points
        c_new[1:-1] = c[1:-1] + D * dt / dx**2 * (c[2:] - 2 * c[1:-1] + c[:-2])
        # zero-flux boundaries
        c_new[0] = c[0] + D * dt / dx**2 * (c[1] - c[0])
        c_new[-1] = c[-1] + D * dt / dx**2 * (c[-2] - c[-1])
        c = c_new
    return c

def fwhm(x, y):
    half_max = np.max(y) / 2.0
    # find indices where signal crosses half max
    above = y >= half_max
    if not np.any(above):
        return 0.0
    idx = np.where(above)[0]
    left = idx[0]
    right = idx[-1]
    # linear interpolation for more accurate crossing points
    if left > 0:
        x1, y1 = x[left - 1], y[left - 1]
        x2, y2 = x[left], y[left]
        left_cross = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
    else:
        left_cross = x[left]
    if right < len(x) - 1:
        x1, y1 = x[right], y[right]
        x2, y2 = x[right + 1], y[right + 1]
        right_cross = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
    else:
        right_cross = x[right]
    return right_cross - left_cross

def experiment_1():
    # Parameters
    L = 10.0  # mm, domain from -L/2 to L/2
    N = 500
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]
    D = 0.5  # mm^2/s
    sigma0 = 0.1  # mm
    c0 = np.exp(-x**2 / (2 * sigma0**2))
    c0 /= (sigma0 * np.sqrt(2*np.pi))  # normalize
    times = [0.0, 0.01, 0.05, 0.2]  # seconds
    dt = 0.4 * dx**2 / D  # stability condition
    plt.figure()
    for t in times:
        if t == 0.0:
            c = c0
        else:
            c = simulate_diffusion(D, x, dt, t, c0)
        plt.plot(x, c, label=f't = {t:.2f}s')
    plt.xlabel('Position (mm)')
    plt.ylabel('Concentration')
    plt.title('1D Diffusion of Gaussian Pulse')
    plt.legend()
    plt.tight_layout()
    plt.savefig('concentration_vs_position.png')
    plt.close()

def experiment_2():
    # Parameters
    L = 10.0
    N = 500
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]
    sigma0 = 0.1
    c0 = np.exp(-x**2 / (2 * sigma0**2))
    c0 /= (sigma0 * np.sqrt(2*np.pi))
    D_vals = [0.1, 0.5, 1.0]  # mm^2/s
    t_fixed = 0.2  # seconds
    fwhm_vals = []
    for D in D_vals:
        dt = 0.4 * dx**2 / D
        c = simulate_diffusion(D, x, dt, t_fixed, c0)
        w = fwhm(x, c)
        fwhm_vals.append(w)
    plt.figure()
    plt.plot(D_vals, fwhm_vals, 'o-')
    plt.xlabel('Diffusion Coefficient D (mm^2/s)')
    plt.ylabel('FWHM (mm)')
    plt.title('FWHM vs Diffusion Coefficient after 0.2 s')
    plt.tight_layout()
    plt.savefig('fwhm_vs_diffusion_coefficient.png')
    plt.close()
    # Return the FWHM for D=0.5 as the primary answer
    return fwhm_vals[1]

if __name__ == '__main__':
    experiment_1()
    answer = experiment_2()
    print('Answer:', answer)

