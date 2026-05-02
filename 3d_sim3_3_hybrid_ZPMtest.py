#!/usr/bin/env python3
"""
3d_sim3_3_hybrid_ZPMtest - Hybrid CPU/GPU ODIM-U warp-bubble simulation
with QFT-style ZPM diagnostic (mode evolution + Bogoliubov coefficients).
Uses CuPy on GPU for core field evolution, NumPy for I/O and plotting.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Try to use CuPy (GPU); fall back to NumPy if unavailable
try:
    import cupy as cp
    xp = cp
    GPU_ENABLED = True
    print("Using CuPy (GPU) backend.")
except ImportError:
    xp = np
    GPU_ENABLED = False
    print("CuPy not found. Falling back to NumPy (CPU) backend.")

# =========================
# TUNABLE PARAMETERS
# =========================
C       = 2.99792458e8
c_I     = 0.3 * C
beta_I  = 100.0
OMEGA_0 = 2.0 * np.pi  # 1 Hz mode in rad/s

R_bubble = 50.0
v_design = 0.05 * C

bubble_v0      = 5.0 * C
bubble_a_drive = 0.0

front_scale = 8.0
rear_scale  = 0.05
L_front     = 4.0 * R_bubble
L_rear      = 0.2 * R_bubble

ADVECT_FIELD       = True
ADVECTION_STRENGTH = 0.10

sigma_I  = 30.0
I_R_amp  = 5.0e-7

Nx = Ny = Nz = 64
Lx = Ly = Lz = 300.0

CFL = 0.3
Nt  = 4_000_000

D_EM     = 2.25e11
X0_solar = 0.0

VERSION = "3d_sim3_3_hybrid_ZPMtest"

# ---- ZPM / QFT diagnostic controls ----
ZPM_ENABLED  = True
ZPM_N_MODES  = 64          # number of k-modes in 1D spectrum
ZPM_LAMBDA   = 1.0 * OMEGA_0**2  # coupling strength λ (tunable)

# =========================
# Derived grid quantities
# =========================
dx = Lx / Nx
dy = Ly / Ny
dz = Lz / Nz

dt = CFL * dx / c_I

print(f"dx = {dx:.3e} m, dt = {dt:.3e} s (CFL-safe)")
print(f"Total physical time ~ {Nt*dt:.3e} s")

# Coordinate grids (NumPy first, then move to GPU if available)
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
z = np.linspace(-Lz/2, Lz/2, Nz)
X_np, Y_np, Z_np = np.meshgrid(x, y, z, indexing='ij')

if GPU_ENABLED:
    X = cp.asarray(X_np)
    Y = cp.asarray(Y_np)
    Z = cp.asarray(Z_np)
else:
    X, Y, Z = X_np, Y_np, Z_np

# =========================
# Helper functions (GPU/CPU via xp)
# =========================
def laplacian(field, dx, dy, dz):
    f_ip = xp.roll(field, -1, axis=0)
    f_im = xp.roll(field,  1, axis=0)
    f_jp = xp.roll(field, -1, axis=1)
    f_jm = xp.roll(field,  1, axis=1)
    f_kp = xp.roll(field, -1, axis=2)
    f_km = xp.roll(field,  1, axis=2)

    lap_x = (f_ip - 2.0*field + f_im) / dx**2
    lap_y = (f_jp - 2.0*field + f_jm) / dy**2
    lap_z = (f_kp - 2.0*field + f_km) / dz**2
    return lap_x + lap_y + lap_z

def gradient(field, dx, dy, dz):
    fx = (xp.roll(field, -1, axis=0) - xp.roll(field, 1, axis=0)) / (2.0*dx)
    fy = (xp.roll(field, -1, axis=1) - xp.roll(field, 1, axis=1)) / (2.0*dy)
    fz = (xp.roll(field, -1, axis=2) - xp.roll(field, 1, axis=2)) / (2.0*dz)
    return fx, fy, fz

def shaping_function_asymmetric(X, Y, Z, bubble_center, R,
                                front_scale=2.0, rear_scale=0.3,
                                L_front=None, L_rear=None):
    if L_front is None:
        L_front = R
    if L_rear is None:
        L_rear = R

    x0, y0, z0 = bubble_center

    dx_front = xp.clip(X - x0, 0.0,  L_front)
    dx_rear  = xp.clip(X - x0, -L_rear, 0.0)

    r2_front = (dx_front / L_front)**2 + ((Y - y0)/R)**2 + ((Z - z0)/R)**2
    r2_rear  = (dx_rear  / L_rear )**2 + ((Y - y0)/R)**2 + ((Z - z0)/R)**2

    shape_front = xp.exp(-r2_front)
    shape_rear  = xp.exp(-r2_rear)

    shape = front_scale * shape_front + rear_scale * shape_rear
    return shape

# =========================
# Initial conditions
# =========================
bubble_pos_grid = np.array([-140.0, 0.0, 0.0], dtype=float)
bubble_vel_grid = np.array([bubble_v0, 0.0, 0.0], dtype=float)

ship_offset_in_bubble = np.array([0.0, 0.0, 0.0], dtype=float)
ship_pos_grid = bubble_pos_grid + ship_offset_in_bubble
ship_vel_grid = bubble_vel_grid.copy()

bubble_pos_solar = X0_solar + bubble_pos_grid[0]
ship_pos_solar   = X0_solar + ship_pos_grid[0]

I_R_np = np.exp(-(((X_np - bubble_pos_grid[0])**2 +
                   (Y_np - bubble_pos_grid[1])**2 +
                   (Z_np - bubble_pos_grid[2])**2) / (2.0 * sigma_I**2)))
I_R_np *= I_R_amp

if GPU_ENABLED:
    I_R     = cp.asarray(I_R_np)
    I_R_dot = cp.zeros_like(I_R)
    N_x     = cp.zeros_like(I_R)
    N_y     = cp.zeros_like(I_R)
    N_z     = cp.zeros_like(I_R)
else:
    I_R     = I_R_np
    I_R_dot = np.zeros_like(I_R)
    N_x     = np.zeros_like(I_R)
    N_y     = np.zeros_like(I_R)
    N_z     = np.zeros_like(I_R)

# =========================
# Output directories
# =========================
root_dir = os.path.dirname(os.path.abspath(__file__))
version_dir = os.path.join(root_dir, VERSION)
os.makedirs(version_dir, exist_ok=True)

run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(version_dir, f"run_{run_timestamp}")
os.makedirs(run_dir, exist_ok=True)

# =========================
# Telemetry storage (CPU-side)
# =========================
sample_stride = 10_000

times = []
rho_eff_list = []
rho_I_list = []
rho_N_list = []
I_R_rms_list = []  # for ZPM/QFT diagnostic

ship_t = []
ship_x_grid = []
ship_x_solar = []
bubble_x_grid = []
bubble_x_solar = []

slice_indices = np.linspace(0, Nt-1, 5, dtype=int)
slice_index_set = set(slice_indices.tolist())
I_R_slices = []
T00_slices = []

# =========================
# Main time loop
# =========================
for n in range(Nt):
    t = n * dt

    # Bubble motion (CPU scalars)
    bubble_vel_grid[0] += bubble_a_drive * dt
    bubble_pos_grid[0] += bubble_vel_grid[0] * dt

    ship_pos_grid = bubble_pos_grid + ship_offset_in_bubble
    ship_vel_grid = bubble_vel_grid.copy()

    bubble_pos_solar = X0_solar + bubble_pos_grid[0]
    ship_pos_solar   = X0_solar + ship_pos_grid[0]

    # Bubble shape (GPU/CPU via xp)
    shape = shaping_function_asymmetric(
        X, Y, Z, bubble_pos_grid, R_bubble,
        front_scale=front_scale,
        rear_scale=rear_scale,
        L_front=L_front,
        L_rear=L_rear
    )

    N_x = -v_design * shape
    N_y[...] = 0.0
    N_z[...] = 0.0

    # Update I_R
    lap_I = laplacian(I_R, dx, dy, dz)
    I_R_ddot = c_I**2 * lap_I - beta_I * I_R_dot - OMEGA_0**2 * I_R

    if ADVECT_FIELD:
        I_R_x, I_R_y, I_R_z = gradient(I_R, dx, dy, dz)
        adv_term = -(N_x * I_R_x + N_y * I_R_y + N_z * I_R_z)
        I_R_ddot += ADVECTION_STRENGTH * adv_term

    I_R_dot += dt * I_R_ddot
    I_R     += dt * I_R_dot

    # Telemetry (sampled)
    if n % sample_stride == 0 or n == Nt - 1:
        I_R_x, I_R_y, I_R_z = gradient(I_R, dx, dy, dz)
        eps_I = 0.5 * (
            I_R_dot**2 +
            c_I**2 * (I_R_x**2 + I_R_y**2 + I_R_z**2) +
            OMEGA_0**2 * I_R**2
        )
        eps_N = xp.zeros_like(eps_I)

        rho_I = float(xp.sum(eps_I) * dx * dy * dz)
        rho_N = float(xp.sum(eps_N) * dx * dy * dz)
        rho_eff = rho_I + rho_N

        # RMS of I_R as a global measure of metric modulation
        if GPU_ENABLED:
            I_R_rms = float(cp.sqrt(cp.mean(I_R**2)))
        else:
            I_R_rms = float(np.sqrt(np.mean(I_R**2)))

        times.append(t)
        rho_eff_list.append(rho_eff)
        rho_I_list.append(rho_I)
        rho_N_list.append(rho_N)
        I_R_rms_list.append(I_R_rms)

        ship_t.append(t)
        ship_x_grid.append(ship_pos_grid[0])
        ship_x_solar.append(ship_pos_solar)
        bubble_x_grid.append(bubble_pos_grid[0])
        bubble_x_solar.append(bubble_pos_solar)

    # Slices (copy back to CPU)
    if n in slice_index_set:
        if GPU_ENABLED:
            I_snap = cp.asnumpy(I_R)
        else:
            I_snap = I_R.copy()
        I_R_slices.append(I_snap)

        if GPU_ENABLED:
            I_R_x, I_R_y, I_R_z = gradient(I_R, dx, dy, dz)
            eps_I = 0.5 * (
                I_R_dot**2 +
                c_I**2 * (I_R_x**2 + I_R_y**2 + I_R_z**2) +
                OMEGA_0**2 * I_R**2
            )
            T_snap = cp.asnumpy(eps_I / (C**2))
        else:
            I_R_x, I_R_y, I_R_z = gradient(I_R, dx, dy, dz)
            eps_I = 0.5 * (
                I_R_dot**2 +
                c_I**2 * (I_R_x**2 + I_R_y**2 + I_R_z**2) +
                OMEGA_0**2 * I_R**2
            )
            T_snap = (eps_I / (C**2)).copy()
        T00_slices.append(T_snap)

    if n % 100_000 == 0:
        if GPU_ENABLED:
            max_IR = float(cp.max(cp.abs(I_R)))
        else:
            max_IR = float(np.max(np.abs(I_R)))
        print(f"Step {n}/{Nt} | t={t:.3e} s | max|I_R|={max_IR:.3e}")

# =========================
# Convert telemetry to NumPy
# =========================
times       = np.array(times)
rho_eff_arr = np.array(rho_eff_list)
rho_I_arr   = np.array(rho_I_list)
rho_N_arr   = np.array(rho_N_list)
I_R_rms_arr = np.array(I_R_rms_list)

ship_t       = np.array(ship_t)
ship_x_grid  = np.array(ship_x_grid)
ship_x_solar = np.array(ship_x_solar)
bubble_x_grid  = np.array(bubble_x_grid)
bubble_x_solar = np.array(bubble_x_solar)

# Power diagnostics (informational bubble)
if len(times) > 1:
    power_arr = np.gradient(rho_eff_arr, times)
    tail_window = min(100, len(power_arr)//5)
    steady_power = np.mean(power_arr[-tail_window:]) if tail_window > 0 else power_arr[-1]
else:
    power_arr = np.array([0.0])
    steady_power = 0.0

# =========================
# QFT / ZPM DIAGNOSTIC
# =========================
def zpm_qft_diagnostic(times, I_r_rms, c_I, omega0, Lx,
                       lambda_couple, n_modes=64):
    """
    Full QFT-style mode evolution in 1D:
    - Treats a scalar field with time-dependent frequency
      ω_k^2(t) = c_I^2 k^2 + ω0^2 + λ * I_r_rms(t)
    - Evolves mode functions u_k(t) and computes Bogoliubov β_k
    - Returns total vacuum power estimate and spectra.
    """
    if len(times) < 2:
        return 0.0, np.zeros(n_modes), np.zeros(n_modes), np.zeros(n_modes)

    hbar = 1.054571817e-34

    t = times
    dt_local = np.mean(np.diff(t))

    # 1D k-spectrum (periodic box of length Lx)
    # Use symmetric modes around k=0
    k_vals = 2.0 * np.pi * np.fft.fftfreq(n_modes, d=Lx/n_modes)
    k_vals = np.fft.fftshift(k_vals)

    # Precompute ω_k(t)
    omega_kt = np.zeros((len(t), n_modes))
    for i in range(len(t)):
        omega_kt[i, :] = np.sqrt(
            c_I**2 * k_vals**2 + omega0**2 + lambda_couple * I_r_rms[i]
        )

    # Mode evolution: u_k'' + ω_k(t)^2 u_k = 0
    # Represent u_k and u_k_dot as complex arrays
    u_k   = np.zeros((len(t), n_modes), dtype=np.complex128)
    u_dot = np.zeros((len(t), n_modes), dtype=np.complex128)

    # Initial vacuum conditions at t0
    omega0_k = omega_kt[0, :]
    u_k[0, :]   = 1.0 / np.sqrt(2.0 * omega0_k)
    u_dot[0, :] = -1j * omega0_k * u_k[0, :]

    # Simple explicit evolution (leapfrog-like)
    for i in range(1, len(t)):
        # Second derivative: u'' = -ω^2 u
        u_ddot = - (omega_kt[i-1, :]**2) * u_k[i-1, :]
        u_dot[i, :] = u_dot[i-1, :] + dt_local * u_ddot
        u_k[i, :]   = u_k[i-1, :] + dt_local * u_dot[i, :]

    # At final time, extract Bogoliubov coefficients relative to instantaneous ω_k(T)
    omega_T = omega_kt[-1, :]
    u_T     = u_k[-1, :]
    u_dot_T = u_dot[-1, :]

    alpha_k = 0.5 * (
        np.sqrt(2.0*omega_T) * u_T +
        1j * u_dot_T / np.sqrt(2.0*omega_T)
    )
    beta_k  = 0.5 * (
        np.sqrt(2.0*omega_T) * u_T -
        1j * u_dot_T / np.sqrt(2.0*omega_T)
    )

    N_k = np.abs(beta_k)**2  # particle number per mode

    # Energy per mode: E_k = ħ ω_T N_k
    E_k = hbar * omega_T * N_k

    # Total energy created over the run
    E_total = np.sum(E_k)

    # Average power over the simulation time
    T_total = t[-1] - t[0] if t[-1] > t[0] else 1.0
    P_vac = E_total / T_total

    return P_vac, N_k, k_vals, E_k

if ZPM_ENABLED:
    P_vac, N_k, k_vals, E_k = zpm_qft_diagnostic(
        times, I_R_rms_arr, c_I, OMEGA_0, Lx,
        lambda_couple=ZPM_LAMBDA,
        n_modes=ZPM_N_MODES
    )
else:
    P_vac = 0.0
    N_k   = np.zeros(ZPM_N_MODES)
    k_vals = np.zeros(ZPM_N_MODES)
    E_k   = np.zeros(ZPM_N_MODES)

# =========================
# Console summary
# =========================
print(f"Simulation time: {times[-1]:.6e} s")
print(f"rho_eff(t): min={np.min(rho_eff_arr):.3e}, max={np.max(rho_eff_arr):.3e}")
print("T00_info is POSITIVE" if np.all(rho_eff_arr > 0.0) else "T00_info has sign changes")
print(f"Final bubble x (grid)  : {bubble_x_grid[-1]:.6e} m")
print(f"Final ship x (grid)    : {ship_x_grid[-1]:.6e} m")
print(f"Final ship x (solar)   : {ship_x_solar[-1]:.6e} m (Earth at 0, Mars at {D_EM:.3e} m)")
print(f"Final total info energy: {rho_eff_arr[-1]:.3e} J")
print(f"Final inst. power (W)  : {power_arr[-1]:.3e}")
print(f"Steady-tail power (W)  : {steady_power:.3e}")
print(f"ZPM/QFT vacuum power (W) estimate: {P_vac:.3e}")

if P_vac > steady_power:
    print(">>> ZPM diagnostic: vacuum-mode excitation COULD, in principle, cover the bubble's steady-state power in this toy model.")
else:
    print(">>> ZPM diagnostic: vacuum-mode excitation is INSUFFICIENT to power the bubble in this toy model.")

# =========================
# Save telemetry
# =========================
with open(os.path.join(run_dir, "telemetry.txt"), "w") as f:
    f.write("# t(s)   rho_eff(J)   power(W)   rho_I(J)   rho_N(J)   I_R_rms\n")
    for ti, re, pw, rI, rN, ir in zip(times, rho_eff_arr, power_arr, rho_I_arr, rho_N_arr, I_R_rms_arr):
        f.write(f"{ti:.9e}  {re:.6e}  {pw:.6e}  {rI:.6e}  {rN:.6e}  {ir:.6e}\n")

with open(os.path.join(run_dir, "ship_trajectory.txt"), "w") as f:
    f.write("# t(s)   x_grid(m)   x_solar(m)   bubble_x_grid(m)   bubble_x_solar(m)\n")
    for ti, xs_g, xs_s, xb_g, xb_s in zip(
        ship_t, ship_x_grid, ship_x_solar, bubble_x_grid, bubble_x_solar
    ):
        f.write(
            f"{ti:.9e}  {xs_g:.6e}  {xs_s:.6e}  {xb_g:.6e}  {xb_s:.6e}\n"
        )

# Save ZPM spectra
if ZPM_ENABLED:
    with open(os.path.join(run_dir, "zpm_spectrum.txt"), "w") as f:
        f.write("# k(1/m)   N_k   E_k(J)\n")
        for kv, Nk, Ek in zip(k_vals, N_k, E_k):
            f.write(f"{kv:.9e}  {Nk:.6e}  {Ek:.6e}\n")

# =========================
# Plots
# =========================
# 1) rho_eff and power vs time
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(times, rho_eff_arr, 'b-')
plt.xlabel("t (s)")
plt.ylabel("rho_eff (J)")
plt.title("Space-integrated informational energy")

plt.subplot(1,2,2)
plt.plot(times, power_arr, 'b-', label="instantaneous")
plt.axhline(steady_power, color='r', linestyle='--', label="steady-tail avg")
plt.xlabel("t (s)")
plt.ylabel("Power (W)")
plt.title("Effective power draw")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "energy_and_power.png"), dpi=150)
plt.close()

# 2) Ship and bubble x-position in grid frame
plt.figure(figsize=(6,4))
plt.plot(ship_t, ship_x_grid, 'k-', label="ship x (grid)")
plt.plot(ship_t, bubble_x_grid, 'r--', label="bubble x (grid)")
plt.xlabel("t (s)")
plt.ylabel("x (m)")
plt.title("Ship & bubble x-position (grid frame)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "ship_bubble_x_grid.png"), dpi=150)
plt.close()

# 3) Ship x-position in solar-system frame
plt.figure(figsize=(6,4))
plt.plot(ship_t, ship_x_solar, 'b-')
plt.axhline(0.0, color='g', linestyle='--', label="Earth (0)")
plt.axhline(D_EM, color='r', linestyle='--', label="Mars (~D_EM)")
plt.xlabel("t (s)")
plt.ylabel("x_solar (m)")
plt.title("Ship position in solar-system frame")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "ship_x_solar.png"), dpi=150)
plt.close()

# 4) I_R slices
k_mid = Nz // 2
plt.figure(figsize=(15,3))
for i, (idx, I_snap) in enumerate(zip(slice_indices, I_R_slices)):
    plt.subplot(1,5,i+1)
    plt.imshow(
        I_snap[:, :, k_mid],
        extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        cmap='seismic',
        aspect='equal'
    )
    plt.colorbar()
    plt.title(f"I_R slice, t≈{idx*dt:.2e}s")
    plt.xlabel("x (m)")
    if i == 0:
        plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "I_R_slices.png"), dpi=150)
plt.close()

# 5) T00 slices
plt.figure(figsize=(15,3))
for i, (idx, T_snap) in enumerate(zip(slice_indices, T00_slices)):
    plt.subplot(1,5,i+1)
    plt.imshow(
        T_snap[:, :, k_mid],
        extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        cmap='viridis',
        aspect='equal'
    )
    plt.colorbar()
    plt.title(f"T00 slice, t≈{idx*dt:.2e}s")
    plt.xlabel("x (m)")
    if i == 0:
        plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "T00_slices.png"), dpi=150)
plt.close()

# 6) ZPM mode spectrum plot
if ZPM_ENABLED:
    plt.figure(figsize=(6,4))
    plt.plot(k_vals, N_k, 'b-')
    plt.xlabel("k (1/m)")
    plt.ylabel("N_k")
    plt.title("ZPM/QFT mode occupation spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "zpm_spectrum.png"), dpi=150)
    plt.close()

print(f"Outputs saved under: {run_dir}")
