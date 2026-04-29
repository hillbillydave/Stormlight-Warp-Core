import numpy as np
import matplotlib.pyplot as plt

# --- Parameters from the paper ---
tau0 = 1.0                 # s
omega0 = 2.0 * np.pi       # rad/s, natural frequency
deltaI0 = 1.0              # initial amplitude of δI_R (arbitrary units)
deltaI0_dot = 0.0          # initial velocity

# Effective energy-density scaling (purely illustrative)
# ρ_eff ∝ (δI_R)^2 is enough to show the qualitative behavior
rho_scale = 1.0

# --- Time grid (several periods) ---
T = tau0                   # period = 1 s
n_periods = 5
t = np.linspace(0.0, n_periods * T, 2000)

# --- Analytic solution for linear oscillator ---
# δI_R(t) = A cos(ω0 t) + (v0/ω0) sin(ω0 t)
deltaI = deltaI0 * np.cos(omega0 * t) + (deltaI0_dot / omega0) * np.sin(omega0 * t)

# Effective energy density (up to a constant factor)
rho_eff = rho_scale * deltaI**2

# --- Plot δI_R(t) ---
plt.figure(figsize=(8, 4))
plt.plot(t, deltaI, label=r'$\delta I_R(t)$')
plt.axhline(0.0, color='k', linewidth=0.5)
plt.xlabel(r'$\tau$ (s)')
plt.ylabel(r'$\delta I_R$ (arb. units)')
plt.title(r'Linearized Informational Mode: $\ddot{\delta I_R} + \omega_0^2 \delta I_R = 0$, $\omega_0 = 2\pi$ rad/s')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("deltaIR_time_series.png", dpi=300)

# --- Plot ρ_eff(t) ---
plt.figure(figsize=(8, 4))
plt.plot(t, rho_eff, color='crimson', label=r'$\rho_{\mathrm{eff}}(t) \propto (\delta I_R)^2$')
plt.axhline(0.0, color='k', linewidth=0.5)
plt.xlabel(r'$\tau$ (s)')
plt.ylabel(r'$\rho_{\mathrm{eff}}$ (arb. units)')
plt.title(r'Effective Energy-Density Proxy from Informational Mode')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("rho_eff_time_series.png", dpi=300)

plt.show()
