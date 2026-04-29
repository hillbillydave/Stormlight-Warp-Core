Stormlight‑Warp‑Core
Stormlight‑Warp‑Core is the reference implementation of the ODIM‑U resonant warp‑drive engine.
It numerically evolves a 3D informational‑geometry warp bubble driven at the natural 2π‑Hz spacetime resonance, demonstrating stable bubble motion, strictly positive informational energy density, and ultra‑low‑power operation without exotic matter.

This repository contains the full simulation code used in the research program:

Informational Geometry, the 2π‑Hz Resonance, and Warp‑Drive Dynamics Without Exotic Matter

The upcoming papers on regenerative driving, minimum‑power warp operation, and solid‑state artificial gravity via resonant time‑axis tilting.

Stormlight‑Warp‑Core is a numerical laboratory for exploring the deep structure of the ODIM‑U framework: informational curvature, shift‑vector dynamics, asymmetric bubble shaping, and the low‑power limit of resonant spacetime engineering.

Features
3D high‑resolution grid (64³) for clean bubble geometry

2π‑Hz resonance engine matching the natural curvature eigenmode

Asymmetric front/rear bubble shaping for forward translation

Shift‑vector evolution tied to bubble motion

Ultra‑low‑power regime with tail power collapsing by orders of magnitude

Strictly positive 
𝑇
00
 informational energy density

Ship and bubble trajectories in both grid and solar‑system frames

Energy diagnostics including instantaneous and steady‑tail power

Slices of 
𝐼
𝑅
 and 
𝑇
00
 for visualization and publication

Scientific Background
Stormlight‑Warp‑Core implements the ODIM‑U (Observer‑Dependent Informational Metric – Unified) geometry, where:

The informational field 
𝐼
𝑅
 defines curvature in an extended configuration space

The shift vector 
𝑁
𝑖
 drives bubble translation

The natural resonance of the system occurs at 2π rad/s

Operating at this resonance minimizes power consumption

The asymmetric bubble shape produces forward motion without exotic matter

The simulation evolves:

𝐼
¨
𝑅
=
𝑐
𝐼
2
∇
2
𝐼
𝑅
−
𝛽
𝐼
𝐼
˙
𝑅
−
Ω
0
2
𝐼
𝑅
+
advection
(
𝑁
𝑖
,
𝐼
𝑅
)
with the informational energy density:

𝜀
𝐼
=
1
2
(
𝐼
˙
𝑅
2
+
𝑐
𝐼
2
∣
∇
𝐼
𝑅
∣
2
+
Ω
0
2
𝐼
𝑅
2
)
.
The engine demonstrates that a warp bubble can be:

Stable

Positive‑energy

Low‑power

Resonantly driven

all within a physically consistent informational geometry.

Simulation Details
Grid: 64 × 64 × 64

Domain: 300 m cubic region

Time: ~0.1–0.2 s physical time

CFL‑safe timestep

Bubble radius: 50 m

Initial external‑frame velocity: 5c (toy model)

Damping: β = 100

Amplitude: 
5
×
10
−
7
 (low‑power regime)

Asymmetry: front expansion ×8, rear compression ×0.05

Outputs include:

energy_and_power.png

ship_bubble_x_grid.png

ship_x_solar.png

I_R_slices.png

T00_slices.png

telemetry.txt

ship_trajectory.txt

Installation
Clone the repository:

Code
git clone https://github.com/hillbillydave/Stormlight-Warp-Core.git
cd Stormlight-Warp-Core
Install dependencies:

Code
pip install numpy matplotlib
Run the simulation:

Code
python3 warp_sim.py
Output Structure
Each run creates a timestamped directory under:

Code
Stormlight-Warp-Core/v3.1_geom_FTL_5c_longrun_hiRes_lowPower/run_YYYY-MM-DD_HH-MM-SS/
Containing:

Telemetry logs

Ship/bubble trajectories

Energy and power plots

Field slices

Console summary

Peaceful‑Use License (IBPUL‑X.1)
This project is released under the Information‑Bubble Peaceful‑Use License (IBPUL‑X.1).

Core principles:

No military, weaponized, or harmful applications

Open hardware, open methods, right‑to‑repair

Attribution required

Derivatives must remain peaceful‑use

Scientific transparency and reproducibility

A full copy of the license will be included in the repository.

Citation
If you use this code in research, please cite:

Blackwell, D. (2026). Stormlight‑Warp‑Core: ODIM‑U Resonant Warp‑Drive Simulation Engine. GitHub Repository.

And the associated papers:

Blackwell, D. & Beardsley, I. (2026). Informational Geometry, the 2π‑Hz Resonance, and Warp‑Drive Dynamics Without Exotic Matter.

Acknowledgments
This engine was developed as part of a collaborative research effort exploring informational geometry, resonance‑driven propulsion, and the deep structure of spacetime.
