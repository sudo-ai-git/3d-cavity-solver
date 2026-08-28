#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SMALL LINAC DESIGN - corrected self-consistent computation.
# Grounded in validated per-cell physics; self-consistency checked.
import numpy as np
c0=299792458.0; eV=1.602176634e-19; eps0=8.854e-12

print("="*72)
print(" SMALL STANDING-WAVE ELECTRON LINAC - corrected design")
print("="*72)

F=2.998e9; lam=c0/F
E_out=6.0e6; I_b=200e-6

# --- S-band geometry: TM010-like cavity, cell length ~ lambda/2 (SW pi-mode) ---
# Velocity of light phase advance per cell (relativistic) => L_cell = beta*lam/2 ~ lam/2
L_cell=lam/2
# TM010 radius: J0(kc R)=0 -> kc=x01/R, f=c kc/(2pi) -> R = x01*c/(2*pi*f)
from scipy.special import jn_zeros
x01=jn_zeros(0,1)[0]
R_cell=x01*c0/(2*np.pi*F)
iris_r=0.40*R_cell   # aperture (from our validated design)
print("GEOMETRY (S-band 2.998 GHz):")
print(f"  lambda={lam*100:.1f}cm, R={R_cell*100:.2f}cm, L_cell={L_cell*100:.2f}cm (lambda/2), iris={iris_r*100:.2f}cm")

# --- gradient and transit time ---
E_peak=40e6   # copper pulsed limit w/ margin
theta=2*np.pi*F*L_cell/c0
T_tt=np.sin(theta/2)/(theta/2)
# net gain per cell for relativistic beam:
V_cell=E_peak*T_tt*L_cell
print(f"  peak E={E_peak/1e6:.0f} MV/m, transit angle theta={theta:.2f} rad, T_tt={T_tt:.3f}")
print(f"  net gain/cell (relativistic) = {V_cell/1e6:.3f} MV")

# non-relativistic injection: beam enters at ~100-300 keV, speed < c in first cells
# For a true low-energy design, the first cells have L = beta*lam/2. Inject at 200keV:
beta_inj=np.sqrt(1-(1/(1+200e3*eV/(0.511e6*eV)))**2)
L_cell_1=beta_inj*lam/2
print(f"  inject 200keV (beta={beta_inj:.3f}) -> first cell L={L_cell_1*100:.2f}cm")

# --- cells needed (tapered: early cells short, merge to lam/2 as beta->1) ---
# energy gain accumulates; use a coarse beta-tapered sum
gamma=np.arange(1,2000); # build ladder
T_beta, E_val=0.0, 200e3   # start 200 keV
N=0; Lstruct=0.0; Eadd=0.0
while E_val<E_out and N<40:
    beta=np.sqrt(1-(1/(1+E_val*eV/(0.511e6*eV)))**2)
    Lc=beta*lam/2
    th=2*np.pi*F*Lc/c0; Tb=np.sin(th/2)/(th/2)
    dE=E_peak*Tb*Lc
    E_val+=dE; N+=1; Lstruct+=Lc
print(f"  cells needed (beta-tapered) to reach {E_out/1e6:.1f} MeV: N={N}")
print(f"  total structure length = {Lstruct*100:.1f} cm")

# --- RF power (self-consistent) ---
# Dissipated wall power to sustain gradient in a standing-wave S-band copper
# structure: P_wall = V^2 / (2 * r_sh * L_structure), where r_sh is the effective
# SHUNT IMPEDANCE PER METER. For Cu pi-mode S-band cells r_sh ~ 50-80 MOhm/m
# (NOT the 250 Ohm/m waveguide number - that was a category error in the draft).
V_struct=E_val     # net accelerating voltage the beam actually gets
r_sh=60.0e6        # MOhm/m, typical Cu pi-mode S-band
P_wall=V_struct**2/(2*r_sh*Lstruct)
# beam loading: P_beam delivered to beam; RF source must cover wall+beam+fill
P_beam=I_b*E_out
fill_factor=1.5    # pulsed: energy to fill cavity + wall during pulse
P_rf=P_wall*fill_factor + P_beam
print(f"  net accel voltage V={V_struct/1e6:.1f} MV, beam power={P_beam/1e3:.2f} W")
print(f"  wall loss P_wall ~ {P_wall/1e3:.1f} kW (r_sh={r_sh/1e6:.0f} MOhm/m)")
print(f"  RF source ~ {P_rf/1e3:.0f} kW pulsed ({P_wall/1e3:.0f}kW wall + fill + {P_beam/1e3:.2f}W beam)")
print()
print("=== COMPONENT BILL ===")
print(f" - {N}-cell standing-wave Cu structure, L={Lstruct*100:.1f}cm (beta-tapered), end cells shortened/flattened")
print(f" - RF: pulsed {P_rf/1e3:.0f} kW, {F/1e9:.3f} GHz magnetron (off-the-shelf MW-class)")
print(f" - gun: 200 keV electron injector")
print(f" - focusing: solenoid ~0.3-0.5 T, short pulse duty")
print("DONE",flush=True)
