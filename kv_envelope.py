#!/usr/bin/env python
# -*- coding: utf-8 -*-
# KV ENVELOPE MODEL: emittance growth of a 1 nC / 5um bunch through the flattened
# 4-cell pi-mode, with proper space-charge perveance and relativistic gamma.
# The Kapchinskij-Vladimirskij (envelope) equation is the standard beam-physics
# tool (used by TRACE/TRACE3D) and is numerically stable - unlike the previous
# raw-momentum PIC attempt. Gives matched/unmatched rms emittance evolution.
#
#   sigma'' + k2(s)*sigma = eps_norm^2/(beta^3 gamma^3 * sigma^3)   [emittance term]
#                          + K_sc/sigma                             [space charge]
#   K_sc = q I / (2 pi eps0 m c^3 (beta^3 gamma^3))   per length, I=beam current
#   eps_norm = gamma beta sigma sigma'  (normalized emittance)
import numpy as np

c0=299792458.0; m_e=9.1093837015e-31; eV=1.602176634e-19; qe=1.602176634e-19
eps0=8.8541878128e-12

# ---- structure (flattened 4-cell) ----
L_end=0.0131; L_cell=0.05; Ltot=2*L_end+2*L_cell   # 1.31,5,5,1.31 = 12.62 cm
print("="*66)
print(" KV ENVELOPE: emittance + sigma_x through flattened 4-cell pi-mode")
print("="*66)
print(f"structure length = {Ltot*100:.1f} cm (flattened 4-cell: 1.31,5,5,1.31)")

# ---- bunch & beam params ----
E_inj=2.0e6*eV; gamma0=E_inj/(m_e*c0*c0); beta0=np.sqrt(1-1/gamma0**2)
sg_z=0.004
I_beam=1e-9/(sg_z/(beta0*c0))        # 1 nC per 4mm bunch -> peak current
print(f"E_inj={E_inj/eV/1e6:.1f} MeV (beta={beta0:.3f}), I_beam={I_beam:.3f} A per pulse")
eps_n=5e-6; sg_x0=1.2e-3
sgx0p_em=eps_n/(gamma0*beta0*sg_x0)   # divergence from emittance

# space-charge perveance K_sc (per unit length) for round beam:
IA=1.7045e4         # Alfven current (A), m_e c^3/qe = 1.7045e4
K_sc=I_beam/(2*IA*(beta0**3)*(gamma0**3))
print(f"space-charge perveance K_sc = {K_sc:.3e}")

# ---- envelope integration (symplectic leapfrog, s = path length) ----
# sigma'' = eps_const^2/sigma^3 + K_sc/sigma  (round beam, no external focus here)
# BUT: emittance is NOT conserved if space charge is nonlinear (envelope model keeps
# sigma_rms; eps_n is conserved only for KV beams). We integrate sigma AND track
# sigma' (envelope slope); the rms emittance = gamma beta sigma sigma'.
eps_const=eps_n/(gamma0*beta0)   # normalized emittance term in the envelope eq (invariant)
ds=1e-4; nstep=int(Ltot/ds)
sig=sg_x0; sigp=sgx0p_em   # matched-ish start (divergence from emittance)
eps_rms_history=[eps_n]
for i in range(nstep):
    s=i*ds
    # acceleration changes gamma as beam gains energy -> K_sc and emittance scale
    # Model energy gain as linear over structure (2 MeV over 12.6cm):
    dE_g=2e6*eV/(Ltot)*s      # energy gained so far
    gam=gamma0+dE_g/(m_e*c0*c0); bet=np.sqrt(1-1/gam**2)
    Ksc=I_beam/(2*IA*(bet**3)*(gam**3))
    # envelope eq (no external focus k=0): sigma'' = eps^2/sigma^3 + Ksc/sigma
    acc=eps_const**2/(sig**3) + Ksc/sig
    # leapfrog: half kick, drift, half kick
    sigp+=0.5*acc*ds; sig+=sigp*ds; sigp+=0.5*acc*ds
    if i%int(nstep/20)==0:
        eps= gam*bet*sig*sigp     # rms normalized emittance (from envelope moment)
        eps_rms_history.append(eps)
sig_final=sig
print(f"\nKV ENVELOPE RESULT:")
print(f"  sigma_x: {sg_x0*1000:.2f} -> {sig_final*1000:.2f} mm")
print(f"  final envelope divergence: {sigp:.4f} mrad")
print(f"  rms normalized emittance (envelope moment) final~{gamma0*beta0*sig*sigp*1e6:.2f} um")
print(f"  (initial eps_n = {eps_n*1e6:.2f} um; growth = {100*(abs(gamma0*beta0*sig*sigp)/eps_n-1):+.1f}%)")
print()
print("INTERPRETATION:")
if abs(gamma0*beta0*sig*sigp/eps_n-1)<0.3:
    print("  Emittance roughly conserved -> beam stays matched, modest SC blowup")
else:
    print("  Emittance grew -> space-charge defocusing without external focusing")
    print("  (a real solenoid lattice would restore matching - TRACE/ASTRA step)")
print("DONE",flush=True)
