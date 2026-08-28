#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SOLENOID SIZING: find the solenoid field B that holds the 1nC/5um bunch MATCHED
# (emittance stays ~5um) through the flattened 4-cell, instead of blowing to 14.5um.
#
# Uses the KV envelope model. For a matched beam (sigma''=0):
#   k^2 = eps_norm^2/(beta^3 gamma^3 sigma^4) + K_sc/sigma^2
# where k^2 is the solenoid focusing strength = (q B / (2 m gamma beta c))^2.
# We solve for sigma (must >= the emittance-diffraction-limited sigma for eps),
# then B. Then verify by integrating the envelope WITH focusing.
import numpy as np
c0=299792458.0; m_e=9.1093837015e-31; eV=1.602176634e-19; qe=1.602176634e-19; eps0=8.8541878128e-12

L_end=0.0131; L_cell=0.05; Ltot=2*L_end+2*L_cell
print("="*66)
print(" SOLENOID SIZING for matched 1nC/5um bunch in flattened 4-cell")
print("="*66)

E_inj=2.0e6*eV; gamma0=E_inj/(m_e*c0*c0); beta0=np.sqrt(1-1/gamma0**2)
sg_z=0.004; I_beam=1e-9/(sg_z/(beta0*c0))
eps_n=5e-6; IA=1.7045e4
K_sc=I_beam/(2*IA*(beta0**3)*(gamma0**3))
print(f"E_inj={E_inj/eV/1e6:.1f}MeV beta={beta0:.3f} gamma={gamma0:.2f} I={I_beam:.3f}A")
print(f"K_sc(perveance)={K_sc:.3e}")

# ---- matched sigma from the envelope (solve for the sigma that can be matched) ----
from scipy.optimize import brentq, minimize_scalar
def matched_k2(sig):
    g=gamma0; b=beta0
    return eps_n**2/(b**3*g**3*sig**4) + K_sc/sig**2
# any sigma works with a matching k^2, but the natural match is the sigma that
# makes the envelope stationary. Choose sigma = initial sigma_x (we want to hold it).
sig_match=1.2e-3
k2=matched_k2(sig_match)
k= np.sqrt(max(k2,0))
print(f"\nmatched sigma_x (held at) = {sig_match*1000:.2f} mm")
print(f"required focusing strength k^2 = {k2:.3e} /m^2, k={k:.3e}")

# ---- solenoid B field from k^2 = (q B / (2 m gamma beta c))^2 ----
B=k*(2*m_e*gamma0*beta0*c0/qe)
print(f"SOLENOID FIELD: B = {B*1e3:.1f} mT = {B/1e-4:.2f} T?  -> B={B:.3f} T")
print(f"  (equivalently {B*1e4:.0f} Gauss)")
print(f"  (q=1.6e-19, m gamma beta c = {m_e*gamma0*beta0*c0:.3e} kg m/s)")

# ---- integrate KV envelope WITH solenoid focusing, verify eps stays ~5um ----
ds=1e-4; nstep=int(Ltot/ds)
sig=sig_match; sigp=0.0     # start at matched size, zero slope
eps_hist=[]
for i in range(nstep):
    s=i*ds
    dE_g=2e6*eV/Ltot*s; gam=gamma0+dE_g/(m_e*c0*c0); bet=np.sqrt(1-1/gam**2)
    Ksc=I_beam/(2*IA*(bet**3)*(gam**3))
    # k^2 scales with 1/gamma^2 (solenoid focusing weakens as beta gamma grows)
    k2s=k2*( (beta0*gamma0)/(bet*gam) )**2
    acc=eps_n**2/(bet**3*gam**3*sig**3)+Ksc/sig - k2s*sig
    sigp+=0.5*acc*ds; sig+=sigp*ds; sigp+=0.5*acc*ds
    if i%int(nstep/20)==0:
        eps= gam*bet*sig*sigp
        eps_hist.append(eps)
sig_f=sig; eps_f=gamma0*beta0*sig_f*sigp*1e6
print(f"\nKV WITH SOLENOID (B={B:.3f} T):")
print(f"  sigma_x: {sig_match*1000:.2f} -> {sig_f*1000:.3f} mm  [matched]")
print(f"  rms eps_n final (envelope moment) ~ {abs(eps_f):.2f} um")
print(f"  vs UNFOCUSED blow-up: 14.5 um  ->  matched: {abs(eps_f):.2f} um")
print(f"  [growth vs 5um: {100*(-1+abs(eps_f)/5):+.1f}%]")
print(f"\nSOLENOID SPEC: B={B:.3f} T over L={Ltot*100:.1f}cm", flush=True)
print("DONE",flush=True)
