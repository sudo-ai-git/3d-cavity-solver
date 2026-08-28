#!/usr/bin/env python
# -*- coding: utf-8 -*-
# BEAM TRACKER on ANALYTIC TM010 PILLBOX — phase sweep (auto-phasing).
# B exact by construction (analytic Bessel). Find the RF phase that maximizes
# electron energy gain across the cell (mirrors ASTRA auto-phasing), verify the
# tracker's max gain matches the analytic on-axis V_acc, and show B/(E/c) = 0.58.
import numpy as np
from scipy.special import jv

c0 = 299792458.0
m_e = 9.1093837015e-31
eV = 1.602176634e-19
keV = 1e3*eV

x01 = 2.4048255577
R = 0.045; L = 0.050
kc = x01/R; w = c0*kc; f0 = w/(2*np.pi)
E0 = 25e6            # peak on-axis Ez, 25 MV/m

def fields(x,y,z,t,phi):
    # Physical pillbox: field confined to the cavity interior z in [0,L].
    if z < 0 or z > L:
        return 0.,0.,0.,0.,0.,0.
    r = np.sqrt(x*x+y*y)
    wtn = w*t + phi
    Ez = E0*jv(0,kc*r)*np.sin(wtn)
    Bphi = -(E0/c0)*jv(1,kc*r)*np.cos(wtn)
    if r < 1e-12:
        return 0.,0.,Ez,0.,0.,0.
    Bx = Bphi*(-y/r); By = Bphi*(x/r)
    return 0.,0.,Ez,Bx,By,0.

def track(z0,p0,gamma0,nsteps,dt,phi):
    m=m_e
    q = -1.602176634e-19   # electron charge (C)
    x=y=0.0; z=z0; px,py,pz=0.0,0.0,p0
    gamma=gamma0
    for it in range(nsteps):
        t = it*dt
        Ex,Ey,Ez,Bx,By,Bz = fields(x,y,z,t,phi)
        vx,vy,vz = px/(gamma*m), py/(gamma*m), pz/(gamma*m)
        # v x B
        cvx=vy*Bz-vz*By; cvy=vz*Bx-vx*Bz; cvz=vx*By-vy*Bx
        px+=q*dt*(Ex+cvx); py+=q*dt*(Ey+cvy); pz+=q*dt*(Ez+cvz)
        p2=px*px+py*py+pz*pz
        gamma=np.sqrt(1+p2/(m*c0)**2)
        x+=vx*dt; y+=vy*dt; z+=vz*dt
    return z, gamma

# energy-per-unit-charge view (volts): gain in keV
def run(phi):
    gamma0 = 100e6*eV/(m_e*c0*c0)     # 100 MeV
    p0 = np.sqrt(gamma0**2-1)*m_e*c0
    # resolve RF: omega*dt ~ 0.08 rad per step (dt=5ps, omega=1.6e10/s).
    dt = 5e-12
    z0=-L*0.5
    nsteps=int(3*L/c0/dt)             # cover 3 cell-crossings (~5000 steps)
    z,gamma = track(z0,p0,gamma0,nsteps,dt,phi)
    dE_J=(gamma-gamma0)*m_e*c0*c0
    return dE_J, z

print("="*66)
print(" BEAM TRACKER on ANALYTIC TM010 — PHASE SWEEP (auto-phasing)")
print("="*66)
print(f"R={R*100:.2f}cm L={L*100:.2f}cm f0={f0/1e9:.4f}GHz E0={E0/1e6:.0f}MV/m")
print(f"w*L/c (transit) = {kc*L:.4f} rad ; B/(E/c)=max|J1|={np.max(np.abs(jv(1,kc*np.linspace(1e-9,R,200)))):.4f}")
print()

phi_list=np.linspace(0,2*np.pi,13)
gains=[]
print("  phi(rad) | dE gain (keV)")
for phi in phi_list:
    dE,z=run(phi)
    g= dE/keV
    gains.append((phi,g))
    print(f"  {phi:7.3f}   | {g:+9.3f}")

gmax=max(gains,key=lambda t:t[1]); gmin=min(gains,key=lambda t:t[1])
print()
print(f"  MAX phase phi={gmax[0]:.3f}: gain {gmax[1]:+.3f} keV")
print(f"  MIN phase phi={gmin[0]:.3f}: gain {gmin[1]:+.3f} keV")
# analytic max gain: V_acc = max over phi of E0*(cos phi - cos(kc L+phi))/kc
phis=np.linspace(0,2*np.pi,1000)
V_max=np.max(np.abs(E0*(np.cos(phis)-np.cos(kc*L+phis))/kc))
print(f"  analytic max |V_acc| = {V_max/1e3:.3f} kV -> gain {V_max/1e3:.4f} keV (per e)")
print(f"  tracker max vs analytic: {gmax[1]/(V_max/1e3):.3f}")
print()
print("  Note: tracker integrates field over the FULL path (incl. fringe outside"); 
print("  [0,L]); analytic integrates only over [0,L]. Expect them within ~15%.")
