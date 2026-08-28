#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DIAGNOSTIC: validate B-field extraction against an ANALYTIC cavity mode.
# Plain cylinder (no nose) TM010 has exact Bessel E and B. We solve it with the
# FEM solver, extract B via B=(i/w)curl E, and compare the RATIO |B|/(E/c) to the
# known value. If the ratio is ~0.58 (the Bessel J1 max / E0), the curl method is
# validated; if it's off (like the earlier 7.8), we isolate gridding vs stencil.
#
# For a TM010 cylinder (radius R), the fields are:
#   Ez(r)   = E0 * J0(kc r)            kc = x01/R, x01 = 2.4048
#   Bphi(r) = -E0 * J1(kc r) / c
# so on-axis Ez=E0, Bphi(0)=0, and max(Bphi)=E0*max(J1)/c ~ 0.58*E0/c.
# Gate:  max|B| / (max|Ez|/c)  ~ 0.58  (the J1 max).  Range 0.5-0.65 acceptable.
import numpy as np
import subprocess, sys
try:
    import skfem
except Exception:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","scikit-fem"])
    import skfem
print(f"scikit-fem {skfem.__version__}", flush=True)
from skfem import (MeshHex, ElementTetN1, Basis, BilinearForm, asm)
from skfem.helpers import dot, curl as _curl
from scipy.linalg import eigh
from scipy.special import jn_zeros, jn

c0=299792458.0

@BilinearForm
def stiff(u,v,w): return dot(_curl(u),_curl(v))
@BilinearForm
def mass(u,v,w):  return dot(u,v)

def solve_mode(mesh):
    basis=Basis(mesh,ElementTetN1())
    A=asm(stiff,basis); B=asm(mass,basis)
    bd=basis.get_dofs(facets=mesh.boundary_facets())
    b_ext=np.sort(np.unique(np.asarray(bd.all()).ravel()))
    free=np.setdiff1d(np.arange(A.shape[0]), b_ext)
    Aff=A[free][:,free].toarray(); Bff=B[free][:,free].toarray()
    lam,X=eigh(Aff,Bff)
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    o=np.argsort(f); f=f[o]; X=X[:,o]
    phys=f>1e8
    mi=np.where(phys)[0][0]
    f0=f[mi]
    vfull=np.zeros(A.shape[0]); vfull[free]=X[:,mi]
    return f0, vfull, basis

def extract_B(f0, vfull, basis, side, L, gx,gy,gz, border=0.03):
    w0=2*np.pi*f0
    xs=np.linspace(border*side,(1-border)*side,gx)
    ys=np.linspace(border*side,(1-border)*side,gy)
    zs=np.linspace(border*L,(1-border)*L,gz)
    xx,yy,zz=np.meshgrid(xs,ys,zs,indexing='ij')
    pts=np.stack([xx.ravel(),yy.ravel(),zz.ravel()],1).T
    P=basis.probes(pts)
    Eshort=(P@vfull).toarray().ravel() if hasattr(P@vfull,'toarray') else np.asarray(P@vfull).ravel()
    E=Eshort.reshape(-1,3).reshape(gx,gy,gz,3)
    hx=xs[1]-xs[0]; hy=ys[1]-ys[0]; hz=zs[1]-zs[0]
    dEx=np.gradient(E,hx,axis=0); dEy=np.gradient(E,hy,axis=1); dEz=np.gradient(E,hz,axis=2)
    curlx=dEz[...,1]-dEy[...,2]
    curly=dEx[...,2]-dEz[...,0]
    curlz=dEy[...,0]-dEx[...,1]
    Bx=curlx/w0; By=curly/w0; Bz=curlz/w0
    return np.asarray(E), (np.asarray(Bx),np.asarray(By),np.asarray(Bz)), (xs,ys,zs)

# ---------- analytic TM010 cylinder ----------
R_th=0.045
side=np.sqrt(np.pi)*R_th   # area-matched square for the FEM prism
L=0.05
x01=jn_zeros(0,1)[0]
kc=x01/R_th
Omega0=c0*kc
f0_an=Omega0/(2*np.pi)
E0=1.0
rgrid=np.linspace(0,R_th,300)
Bmax_an=E0*np.max(jn(1,kc*rgrid))/c0
ratio_an=Bmax_an/(E0/c0)

print("="*64)
print(" ANALYTIC TM010 CYLINDER — B-EXTRACTION GATE")
print("="*64)
print(f"analytic: R={R_th*100:.2f}cm, f0={f0_an/1e9:.4f}GHz")
print(f"analytic ratio max|B|/(E_peak/c) = {ratio_an:.4f}   (expect ~0.58, the J1 max)")
print()

# FEM on area-matched square prism
mesh=MeshHex.init_tensor(np.linspace(0,side,10),np.linspace(0,side,10),
                         np.linspace(0,L,8)).to_meshtet()
f0_fem,vfull,basis=solve_mode(mesh)
print(f"FEM square-prism f0 = {f0_fem/1e9:.4f}GHz (analytic {f0_an/1e9:.4f})  "
      f"ratio {f0_fem/f0_an:.3f}")
print()

# Grid convergence of the ||B||/(|E|/c) ratio on the SAME solved field
for (gx,gy,gz) in [(8,8,6),(12,12,8),(16,16,10),(20,20,12)]:
    E,(Bx,By,Bz),(xs,ys,zs)=extract_B(f0_fem,vfull,basis,side,L,gx,gy,gz)
    Emax=np.abs(E).max(); Bmax=max(np.abs(Bx).max(),np.abs(By).max(),np.abs(Bz).max())
    ratio=Bmax/(Emax/c0)
    print(f"  grid {gx:2d}x{gy:2d}x{gz:2d}: |E|max={Emax:8.2f} |B|max={Bmax:.3e} "
          f"ratio={ratio:.3f}  (analytic {ratio_an:.3f})")

print()
print("INTERPRETATION:")
print(" - ratio ~0.58 (=analytic): curl extraction is CORRECT.")
print(" - ratio >> 1 (like 7.8): stencil or gridding inflates curl near field")
print("   gradients -> need finer grid / higher-order stencil / wall-conforming curl.")
