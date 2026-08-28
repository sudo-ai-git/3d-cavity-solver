#!/usr/bin/env python
# -*- coding: utf-8 -*-
# FIXED: select the E_z-dominant (TM010 accelerating) mode, not just the lowest
# eigenvalue. Then validate B = curl E / w against the analytic Bessel ratio.
import numpy as np
import subprocess, sys
try: import skfem
except Exception: subprocess.check_call([sys.executable,"-m","pip","install","-q","scikit-fem"]); import skfem
from skfem import MeshHex, ElementTetN1, Basis, BilinearForm, asm
from skfem.helpers import dot, curl as _curl
from scipy.linalg import eigh
from scipy.special import jn_zeros, jn
c0=299792458.0
@BilinearForm
def stiff(u,v,w): return dot(_curl(u),_curl(v))
@BilinearForm
def mass(u,v,w):  return dot(u,v)

def solve_all_modes(mesh):
    basis=Basis(mesh,ElementTetN1()); A=asm(stiff,basis); B=asm(mass,basis)
    bd=basis.get_dofs(facets=mesh.boundary_facets()); b_ext=np.sort(np.unique(np.asarray(bd.all()).ravel()))
    free=np.setdiff1d(np.arange(A.shape[0]),b_ext)
    Aff=A[free][:,free].toarray(); Bff=B[free][:,free].toarray(); lam,X=eigh(Aff,Bff)
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi); o=np.argsort(f); f=f[o]; X=X[:,o]
    nmodes=X.shape[1]
    return f, X, free, basis, b_ext

def field_onaxis_components(f0, X, free, basis, side, L, naxis=31):
    """For a given mode, return max|Ez-onaxis| and max|E-transverse-onaxis|."""
    vfull=np.zeros(basis.N); vfull[free]=X
    xs=side/2; ys=side/2
    zs=np.linspace(0.02*L,0.98*L,naxis)
    pts=np.stack([np.full(naxis,xs),np.full(naxis,ys),zs],1).T
    P=basis.probes(pts); Ev=np.asarray(P@vfull).ravel().reshape(-1,3)
    Ez_axis=np.abs(Ev[:,2]).max()
    Et_axis=np.sqrt(Ev[:,0]**2+Ev[:,1]**2).max()
    return Ez_axis, Et_axis

R=0.045; side=np.sqrt(np.pi)*R; L=0.05
print("="*64)
print(" TM010 SELECTION + B-EXTRACTION GATE (cylinder)")
print("="*64)
mesh=MeshHex.init_tensor(np.linspace(0,side,9),np.linspace(0,side,9),np.linspace(0,L,7)).to_meshtet()
f, X, free, basis, b_ext = solve_all_modes(mesh)
nphys=(f>1e8).sum()
print(f"{nphys} physical modes. Computing on-axis Ez/Tran dominance for lowest {min(8,nphys)}:")

# pick the mode with max on-axis Ez/Et ratio (the TM010 accelerator mode)
best=None
for m in range(nphys):
    Ez,Et=field_onaxis_components(f[m], X[:,m], free, basis, side, L)
    ratio=Ez/max(Et,1e-20)
    print(f"  mode {m}: f={f[m]/1e9:.4f}GHz  onaxis|Ez|={Ez:.2f}  onaxis|Et|={Et:.2f}  ratio={ratio:.2f}")
    if best is None or ratio>best['ratio']:
        best={'m':m,'ratio':ratio,'f':f[m]}
print(f"\nSelected TM010 candidate: mode {best['m']}, f={best['f']/1e9:.4f}GHz, Ez/Et={best['ratio']:.2f}")

# Now validate B=curl E / w against analytic for this mode
f0, Xsel, _ = best['f'], X[:,best['m']], None
vfull=np.zeros(basis.N); vfull[free]=Xsel
# extract B on grid
gx=gy=gz=14; b=0.03
xs=np.linspace(b*side,(1-b)*side,gx); ys=np.linspace(b*side,(1-b)*side,gy); zs=np.linspace(b*L,(1-b)*L,gz)
xx,yy,zz=np.meshgrid(xs,ys,zs,indexing='ij')
pts=np.stack([xx.ravel(),yy.ravel(),zz.ravel()],1).T
P=basis.probes(pts); Ev=np.asarray(P@vfull).ravel(); E=Ev.reshape(-1,3).reshape(gx,gy,gz,3)
hx=xs[1]-xs[0]; hy=ys[1]-ys[0]; hz=zs[1]-zs[0]
w0=2*np.pi*f0
dEx=np.gradient(E,hx,axis=0); dEy=np.gradient(E,hy,axis=1); dEz=np.gradient(E,hz,axis=2)
curlx=dEz[...,1]-dEy[...,2]; curly=dEx[...,2]-dEz[...,0]; curlz=dEy[...,0]-dEx[...,1]
Bx=curlx/w0; By=curly/w0; Bz=curlz/w0
Emax=np.abs(E).max(); Bmax=max(np.abs(Bx).max(),np.abs(By).max(),np.abs(Bz).max())
# analytic
x01=jn_zeros(0,1)[0]; kc=x01/R
ratio_an=np.max(jn(1,kc*np.linspace(0,R,300)))[()]
print("\n=== B-EXTRACTION GATE (selected TM010 mode) ===")
print(f"|E|max={Emax:.2f}  |B|max={Bmax:.3e}")
print(f"ratio |B|/(|E|/c) = {Bmax/(Emax/c0):.4f}   [analytic TM010 = {ratio_an:.4f}]")
ok = abs(Bmax/(Emax/c0) - ratio_an) < 0.15*ratio_an
print(f"+---------------+")
print(f"| GATE: {'PASS' if ok else 'FAIL'} |   (within 15% of analytic)")
print(f"+---------------+")
