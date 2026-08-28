#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Kaggle GPU: Nose-cone accelerating cell DESIGN SWEEP on the validated 3D FEM
# solver. Sweep nose geometry (radius, penetration) to maximize the on-axis
# transit-time factor and accelerating voltage (the real 'stronger' metrics).
import numpy as np

c0 = 299792458.0
import subprocess, sys
try:
    import skfem
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-fem"])
    import skfem
print(f"scikit-fem {skfem.__version__}", flush=True)

from skfem import (MeshHex, ElementTetN1, Basis, BilinearForm, asm)
from skfem.helpers import dot, curl

@BilinearForm
def stiff(u,v,w): return dot(curl(u),curl(v))
@BilinearForm
def mass(u,v,w):  return dot(u,v)

def assemble_pair(mesh):
    basis = Basis(mesh, ElementTetN1())
    A=asm(stiff,basis); B=asm(mass,basis)
    bd=basis.get_dofs(facets=mesh.boundary_facets())
    b_ext=np.sort(np.unique(np.asarray(bd.all()).ravel()))
    return A,B,basis,b_ext

Rc=0.045; L=0.05
side=np.sqrt(np.pi)*Rc
nx=ny=12; nz=10
base_mesh=MeshHex.init_tensor(
    np.linspace(0,side,nx+1),np.linspace(0,side,ny+1),
    np.linspace(0,L,nz+1)).to_meshtet()
A,B,basis,b_ext=assemble_pair(base_mesh)
xyz=basis.doflocs
b_ext_set=set(b_ext.tolist())

def analyze(nose_params):
    """Return dict with f0, T, Vacc for given nose (r_frac,l_frac) or None."""
    all_pec=b_ext
    if nose_params is not None:
        nr_frac,nl_frac=nose_params
        nr=nr_frac*side/2; nl=nl_frac*(L/2)
        nd=[]
        for i in range(A.shape[0]):
            if i in b_ext_set: continue
            x,y,z=xyz[0,i]-side/2, xyz[1,i]-side/2, xyz[2,i]
            rr=np.hypot(x,y)
            if (rr<=nr and z<=nl) or (rr<=nr and z>=L-nl):
                nd.append(i)
        all_pec=np.unique(np.concatenate([b_ext,np.array(nd)]))
    free=np.setdiff1d(np.arange(A.shape[0]),all_pec)
    from scipy.linalg import eigh
    Aff=A[free][:,free].toarray(); Bff=B[free][:,free].toarray()
    lam,X=eigh(Aff,Bff)
    f=np.sqrt(np.maximum(lam,0))*c0/(2*np.pi)
    o=np.argsort(f); f=f[o]; X=X[:,o]
    phys=f>1e8
    if len(f[phys])==0: return None
    f0=f[phys][0]
    vfull=np.zeros(A.shape[0]); vfull[free]=X[:,phys][:,0]
    # on-axis
    zi=[]; ei=[]
    for i in range(A.shape[0]):
        if abs(xyz[0,i]-side/2)<1e-9 and abs(xyz[1,i]-side/2)<1e-9:
            zi.append(xyz[2,i]); ei.append(vfull[i])
    zi=np.array(zi); ei=np.array(ei)
    o2=np.argsort(zi); zi=zi[o2]; ei=ei[o2]
    w=2*np.pi*f0
    Ii=np.trapezoid(ei*np.exp(-1j*w/c0*zi),zi)
    Ir=np.trapezoid(np.abs(ei),zi)
    T=abs(Ii)/(Ir+1e-30); Vacc=abs(Ii)
    return dict(f0=f0,T=T,Vacc=Vacc,naxis=len(zi))

print("="*72)
print(" NOSE-CONE DESIGN SWEEP — maximize transit-time & accelerating voltage")
print("="*72)
print(f"cell: area-matched square side {side*100:.2f}cm, L={L*100:.0f}cm, "
      f"dofs={A.shape[0]}", flush=True)

configs=[("PLAIN (no nose)",None),
         ("nose r=40% l=25%",(0.40,0.25)),
         ("nose r=40% l=50%",(0.40,0.50)),
         ("nose r=40% l=75%",(0.40,0.75)),
         ("nose r=30% l=50%",(0.30,0.50)),
         ("nose r=50% l=50%",(0.50,0.50)),
         ("nose r=30% l=75%",(0.30,0.75))]

results=[]
for name,param in configs:
    r=analyze(param)
    if r:
        results.append((name,param,r))
        print(f"  {name:<24} f0={r['f0']/1e9:.4f}GHz  T={r['T']:.4f}  Vacc={r['Vacc']:.4e}  (naxis={r['naxis']})",
              flush=True)

print()
print("="*72)
print(" RANKING by effective accelerating voltage (Vacc) and T")
print("="*72)
results.sort(key=lambda x: -x[2]['Vacc'])
best=results[0]
for i,(name,param,r) in enumerate(results):
    mark=" <== BEST" if i==0 else ""
    print(f"  {i+1}. {name:<24} T={r['T']:.4f}  Vacc={r['Vacc']:.4e}{mark}", flush=True)

base=next(r for (n,p,r) in results if p is None)
print()
print(f"BEST ({best[0]}) vs PLAIN:")
print(f"  transit-time T:   {base['T']:.4f} -> {best[2]['T']:.4f}  ({100*(best[2]['T']-base['T'])/base['T']:+.1f}%)")
print(f"  accelerating V:   {base['Vacc']:.3e} -> {best[2]['Vacc']:.3e}  ({100*(best[2]['Vacc']-base['Vacc'])/base['Vacc']:+.1f}%)")
print()
print("(higher Vacc = more on-axis voltage the beam gains = stronger cell)")
print("DONE", flush=True)
