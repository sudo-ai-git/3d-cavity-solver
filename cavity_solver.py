"""
SETTLED solver. Correct Dirichlet Laplacian: diagonal=4 (fixed), off-diag -1
for interior neighbors only. Validated against analytic square, then hexagon.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.special import jn_zeros
c = 299792458.0

def solve(mask, h, neigs=8):
    ny, nx = mask.shape
    node = {}
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            if mask[iy, ix]: node[(ix, iy)] = idx; idx += 1
    row=[]; col=[]; val=[]
    for (ix,iy),i in node.items():
        row.append(i); col.append(i); val.append(4.0)
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            jx,jy=ix+dx,iy+dy
            if 0<=jx<nx and 0<=jy<ny and mask[jy,jx]:
                row.append(i); col.append(node[(jx,jy)]); val.append(-1.0)
    A = csr_matrix((val,(row,col)), shape=(idx,idx))
    vv, _ = eigsh(A, k=min(neigs, idx-1), which='SM')
    vv = np.sort(np.clip(vv.real, 1e-15, None))
    return np.sqrt(vv)/h

def square_mask(n): return np.pad(np.ones((n-2,n-2),bool),1,constant_values=False)

def hexagon_mask(n, R_cells):
    cx=cy=(n-1)/2.0
    verts=[(R_cells*np.cos(np.radians(90+k*60)),R_cells*np.sin(np.radians(90+k*60))) for k in range(6)]
    verts=[(x+cx,y+cy) for (x,y) in verts]
    m=np.zeros((n,n),bool)
    for iy in range(n):
        for ix in range(n):
            x,y=ix,iy; inside=False;j=5
            for i in range(6):
                xi,yi=verts[i];xj,yj=verts[j]
                if ((yi>y)!=(yj>y)) and (x<(xj-xi)*(y-yi)/(yj-yi)+xi): inside=not inside
                j=i
            if inside: m[iy,ix]=True
    return m

print("=== 1) VALIDATION: analytic square a=0.10 m (Dirichlet rim) ===")
a=0.10
for n in (60,100,150):
    k=solve(square_mask(n),a/n,6)
    f=np.sort(k)*c/(2*np.pi)
    fa=sorted([c/2*np.sqrt(mm**2/a**2+nn**2/a**2) for mm in range(1,6) for nn in range(1,6)])[:6]
    # align by value not index (degeneracies)
    errs=[]
    for fi in f:
        errs.append(min(abs(fi-faj)/faj for faj in fa))
    print(f"  n={n}: nTM1={f[0]/1e9:.4f}GHz aTM1={fa[0]/1e9:.4f}GHz  minfirst={f[0]/fa[0]:.4f}  maxrelerr={max(errs)*100:.2f}%")
print()

print("=== 2) HEXAGON: TM modes ===")
for R0 in (0.05, 0.05):
    res=[]
    for n in (100,140,200):
        h=R0/(n/2.0)
        mask=hexagon_mask(n,n/2.0)
        k=solve(mask,h,4)
        f0=k[0]*c/(2*np.pi)
        res.append(f0)
        area=3*np.sqrt(3)/2*R0**2; Rc=np.sqrt(area/np.pi); fc=jn_zeros(0,1)[0]*c/(2*np.pi*Rc)
        print(f"    circumR={R0*100:.1f}cm grid={n}: TM0={f0/1e9:.4f}GHz  (area-cir {fc/1e9:.4f}GHz, ratio {f0/fc:.3f})")
    print(f"  -> converged TM0 ≈ {np.mean(res)/1e9:.3f} GHz for circumradius {R0*100:.1f}cm")
print()
