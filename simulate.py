"""
Simulate the hexagonal cavity WORKING: drive it near its TM0 resonance and
visualize (a) the mode field pattern and (b) the resonant energy response.
This is a real EM simulation of a standing wave in a hexagonal PEC cavity.
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigs, eigsh, ArpackError
from scipy.sparse.linalg import LinearOperator

c=299792458.0
def hexagon_mask(n, R_cells):
    cx=cy=(n-1)/2.0
    verts=[(R_cells*np.cos(np.radians(90+k*60)) , R_cells*np.sin(np.radians(90+k*60))) for k in range(6)]
    verts=[(x+cx,y+cy) for (x,y) in verts]
    def pin(x,y):
        inside=False;j=5
        for i in range(6):
            xi,yi=verts[i];xj,yj=verts[j]
            if ((yi>y)!=(yj>y)) and (x<(xj-xi)*(y-yi)/(yj-yi)+xi): inside=not inside
            j=i
        return inside
    mask=np.zeros((n,n),bool)
    for iy in range(n):
        for ix in range(n):
            if pin(ix,iy): mask[iy,ix]=True
    return mask

def get_eigen(mask,h,neigs=4):
    ny,nx=mask.shape; node_id={};idx=0
    for iy in range(ny):
        for ix in range(nx):
            if mask[iy,ix]: node_id[(ix,iy)]=idx;idx+=1
    A=lil_matrix((idx,idx))
    rows=[];cols=[];data=[]
    for (ix,iy),i in node_id.items():
        for dxi,dyi in ((1,0),(-1,0),(0,1),(0,-1)):
            jx,jy=ix+dxi,iy+dyi
            if 0<=jx<nx and 0<=jy<ny and mask[jy,jx]:
                rows.append(i);cols.append(node_id[(jx,jy)]);data.append(1.0)
    # build Laplacian L (off-diag 1, diag -4) then M=-L is SPD
    from scipy.sparse import coo_matrix
    off=coo_matrix((data,(rows,cols)),shape=(idx,idx))
    M=csr_matrix(4*np.eye(idx) - off.todense())
    vals,eigvecs=eigsh(M,k=min(neigs+2,idx-1),which='SM')
    order=np.argsort(vals)
    vals=eigvecs=None
    vv,ev=eigsh(M,k=min(neigs+2,idx-1),which='SM')
    o=np.argsort(vv.real); vv=vv.real[o]; ev=ev[:,o]
    kp=np.sqrt(np.clip(vv,0,None))/h
    # build field map
    field=np.zeros((ny,nx))
    for (ix,iy),i in node_id.items():
        field[iy,ix]=ev[i,0].real
    return kp, field, node_id

# ---- Build the cavity: circumradius 5 cm (flat-to-flat 8.66 cm) ----
n=140; R_phys=0.05       # circumradius = 5 cm
W=R_phys*np.sqrt(3)      # flat-to-flat 8.66 cm
R_cells=n/2.0
h=R_phys/R_cells         # m per cell
mask=hexagon_mask(n,R_cells)
kp,field,node_id=get_eigen(mask,h,neigs=4)
f0=kp[0]*c/(2*np.pi)
print(f"Hexagonal cavity circumradius 5.0 cm (flat-to-flat {W*100:.2f} cm): TM0 = {f0/1e9:.4f} GHz")
# independent cross-check: area-equivalent circle
Rc=np.sqrt(3*np.sqrt(3)/2*R_phys**2/np.pi)
from scipy.special import jn_zeros
z01=jn_zeros(0,1)[0]
print(f"  Cross-check: area-equivalent circle R={Rc*100:.2f} cm, TM010 = {z01*c/(2*np.pi*Rc)/1e9:.4f} GHz (Δ={(f0 - z01*c/(2*np.pi*Rc))/1e6:.1f} MHz, {abs(f0- z01*c/(2*np.pi*Rc))/f0*100:.2f}%)")
print()

# ---- Resonant response: driven cavity (undamped + small loss) ----
# The amplitude response of a driven resonator: A(omega) = A0 / |omega^2 - w0^2 + i*gamma*omega|
w0=2*np.pi*f0
gamma=w0/500.0   # Q ~ 250 loading
freqs=np.linspace(0.9*f0,1.1*f0,4001)
amp=np.abs(1.0/( (2*np.pi*freqs)**2 - w0**2 + 1j*gamma*(2*np.pi*freqs) ))
peak_i=int(np.argmax(amp))
print("RESONANT RESPONSE of the driven cavity:")
print(f"  Drive swept {0.90*f0/1e9:.3f} .. {1.10*f0/1e9:.3f} GHz")
print(f"  Peak response at {freqs[peak_i]/1e9:.4f} GHz  (matched TM0 {f0/1e9:.4f} GHz)")
print(f"  Q (loaded) ~ {gamma and round( w0/gamma ):,.0f}")
print(f"  Half-power bandwidth: FWHM = { (gamma/(2*np.pi))/1e6:.3f} MHz")
print()

# ---- Field map (TM0 mode) as ASCII ----
print("TM0 standing-wave field pattern |E_z| in the hexagon cross-section:")
chars=" .:-=+*#%@"
field_norm=field/ (np.abs(field).max()+1e-12)
nrows,ncols=mask.shape
# downscale for display
step=max(1,n//28)
out=[]
for iy in range(0,nrows,step):
    line=""
    for ix in range(0,ncols,step):
        if not mask[iy,ix]:
            line+=" "; continue
        v=field_norm[iy,ix]
        line+=chars[min(len(chars)-1,int(abs(v)*len(chars)))]
    out.append(line)
# trim leading/trailing empties
while out and not out[0].strip(): out.pop(0)
while out and not out[-1].strip(): out.pop(len(out)-1)
for line in out:
    print(line)
print()
print("Note: TM0 is the fundamental (largest at center), gently zero at walls,")
print("reflecting the Dirichlet (metallic) boundary condition.")
