#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D Maxwell cavity eigen-solver using scikit-fem (Nedelec edge elements).

    curl x curl E = k0^2 E  in box,  n x E = 0  on walls (PEC)

Weak form -> generalized eigenproblem  A x = lambda B x, lambda = k0^2:
    A = int curl(E).curl(v)      (curl-curl stiffness)
    B = int E.v                  (mass/identity)

PEC: condense out the boundary EDGE DOFs (tangential E on the walls).

Validated against the rectangular box analytic TM/TE spectrum.
"""
import numpy as np
import skfem
from skfem import (
    MeshHex, MeshTet, ElementTetN1, Basis, BilinearForm, asm
)
from skfem.helpers import dot, curl

c0 = 299792458.0

@BilinearForm
def stiff(u, v, w):
    return dot(curl(u), curl(v))

@BilinearForm
def mass(u, v, w):
    return dot(u, v)


def box_modes(nx, ny, nz, a, b, d, nmodes=10):
    """Assemble A (curl-curl), B (mass) and return free (non-PEC) DOF indices."""
    mesh = MeshHex.init_tensor(
        np.linspace(0, a, nx+1),
        np.linspace(0, b, ny+1),
        np.linspace(0, d, nz+1)
    ).to_meshtet()
    basis = Basis(mesh, ElementTetN1())
    A = asm(stiff, basis)
    B = asm(mass, basis)
    # PEC: boundary edges have tangential E; condense them out
    bfacets = mesh.boundary_facets()
    bdofs = basis.get_dofs(facets=bfacets)
    b_idx = np.sort(np.unique(np.asarray(bdofs.all()).ravel()))
    all_dofs = np.arange(A.shape[0])
    free = np.setdiff1d(all_dofs, b_idx)
    return A, B, mesh, basis, free


if __name__ == "__main__":
    import numpy as np
    from scipy.linalg import eigh
    a, b, d = 0.30, 0.20, 0.25
    A, B, mesh, basis, free = box_modes(8, 6, 7, a, b, d)
    Aff = A[free][:, free].toarray(); Bff = B[free][:, free].toarray()
    lam, X = eigh(Aff, Bff)
    f = np.sort(np.sqrt(np.maximum(lam, 0))*c0/(2*np.pi))
    fs = f[f > 1e8][:6]
    an = []
    for m in range(5):
        for n in range(5):
            for p in range(5):
                if m+n+p < 2:
                    continue
                an.append(c0/2*np.sqrt((m/a)**2+(n/b)**2+(p/d)**2))
    an = sorted(set(round(x,1) for x in an))[:6]
    print("analytic:", [f"{x/1e9:.3f}" for x in an])
    print("numeric :", [f"{x/1e9:.3f}" for x in fs])
    e = [100*abs(fs[i]-an[i])/an[i] for i in range(6)]
    print(f"max rel err: {max(e):.2f}%")
