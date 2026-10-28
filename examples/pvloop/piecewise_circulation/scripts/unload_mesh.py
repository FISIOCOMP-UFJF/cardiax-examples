#!/usr/bin/env python3
"""
Gera malha_descarregada.xml a partir de malha.xml, reproduzindo no cardiax a
configuracao de referencia usada pelo complete_cycle.py (fenicsx-pulse):

  1. Nos: X_descarregado = X_ED + u_pre, com u_pre lido do prestress do pulse
     (prestress_biv_inverse_LV1000Pa_RV400Pa.bp, P2) e girado de volta para o
     referencial do malha.xml (o pulse gira a malha para a normal da base = x).
  2. Fibras: f -> F f / |F f|, com F = I + grad(u_pre), como
     pulse.utils.map_vector_field. Feito para fiber/sheet/normal e long/circ/rad,
     por elemento (F constante no tetraedro P1).
  3. Material: Holzapfel-Ogden transversalmente isotropico do pulse
     (a = 2280 Pa, b = 9.726, a_f = 1685 Pa, b_f = 15.779), kappa = --kappa [Pa].
  4. Mola do epicardio (marker 40): 1e5 Pa/m, como o RobinBC do pulse.
  5. Referencial do pulse: a malha (nos e fibras) e girada para a normal da base
     = x e transladada para a base ficar no plano x = 0. O cardiax mede o volume
     com (1/3) int x.n so no endocardio, aberto na base; com a origem no plano
     da base a tampa nao contribui e o volume sai exato.
  6. Base deslizante: so u_x = 0 nos nos da base (no pulse, dirichlet em V.sub(0)).

Roda no ambiente fenicsx-env, de dentro de pulse_examples/:
    cd pulse_examples
    python ../piecewise_circulation/unload_mesh.py [--kappa 5e5] [-o malha_descarregada.xml]
"""
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI

import cardiac_geometries.geometry
import dolfinx
import io4dolfinx

HERE = Path(__file__).resolve().parent
RESULTS = Path("results_biv_complete_cycle")
PRESTRESS = RESULTS / "prestress_biv_inverse_LV1000Pa_RV400Pa.bp"
XML_IN = HERE / "malha.xml"

# a, a_f, a_s, a_fs, b, b_f, b_s, b_fs, (ativo), kappa  -- ordem do HolzapfelOgden do cardiax
HO_COEFFS = "2280.0, 1685.0, 0.0, 0.0, 9.726, 15.779, 0.0, 0.0, 0, {kappa:.1f}"
K_EPI = 1.0e5  # Pa/m
MARKER_BASE = 10


def pulse_displacement():
    """u_pre nos vertices, na ordem do mesh.h5 (= ordem do malha.xml), referencial original [m]."""
    geo = cardiac_geometries.geometry.Geometry.from_folder(
        comm=MPI.COMM_WORLD, folder=RESULTS / "geometry",
    )
    geo.mesh.geometry.x[:] *= 1e-3
    Y = geo.mesh.geometry.x.copy()
    idx = np.asarray(geo.mesh.geometry.input_global_indices)
    X = h5py.File(RESULTS / "geometry/mesh.h5", "r")["/Mesh/Mesh/geometry"][()] * 1e-3

    # rotacao rigida original -> girado (Kabsch), com correspondencia exata pelos indices
    A, B = X[idx], Y
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T                       # B - cb = (A - ca) @ R.T
    res = np.abs((A - ca) @ R.T - (B - cb)).max()
    assert res < 1e-10, f"malhas nao correspondem (residuo {res:.2e} m)"

    V2 = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 2, (3,)))
    u2 = dolfinx.fem.Function(V2)
    io4dolfinx.read_function(PRESTRESS, u2, time=0.0, name="u_pre")
    V1 = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1, (3,)))
    u1 = dolfinx.fem.Function(V1)
    u1.interpolate(u2)                       # exato nos vertices

    # dof P1 -> no da geometria (mesma posicao)
    dof_x = V1.tabulate_dof_coordinates()
    order = {tuple(np.round(p, 12)): i for i, p in enumerate(dof_x)}
    u_rot = np.empty_like(Y)
    vals = u1.x.array.reshape(-1, 3)
    for g, p in enumerate(Y):
        u_rot[g] = vals[order[tuple(np.round(p, 12))]]

    u = np.empty_like(X)
    u[idx] = u_rot @ R                       # volta ao referencial original: R^T u
    return X, u, R


def deformation_gradients(X, u, T):
    """F = I + grad u por tetraedro (P1), grad em relacao a X."""
    v, w = X[T], u[T]
    dX = np.stack([v[:, k] - v[:, 0] for k in (1, 2, 3)], axis=2)   # [m,3,3] colunas
    dU = np.stack([w[:, k] - w[:, 0] for k in (1, 2, 3)], axis=2)
    return np.eye(3) + dU @ np.linalg.inv(dX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=5.0e5, help="kappa do HO [Pa]")
    ap.add_argument("-o", "--output", default="malha_descarregada.xml")
    args = ap.parse_args()
    xml_out = HERE / args.output

    X, u, R = pulse_displacement()
    txt = XML_IN.read_text()

    # nos da base
    bnd = txt[txt.index("<boundary"):]
    base = sorted({int(v) for m, *vs in re.findall(
        r'marker="(\d+)" v0="(\d+)" v1="(\d+)" v2="(\d+)"', bnd) if int(m) == MARKER_BASE for v in vs})

    # --- nos: descarrega, gira para o referencial do pulse, base em x = 0 ---
    Xn = (X + u) @ R.T
    Xn[:, 0] -= Xn[base, 0].mean()
    assert np.abs(Xn[base, 0]).max() < 1e-7, "base nao ficou plana em x = 0"
    def node_sub(m):
        i = int(m.group(1))
        x, y, z = Xn[i]
        return f'<node id="{i}" x="{x:.8f}" y="{y:.8f}" z="{z:.8f}" />'
    mesh_at = txt.index("<mesh")
    head, mesh = txt[:mesh_at], txt[mesh_at:]
    mesh, n = re.subn(r'<node id="(\d+)" x="[^"]*" y="[^"]*" z="[^"]*" />', node_sub, mesh)
    assert n == len(X), (n, len(X))

    # --- fibras ---
    T = np.array([[int(a) for a in m] for m in re.findall(
        r'<element id="\d+" marker="\d+" aha="\d+" v0="(\d+)" v1="(\d+)" v2="(\d+)" v3="(\d+)" />', mesh)])
    F = R @ deformation_gradients(X, u, T)   # mapeia e ja gira para o referencial do pulse

    def map_block(block):
        def elem_sub(m):
            e = int(m.group(1))
            def vec_sub(mv):
                a = np.array([float(c) for c in mv.group(2).split(",")])
                b = F[e] @ a
                b /= np.linalg.norm(b)
                return f"<{mv.group(1)}>{b[0]:.6f},{b[1]:.6f},{b[2]:.6f}</{mv.group(1)}>"
            return re.sub(r"<(fiber|sheet|normal|long|circ|rad)>([^<]*)</\1>", vec_sub, m.group(0))
        return re.sub(r'<element id="(\d+)">.*?</element>', elem_sub, block, flags=re.S)

    for tag in ("element_data", "local_basis"):
        a = mesh.index(f"<{tag}")
        b = mesh.index(f"</{tag}>")
        mesh = mesh[:a] + map_block(mesh[a:b]) + mesh[b:]

    # --- material e mola ---
    head = re.sub(r'type="Guccione"', 'type="HolzapfelOgden"', head)
    head = re.sub(r"<coefficients>[^<]*</coefficients>",
                  f"<coefficients>{HO_COEFFS.format(kappa=args.kappa)}</coefficients>", head)
    head, n = re.subn(r'(<spring>\s*<node id="\d+" marker="40" value=")[^"]*"',
                      lambda m: f'{m.group(1)}{K_EPI:g}"', head)
    assert n == 1, "bloco <spring> com marker 40 nao encontrado"

    # --- base deslizante: so u_x = 0 ---
    pd = "".join(f'    <node id="{i}" direction="0" value="0.000000" />\n' for i in base)
    head, n = re.subn(r"(<prescribed_displacement>\n).*?(\s*</prescribed_displacement>)",
                      lambda m: m.group(1) + pd.rstrip("\n") + m.group(2), head, flags=re.S)
    assert n == 1

    xml_out.write_text(head + mesh)
    print(f"|u_pre| max = {np.linalg.norm(u, axis=1).max() * 1e3:.3f} mm")
    print(f"det F: min {np.linalg.det(F).min():.3f}, max {np.linalg.det(F).max():.3f}")
    print(f"kappa = {args.kappa:g} Pa, {len(base)} nos da base com u_x = 0")
    print(f"escrito {xml_out}")


if __name__ == "__main__":
    main()
