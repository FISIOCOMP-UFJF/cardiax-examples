#!/usr/bin/env python3
"""
Troca as fibras de um XML do cardiax pelas MESMAS fibras que o complete_cycle.py
usa no problema direto: LDRB no espaco de quadratura (grau 6), levadas para a
configuracao descarregada com pulse.utils.map_vector_field(u = u_pre).

Por elemento do cardiax (P1, fibra constante no tetraedro): media dos pontos de
quadratura ponderada pelos pesos, normalizada; s e ortogonalizado em relacao a f
e n = f x s. A correspondencia celula do pulse <-> elemento do cardiax e feita
pelo centroide (os dois referenciais diferem so por uma translacao).

Rodar no ambiente fenicsx-env, de dentro de pulse_examples/:
    cd pulse_examples
    python ../piecewise_circulation/fibras_pulse.py \
        [-i malha_descarregada.xml] [-o malha_descarregada_ldrb.xml]
"""
import argparse
import re
from pathlib import Path

import dolfinx
import io4dolfinx
import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

import cardiac_geometries.geometry
import pulse

HERE = Path(__file__).resolve().parent
RESULTS = Path("results_biv_complete_cycle")
PRESTRESS = RESULTS / "prestress_biv_inverse_LV1000Pa_RV400Pa.bp"


def pulse_fibers():
    """Centroides das celulas e f0/s0 medios por celula, como no complete_cycle.py."""
    comm = MPI.COMM_WORLD
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=RESULTS / "geometry")
    geo.mesh.geometry.x[:] *= 1e-3

    V = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 2, (3,)))
    u_pre = dolfinx.fem.Function(V)
    io4dolfinx.read_function(PRESTRESS, u_pre, time=0.0, name="u_pre")

    # mesma ordem do complete_cycle.py: deforma a malha e entao mapeia as fibras
    geometry = pulse.HeartGeometry.from_cardiac_geometries(geo, metadata={"quadrature_degree": 6})
    geometry.deform(u_pre)
    f0 = pulse.utils.map_vector_field(f=geo.f0, u=u_pre, normalize=True, name="f0_unloaded")
    s0 = pulse.utils.map_vector_field(f=geo.s0, u=u_pre, normalize=True, name="s0_unloaded")

    Vq = f0.function_space
    dofs = Vq.dofmap.list                                  # (ncells, nqp)
    nqp = dofs.shape[1]
    try:   # pesos da regra de quadratura (elemento de quadratura do basix.ufl)
        w = np.asarray(Vq.ufl_element().sub_elements[0]._weights, dtype=float)
    except (AttributeError, IndexError):
        w = np.ones(nqp)
    if w.shape != (nqp,):
        w = np.ones(nqp)
    print(f"media por celula: {nqp} pontos de quadratura,"
          f" pesos {'da regra' if np.ptp(w) > 0 else 'uniformes'}")
    w = w / w.sum()

    def cell_mean(fun):
        a = fun.x.array.reshape(-1, 3)[dofs]                # (ncells, nqp, 3)
        m = np.einsum("cqd,q->cd", a, w)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    mesh = geometry.mesh
    ncells = mesh.topology.index_map(3).size_local
    cent = dolfinx.mesh.compute_midpoints(mesh, 3, np.arange(ncells, dtype=np.int32))
    return cent, cell_mean(f0), cell_mean(s0), mesh.geometry.x.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="malha_descarregada.xml")
    ap.add_argument("-o", "--output", default="malha_descarregada_ldrb.xml")
    args = ap.parse_args()

    cent_p, f_p, s_p, Xp = pulse_fibers()

    txt = (HERE / args.input).read_text()
    mesh_at = txt.index("<mesh")
    head, mesh = txt[:mesh_at], txt[mesh_at:]
    X = np.array([[float(a), float(b), float(c)] for a, b, c in
                  re.findall(r'<node id="\d+" x="([^"]*)" y="([^"]*)" z="([^"]*)" />', mesh)])
    T = np.array([[int(a) for a in m] for m in re.findall(
        r'<element id="\d+" marker="\d+" aha="\d+" v0="(\d+)" v1="(\d+)" v2="(\d+)" v3="(\d+)" />', mesh)])
    cent_c = X[T].mean(axis=1)

    # os referenciais diferem por uma translacao (o XML tem a base em x = 0)
    shift = Xp.mean(axis=0) - X.mean(axis=0)
    dist, idx = cKDTree(cent_p).query(cent_c + shift)
    assert len(set(idx)) == len(idx) == len(cent_p), "correspondencia de celulas nao e 1 para 1"
    assert dist.max() < 1e-7, f"centroides nao casam (dist max {dist.max():.2e} m)"

    f = f_p[idx]
    s = s_p[idx] - np.einsum("ij,ij->i", s_p[idx], f)[:, None] * f
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    n = np.cross(f, s)

    old = np.array([[float(v) for v in m.split(",")] for m in re.findall(r"<fiber>([^<]*)</fiber>", mesh)])
    ang = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", old, f)), 0, 1)))

    def fmt(v):
        return f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}"

    vecs = {"fiber": f, "sheet": s, "normal": n}
    a, b = mesh.index("<element_data"), mesh.index("</element_data>")

    def elem_sub(m):
        e = int(m.group(1))
        return re.sub(r"<(fiber|sheet|normal)>[^<]*</\1>",
                      lambda mv: f"<{mv.group(1)}>{fmt(vecs[mv.group(1)][e])}</{mv.group(1)}>", m.group(0))

    block = re.sub(r'<element id="(\d+)">.*?</element>', elem_sub, mesh[a:b], flags=re.S)
    (HERE / args.output).write_text(head + mesh[:a] + block + mesh[b:])

    print(f"correspondencia: {len(idx)} celulas, distancia max {dist.max():.1e} m")
    print(f"angulo entre a fibra antiga (Bayer) e a do pulse (LDRB): mediana {np.median(ang):.1f} graus,"
          f" p90 {np.percentile(ang, 90):.1f}, max {ang.max():.1f}")
    print(f"escrito {HERE / args.output}")


if __name__ == "__main__":
    main()
