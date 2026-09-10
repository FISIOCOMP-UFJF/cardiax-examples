#!/usr/bin/env python3
"""
convergence.py -- Poisson convergence study with on-the-fly mesh generation.

Meshes are generated (never stored) via scripts/mesh/unit_cube_poisson.py.
Supports all four element types, grouped by dimension:

    2D : quad, tri   -> base input_unit_quad.toml  (marker 7)
    3D : hex,  tet    -> base input_unit_cube.toml  (marker 27)

Within a dimension the PDE, exact solution and boundary marker are identical,
so tri reuses the quad base and tet reuses the cube base -- only the element
type (and thus the generated mesh) changes.

Run ONE element type:
    python convergence.py --type quad --base input_unit_quad.toml \
        --solver ../../../cardiax/build/app/poisson  \
        --generator ../../../cardiax/scripts/mesh/unit_cube_poisson.py \
        --csv conv_2d.csv

Run SEVERAL types on one plot (they must share a base toml / dimension):
    python convergence.py --types quad,tri --base input_unit_quad.toml ...
    python convergence.py --types hex,tet  --base input_unit_cube.toml ...

Sizes default to 4,8,16,32,64 (override with --sizes 8,16,32).
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ERROR_RE = re.compile(r"L2\s+error\s+norm\s*:\s*([0-9.eE+\-]+)")
MESH_LINE_RE = re.compile(r'^(?P<indent>\s*)mesh\s*=.*$', re.MULTILINE)

DEFAULT_SIZES = [4, 8, 16, 32, 64]

# which base toml each type belongs to, by dimension (for a sanity check)
TYPE_DIM = {"quad": 2, "tri": 2, "hex": 3, "tet": 3}


def rewrite_mesh(base_text, mesh_path):
    def repl(m):
        line = m.group(0)
        if line.lstrip().startswith("#"):
            return line
        return f'{m.group("indent")}mesh  = "{mesh_path}"'
    return MESH_LINE_RE.sub(repl, base_text)


def generate_mesh(generator, etype, n, out_dir):
    fname = f"unit_{etype}_n{n}_{os.getpid()}.xml"
    cmd = [sys.executable, generator, "--type", etype, "--n", str(n),
           "--out", fname, "--out_dir", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    path = os.path.join(out_dir, fname)
    if proc.returncode != 0 or not os.path.exists(path):
        print(f"   !! mesh generation failed for {etype} n={n}")
        print("\n".join((proc.stdout + proc.stderr).splitlines()[-15:]))
        return None
    return path


def run_case(solver, base_text, mesh_path, run_dir):
    text = rewrite_mesh(base_text, mesh_path)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", dir=run_dir,
                                     delete=False) as tf:
        tf.write(text)
        tmp_toml = tf.name
    try:
        proc = subprocess.run([solver, "-f", tmp_toml],
                              cwd=run_dir, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        m = ERROR_RE.search(out)
        if not m:
            print(f"   !! no L2 error for {os.path.basename(mesh_path)}")
            print("\n".join(out.splitlines()[-15:]))
            return float("nan")
        return float(m.group(1))
    finally:
        os.unlink(tmp_toml)


def observed_orders(h, e):
    h = np.asarray(h); e = np.asarray(e)
    p = np.full_like(e, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        p[1:] = np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])
    return p


def sweep_type(etype, sizes, generator, solver, base_text, run_dir,
               keep_meshes):
    mesh_dir = tempfile.mkdtemp(prefix=f"conv_{etype}_")
    hs, errs = [], []
    print(f"\n --- element type: {etype} ---")
    try:
        for n in sizes:
            mp = generate_mesh(generator, etype, n, mesh_dir)
            if mp is None:
                hs.append(1.0 / n); errs.append(float("nan")); continue
            e = run_case(solver, base_text, mp, run_dir)
            hs.append(1.0 / n); errs.append(e)
            print(f"   n={n:>4d}  h={1.0/n:.6f}  L2={e:.6e}")
            if not keep_meshes:
                os.unlink(mp)
    finally:
        if not keep_meshes:
            try: os.rmdir(mesh_dir)
            except OSError: pass
    return np.array(hs), np.array(errs)


def print_table(etype, sizes, hs, errs):
    p = observed_orders(hs, errs)
    print("\n" + "=" * 70)
    print(f" {etype}")
    print(f"{'n':>6} {'h':>12} {'L2 error':>16} {'order p':>10}")
    print("-" * 70)
    for n, h, e, pi in zip(sizes, hs, errs, p):
        ps = "   --  " if np.isnan(pi) else f"{pi:7.3f}"
        print(f"{n:>6d} {h:>12.6f} {e:>16.6e} {ps:>10}")
    good = np.isfinite(errs) & (errs > 0)
    if good.sum() >= 2:
        slope, _ = np.polyfit(np.log(hs[good]), np.log(errs[good]), 1)
        print(f" fitted order: {slope:.4f}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--type", help="single element type: quad|tri|hex|tet")
    g.add_argument("--types", help="comma list on one plot, e.g. quad,tri or hex,tet")
    ap.add_argument("--base", required=True, help="base .toml (must match the dimension)")
    ap.add_argument("--solver", required=True)
    ap.add_argument("--generator", required=True, help="unit_cube_poisson.py")
    ap.add_argument("--sizes", help="comma list (default 4,8,16,32,64)")
    ap.add_argument("--out", default=None, help="plot filename")
    ap.add_argument("--csv", default=None, help="CSV output (one file, all types)")
    ap.add_argument("--keep-meshes", action="store_true")
    args = ap.parse_args()

    types = [args.type] if args.type else [t.strip() for t in args.types.split(",")]

    for t in types:
        if t not in TYPE_DIM:
            sys.exit(f" unknown type '{t}' (expected quad|tri|hex|tet)")
    dims = {TYPE_DIM[t] for t in types}
    if len(dims) > 1:
        sys.exit(" cannot mix 2D and 3D types on one run/plot; "
                 "they need different base tomls")

    sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes
             else DEFAULT_SIZES)

    with open(args.base) as f:
        base_text = f.read()
    run_dir = os.path.dirname(os.path.abspath(args.base)) or "."
    solver = os.path.abspath(args.solver)
    generator = os.path.abspath(args.generator)
    if not os.path.exists(generator):
        sys.exit(f" generator not found: {generator}")

    results = {}
    for t in types:
        hs, errs = sweep_type(t, sizes, generator, solver, base_text,
                              run_dir, args.keep_meshes)
        results[t] = (hs, errs)
        print_table(t, sizes, hs, errs)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["type", "n", "h", "L2_error", "order_p"])
            for t in types:
                hs, errs = results[t]
                p = observed_orders(hs, errs)
                for n, h, e, pi in zip(sizes, hs, errs, p):
                    w.writerow([t, n, h, e, "" if np.isnan(pi) else pi])
        print(f"\n wrote {args.csv}")

    # ---- combined plot ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 5))
        markers = {"quad": "o", "tri": "s", "hex": "o", "tet": "s"}
        all_h, all_e = [], []
        for t in types:
            hs, errs = results[t]
            good = np.isfinite(errs) & (errs > 0)
            ax.loglog(hs[good], errs[good], markers.get(t, "o") + "-",
                      label=f"{t}", zorder=3)
            all_h.append(hs[good]); all_e.append(errs[good])

        H = np.concatenate(all_h); E = np.concatenate(all_e)
        h_ref = np.array([H.min(), H.max()])
        c = E[np.argmin(H)] / (H.min() ** 2)
        ax.loglog(h_ref, c * h_ref ** 2, "k--", label="slope 2", zorder=2)

        dim = "2D" if dims == {2} else "3D"
        ax.set_xlabel("mesh size  h"); ax.set_ylabel(r"$\|u-u_h\|_{L^2}$")
        ax.set_title(f"Poisson convergence — {dim} ({', '.join(types)})")
        ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend()
        fig.tight_layout()
        out = args.out or f"convergence_{'_'.join(types)}.png"
        fig.savefig(out, dpi=150); print(f" wrote {out}")
    except ImportError:
        print(" matplotlib unavailable; skipped plot")


if __name__ == "__main__":
    main()
