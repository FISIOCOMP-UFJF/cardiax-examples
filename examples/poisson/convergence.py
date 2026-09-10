#!/usr/bin/env python3
"""
convergence.py -- Poisson convergence study driver.

Reads ONE base .toml, and for a list of mesh resolutions rewrites only the
`mesh = "..."` line in memory, runs the solver, parses the printed L2 error,
and produces a table + a log-log plot with a reference slope-2 line.

No extra .toml files are written to disk (a temp file is used per run and
removed). Mesh h is inferred from the resolution N as h = 1/N.

Usage
-----
    python convergence.py --base input_unit_quad.toml \
        --solver ../../../cardiax/build/app/poisson \
        --family quad

    python convergence.py --base input_unit_cube.toml \
        --solver ../../../cardiax/build/app/poisson \
        --family cube
        
    python convergence.py --base input_unit_quad.toml --solver ../../build/app/poisson --family quad --csv conv_quad.csv
python convergence.py --base input_unit_cube.toml --solver ../../build/app/poisson --family cube --csv conv_cube.csv

Family presets
--------------
    quad : unit_quad_{NN}.xml    with N in 8,16,32,64
    cube : unit_cube_hex{NN}.xml with N in 4,8,16,32
You can override the list with --sizes 8,16,32.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np


# --- how each family maps a resolution N to a mesh filename ------------------
FAMILIES = {
    "quad": dict(sizes=[8, 16, 32, 64],
                 pattern=lambda n: f"unit_quad_{n:02d}.xml"),
    "cube": dict(sizes=[4, 8, 16, 32],
                 pattern=lambda n: f"unit_cube_hex{n:02d}.xml"),
}

# matches the printed line:  L2 error norm: 1.23456e-03
ERROR_RE = re.compile(r"L2\s+error\s+norm\s*:\s*([0-9.eE+\-]+)")

# matches the mesh line in the toml (keeps indentation/spacing agnostic)
MESH_LINE_RE = re.compile(r'^\s*mesh\s*=.*$', re.MULTILINE)


def rewrite_mesh(base_text: str, mesh_name: str) -> str:
    """Replace the ACTIVE mesh= line; leave commented #mesh= lines alone."""
    # Only the first non-comment mesh= line is the active one. Replace it.
    def repl(m):
        line = m.group(0)
        if line.lstrip().startswith("#"):
            return line                      # keep commented alternatives
        return f'mesh  = "{mesh_name}"'
    return MESH_LINE_RE.sub(repl, base_text, count=0)


def run_case(solver: str, base_text: str, mesh_name: str, workdir: str) -> float:
    """Write a temp toml with the given mesh, run the solver, parse L2 error."""
    text = rewrite_mesh(base_text, mesh_name)

    with tempfile.NamedTemporaryFile("w", suffix=".toml", dir=workdir,
                                     delete=False) as tf:
        tf.write(text)
        tmp_path = tf.name

    try:
        proc = subprocess.run([solver, "-f", tmp_path],
                              cwd=workdir, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        m = ERROR_RE.search(out)
        if not m:
            print(f"   !! could not find L2 error in output for {mesh_name}")
            print("   --- solver output (tail) ---")
            print("\n".join(out.splitlines()[-15:]))
            return float("nan")
        return float(m.group(1))
    finally:
        os.unlink(tmp_path)


def observed_orders(h, e):
    """p_i = log(e_i/e_{i-1}) / log(h_i/h_{i-1})"""
    h = np.asarray(h); e = np.asarray(e)
    p = np.full_like(e, np.nan)
    p[1:] = np.log(e[1:] / e[:-1]) / np.log(h[1:] / h[:-1])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base .toml file")
    ap.add_argument("--solver", required=True, help="path to poisson executable")
    ap.add_argument("--family", choices=FAMILIES.keys(), required=True)
    ap.add_argument("--sizes", help="comma list to override, e.g. 8,16,32")
    ap.add_argument("--workdir", default=None,
                    help="dir to run in (default: dir of --base)")
    ap.add_argument("--out", default=None, help="plot filename (png/pdf)")
    ap.add_argument("--csv", default=None, help="optional CSV output")
    args = ap.parse_args()

    fam = FAMILIES[args.family]
    sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes
             else fam["sizes"])

    with open(args.base) as f:
        base_text = f.read()

    workdir = args.workdir or (os.path.dirname(os.path.abspath(args.base)) or ".")
    solver = os.path.abspath(args.solver)

    hs, errs, names = [], [], []
    print(f"\n Convergence study ({args.family}), base = {args.base}\n")
    for n in sizes:
        mesh = fam["pattern"](n)
        e = run_case(solver, base_text, mesh, workdir)
        h = 1.0 / n
        hs.append(h); errs.append(e); names.append(mesh)
        print(f"   N={n:>4d}  h={h:.6f}  mesh={mesh:<24s}  L2={e:.6e}")

    hs = np.array(hs); errs = np.array(errs)
    p = observed_orders(hs, errs)

    # ---- table ----
    print("\n" + "=" * 66)
    print(f"{'N':>6} {'h':>12} {'L2 error':>16} {'order p':>10}")
    print("-" * 66)
    for n, h, e, pi in zip(sizes, hs, errs, p):
        ps = "   --  " if np.isnan(pi) else f"{pi:7.3f}"
        print(f"{n:>6d} {h:>12.6f} {e:>16.6e} {ps:>10}")
    print("=" * 66)

    # least-squares slope over all points (log-log)
    good = np.isfinite(errs) & (errs > 0)
    if good.sum() >= 2:
        slope, intercept = np.polyfit(np.log(hs[good]), np.log(errs[good]), 1)
        print(f" Least-squares slope (fitted order): {slope:.4f}\n")
    else:
        slope = float("nan")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N", "h", "L2_error", "order_p"])
            for n, h, e, pi in zip(sizes, hs, errs, p):
                w.writerow([n, h, e, "" if np.isnan(pi) else pi])
        print(f" wrote {args.csv}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(hs, errs, "o-", label=f"L2 error ({args.family})", zorder=3)

        # reference slope-2 line, anchored at the finest-mesh point
        h_ref = np.array([hs.min(), hs.max()])
        c = errs[good][np.argmin(hs[good])] / (hs[good].min() ** 2)
        ax.loglog(h_ref, c * h_ref ** 2, "k--",
                  label="slope 2 (reference)", zorder=2)

        ax.set_xlabel("mesh size  h")
        ax.set_ylabel(r"$\|u-u_h\|_{L^2}$")
        ax.set_title(f"Poisson convergence — {args.family}")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()

        out = args.out or f"convergence_{args.family}.png"
        fig.savefig(out, dpi=150)
        print(f" wrote {out}")
    except ImportError:
        print(" matplotlib not available; skipped plot (table/CSV still valid)")


if __name__ == "__main__":
    main()
