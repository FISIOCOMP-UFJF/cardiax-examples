#!/usr/bin/env python3
"""
cardiax (-pw 1) x fenicsx-pulse, mesmas fibras DG0, quase-estatico.

  pulse    referencia/traces_pulse_dg0.npz
  cardiax  malha_descarregada_dg0.xml_piecewise.dat (saida do ./rodar.sh);
           sem ela, usa referencia/resultado_esperado_piecewise.dat

Uso:
    python3 compara.py [--out comparacao.png] [--beats 2]

Fases (mesma numeracao nos dois): 0 PRELOAD, 1 IVC, 2 EJECTION, 3 IVR, 4 FILLING;
a fase gravada e a do passo resolvido.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MMHG = 133.322387415
PERIOD = 800.0


def load_cardiax(path):
    d = np.loadtxt(path)
    out = {"t": d[:, 0]}
    for name, k in (("LV", 1), ("RV", 6)):
        out[name] = dict(phase=d[:, k].astype(int), P=d[:, k + 1] / MMHG, V=d[:, k + 2],
                         Pc=d[:, k + 4] / MMHG)
    return out


def load_pulse(path):
    p = np.load(path)
    out = {"t": p["time"] * 1e3}
    for name in ("LV", "RV"):
        out[name] = dict(phase=p["phase_" + name].astype(int), P=p["p_" + name],
                         V=p["V_" + name], Pc=p["Pc_" + name])
    return out


def indicators(run, name, beat):
    w = (run["t"] > beat * PERIOD + 1e-9) & (run["t"] <= (beat + 1) * PERIOD + 1e-9)
    c = {k: v[w] for k, v in run[name].items()}
    t = run["t"][w]
    if t.size == 0:
        return None

    def first(ph):
        i = np.flatnonzero(c["phase"] == ph)
        return i[0] if i.size else None
    i_ivc, i_ej, i_ivr, i_fill = first(1), first(2), first(3), first(4)
    edv = c["V"][i_ivc] if i_ivc is not None else np.nan
    esv = c["V"][i_ivr] if i_ivr is not None else np.nan
    g = lambda i, a: a[i] if i is not None else np.nan
    return dict(EDV=edv, ESV=esv, SV=edv - esv, EF=100 * (edv - esv) / edv,
                Pmax=c["P"].max(), t_Pmax=t[c["P"].argmax()],
                t_open=g(i_ej, t), P_open=g(i_ej, c["P"]),
                t_close=g(i_ivr, t), t_fill=g(i_fill, t), Pmin=c["P"].min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="comparacao.png")
    ap.add_argument("--beats", type=int, default=2)
    a = ap.parse_args()

    cx = "malha_descarregada_dg0.xml_piecewise.dat"
    if not os.path.exists(cx):
        cx = "referencia/resultado_esperado_piecewise.dat"
        print(f"(sem saida do ./rodar.sh; usando {cx})")
    runs = [("pulse", load_pulse("referencia/traces_pulse_dg0.npz"), dict(color="tab:blue", lw=1.8)),
            ("cardiax", load_cardiax(cx), dict(color="tab:red", lw=1.2, ls="--"))]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for row, name in enumerate(("LV", "RV")):
        for label, r, st in runs:
            c = r[name]
            ax[row, 0].plot(c["V"], c["P"], label=label, **st)
            ax[row, 1].plot(r["t"], c["P"], label=label, **st)
            ax[row, 1].plot(r["t"], c["Pc"], alpha=0.5, lw=0.8, color=st["color"], ls=":")
            ax[row, 2].plot(r["t"], c["V"], label=label, **st)
        ax[row, 0].set(xlabel="V [mL]", ylabel="P [mmHg]", title=f"{name}: laco PV")
        ax[row, 1].set(xlabel="t [ms]", ylabel="P [mmHg]", title=f"{name}: pressao (pontilhado: P_c)")
        ax[row, 2].set(xlabel="t [ms]", ylabel="V [mL]", title=f"{name}: volume")
        ax[row, 0].legend(frameon=False, fontsize=8)
    fig.savefig(a.out, dpi=130)
    print(f"figura: {a.out}\n")

    keys = ["EDV", "ESV", "SV", "EF", "Pmax", "t_Pmax", "t_open", "P_open", "t_close", "t_fill", "Pmin"]
    for name in ("LV", "RV"):
        for b in range(a.beats):
            print(f"{name}, batimento {b + 1}")
            print(f"  {'':10s}" + "".join(f"{k:>9s}" for k in keys))
            for label, r, _ in runs:
                ind = indicators(r, name, b)
                if ind is not None:
                    print(f"  {label:10s}" + "".join(f"{ind[k]:9.2f}" for k in keys))
        print()
    print("unidades: mL, %, mmHg, ms")


if __name__ == "__main__":
    main()
