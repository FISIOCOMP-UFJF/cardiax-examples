#!/usr/bin/env python3
"""
Compara uma rodada do cardiax (-pw 1) com o pulse (complete_cycle.py), por batimento.

Uso:
    python3 compara_graficos.py <cardiax _piecewise.dat> <pulse traces.npz> <saida.png> \
        [--ref2 <traces.npz ou _piecewise.dat> --ref2-label "..."] [--title "..."] [--beats N]

Gera uma figura 2x3: PV loop, pressao x tempo e volume x tempo, VE em cima e VD
embaixo, e imprime uma tabela com os indicadores de cada batimento (--beats, padrao 1).
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MMHG = 133.322
PERIOD_MS = 800.0
T_END_MS = PERIOD_MS     # ate onde ler e plotar; --beats muda

# pulse.cycle.Phase -> fase equivalente do cardiax (0..5)
PULSE_TO_CARDIAX = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}   # PRELOAD conta como fase 1


def load_cardiax(path):
    d = np.loadtxt(path)
    t = d[:, 0]
    m = t <= T_END_MS
    out = {"t": t[m]}
    for name, k in (("LV", 1), ("RV", 6)):
        out[name] = dict(phase=d[m, k].astype(int), P=d[m, k + 1] / MMHG, V=d[m, k + 2],
                         Part=d[m, k + 4] / MMHG)
    return out


def load_pulse(path):
    p = np.load(path)
    t = p["time"] * 1e3
    m = t <= T_END_MS
    out = {"t": t[m]}
    for name in ("LV", "RV"):
        ph = np.array([PULSE_TO_CARDIAX[int(x)] for x in p["phase_" + name][m]])
        out[name] = dict(phase=ph, P=p["p_" + name][m], V=p["V_" + name][m], Part=p["Pc_" + name][m])
    return out


def indicators(run, name, beat=0):
    w = (run["t"] > beat * PERIOD_MS) & (run["t"] <= (beat + 1) * PERIOD_MS)
    c = {k: v[w] for k, v in run[name].items()}
    t = run["t"][w]
    def first(ph):
        i = np.flatnonzero(c["phase"] == ph)
        return i[0] if i.size else None
    i_ivc, i_ej, i_ivr = first(2), first(3), first(4)
    edv = c["V"][i_ivc] if i_ivc is not None else np.nan
    esv = c["V"][i_ivr] if i_ivr is not None else np.nan
    return dict(
        EDV=edv, ESV=esv, SV=edv - esv, EF=100 * (edv - esv) / edv,
        Pmax=c["P"].max(), t_Pmax=t[c["P"].argmax()],
        t_open=t[i_ej] if i_ej is not None else np.nan,
        P_open=c["P"][i_ej] if i_ej is not None else np.nan,
        t_close=t[i_ivr] if i_ivr is not None else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cardiax")
    ap.add_argument("pulse")
    ap.add_argument("output")
    ap.add_argument("--pulse-label", default="pulse P1")
    ap.add_argument("--cardiax-label", default="cardiax")
    ap.add_argument("--ref2")
    ap.add_argument("--ref2-label", default="pulse P2")
    ap.add_argument("--title", default="")
    ap.add_argument("--beats", type=int, default=1)
    args = ap.parse_args()

    global T_END_MS
    T_END_MS = args.beats * PERIOD_MS

    cx, pu = load_cardiax(args.cardiax), load_pulse(args.pulse)
    # --ref2: um traces.npz do pulse ou outro _piecewise.dat do cardiax
    r2 = None
    if args.ref2:
        r2 = load_pulse(args.ref2) if args.ref2.endswith(".npz") else load_cardiax(args.ref2)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, (name, color) in enumerate((("LV", "C0"), ("RV", "C3"))):
        tag = "VE" if name == "LV" else "VD"
        a_pv, a_p, a_v = ax[row]
        if r2 is not None:
            a_pv.plot(r2[name]["V"], r2[name]["P"], color="0.6", lw=1, ls="--", label=args.ref2_label)
            a_p.plot(r2["t"], r2[name]["P"], color="0.6", lw=1, ls="--")
            a_v.plot(r2["t"], r2[name]["V"], color="0.6", lw=1, ls="--")
        # PV loop por batimento; os anteriores ao ultimo mais claros
        for b in range(args.beats):
            last = b == args.beats - 1
            for run, kw, lab in ((pu, dict(alpha=.35 if last else .15, lw=4), args.pulse_label),
                                 (cx, dict(alpha=1.0 if last else .4, lw=1.5), args.cardiax_label)):
                w = (run["t"] > b * PERIOD_MS) & (run["t"] <= (b + 1) * PERIOD_MS)
                a_pv.plot(run[name]["V"][w], run[name]["P"][w], color=color,
                          label=f"{lab}, bat. {b + 1}" if args.beats > 1 else lab, **kw)
        a_p.plot(pu["t"], pu[name]["P"], color=color, alpha=.35, lw=4, label=args.pulse_label)
        a_p.plot(cx["t"], cx[name]["P"], color=color, lw=1.5, label=args.cardiax_label)
        a_p.plot(pu["t"], pu[name]["Part"], color="k", alpha=.35, lw=3, ls=":", label=f"P_c ({args.pulse_label})")
        a_p.plot(cx["t"], cx[name]["Part"], color="k", lw=1, ls=":", label=f"P_art ({args.cardiax_label})")
        a_v.plot(pu["t"], pu[name]["V"], color=color, alpha=.35, lw=4, label=args.pulse_label)
        a_v.plot(cx["t"], cx[name]["V"], color=color, lw=1.5, label=args.cardiax_label)

        # inicio da ejecao de cada um, em cada batimento
        for run, ls in ((pu, "-"), (cx, "--")):
            ej = run[name]["phase"] == 3
            for i in np.flatnonzero(ej[1:] & ~ej[:-1]) + 1:
                for a in (a_p, a_v):
                    a.axvline(run["t"][i], color=color, lw=0.8, ls=ls, alpha=.6)

        a_pv.set(xlabel="V [mL]", ylabel="P [mmHg]", title=f"{tag}: PV loop" + (" (1o batimento)" if args.beats == 1 else ""))
        a_p.set(xlabel="t [ms]", ylabel="P [mmHg]", title=f"{tag}: pressao", xlim=(0, T_END_MS))
        a_v.set(xlabel="t [ms]", ylabel="V [mL]", title=f"{tag}: volume (verticais = abertura da valva)",
                xlim=(0, T_END_MS))
        a_pv.legend(fontsize=8)
        a_p.legend(fontsize=7)
        for a in ax[row]:
            a.grid(alpha=.3)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=90)

    keys = [("EDV", "mL", "{:.1f}"), ("ESV", "mL", "{:.1f}"), ("SV", "mL", "{:.1f}"),
            ("EF", "%", "{:.0f}"), ("Pmax", "mmHg", "{:.1f}"), ("t_Pmax", "ms", "{:.0f}"),
            ("P_open", "mmHg", "{:.1f}"), ("t_open", "ms", "{:.0f}"), ("t_close", "ms", "{:.0f}")]
    runs = [(args.pulse_label, pu), (args.cardiax_label, cx)] + ([(args.ref2_label, r2)] if r2 else [])
    for b in range(args.beats):
        for name in ("LV", "RV"):
            print(f"\n{'VE' if name == 'LV' else 'VD'}, batimento {b + 1}:"
                  + "".join(f"{lab:>14s}" for lab, _ in runs))
            ind = [indicators(r, name, b) for _, r in runs]
            for k, unit, fmt in keys:
                print(f"  {k:7s}[{unit:4s}]" + "".join(f"{fmt.format(i[k]):>14s}" for i in ind))
    print(f"\nfigura: {args.output}")


if __name__ == "__main__":
    main()
