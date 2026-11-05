#!/usr/bin/env python3
"""
Gera ta_bestel_pulse.dat: a mesma Ta(t) do complete_cycle.py (Bestel), para o
-ta_curve do cardiac_cycle. Um periodo, t [ms] e Ta [Pa].

Rodar no ambiente fenicsx-env (precisa de circulation e scipy):
    python gera_ta_curve.py
"""
from pathlib import Path

import numpy as np
from circulation import bestel
from scipy.integrate import solve_ivp

# --- identico ao complete_cycle.py ---
PERIOD = 0.8          # s
DT = 1e-3             # s
T_MAX = 100.0         # kPa
T_END_DIASTOLE = 0.12 # s, lv_params.t_end_diastole

activation_model = bestel.BestelActivation(
    parameters={"t_sys": T_END_DIASTOLE, "t_dias": T_END_DIASTOLE + 0.28},
)
beat_times = np.arange(0.0, PERIOD + DT / 2, DT)
activation_shape = solve_ivp(
    activation_model, [0.0, PERIOD], [0.0],
    t_eval=beat_times, method="Radau", max_step=DT,
).y[0]
activation_shape /= activation_shape.max()
# -------------------------------------

out = Path(__file__).resolve().parent / "ta_bestel_pulse.dat"
np.savetxt(
    out,
    np.column_stack([beat_times * 1e3, T_MAX * 1e3 * activation_shape]),
    fmt="%.6e",
    header=("Ta prescrita do complete_cycle.py (Bestel, t_sys = 120 ms, t_dias = 400 ms,"
            " pico 100 kPa)\nuso: cardiac_cycle ... -ta_curve ta_bestel_pulse.dat\n"
            "t[ms] Ta[Pa]   (repetida a cada periodo)"),
)
print(f"{out}: {len(beat_times)} pontos, pico {T_MAX * 1e3:.0f} Pa em "
      f"{beat_times[activation_shape.argmax()] * 1e3:.0f} ms")
