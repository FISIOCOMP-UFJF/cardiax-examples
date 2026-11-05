# PV loop por fases (`cardiac_cycle -pw 1`) x fenicsx-pulse

Ciclo cardiaco biventricular quase-estatico, 2 batimentos de 800 ms, com o
controlador de cinco fases do `-pw` (PRELOAD, IVC, EJECTION em Windkessel de 3
elementos, IVR, FILLING) e Ta(t) prescrita (`ta_bestel_pulse.dat`, curva de Bestel).
A referencia e o mesmo problema rodado no fenicsx-pulse, com a mesma malha, as
mesmas fibras (DG0, uma por elemento) e a mesma configuracao de referencia.

## Rodar

```bash
./rodar.sh            # ~13 min com 8 threads (NUM_THREADS=n para mudar)
python3 compara.py    # comparacao.png e tabela por batimento
```

## Conteudo

| arquivo | o que e |
|---|---|
| `malha_descarregada_dg0.xml` | malha descarregada (u_pre do pulse), fibras DG0 do pulse, HO com kappa 5e5, mola de 1e5 Pa/m no epicardio, u_x = 0 na base |
| `ta_bestel_pulse.dat` | Ta(t) de um periodo [ms, Pa] (`scripts/gera_ta_curve.py`) |
| `rodar.sh` | a rodada, com todos os parametros do `-pw` explicitos |
| `compara.py` | cardiax x pulse: lacos PV, P(t), V(t) e indicadores |
| `referencia/traces_pulse_dg0.npz`, `pulse_dg0.log` | resultado do pulse |
| `referencia/resultado_esperado_piecewise.dat`, `comparacao_esperada.png` | resultado esperado do cardiax |
| `pulse/complete_cycle_p1_static_dg0.py` | script do pulse que gera a referencia |
| `scripts/` | geracao da malha e da Ta a partir dos resultados do pulse |

## Resultado esperado (batimento 2; mL, mmHg)

|         | VE EDV | VE ESV | VE Pmax | VD EDV | VD ESV | VD Pmax |
|---------|--------|--------|---------|--------|--------|---------|
| pulse   | 107.63 | 66.52  | 81.5    | 95.56  | 67.17  | 23.0    |
| cardiax | 107.76 | 66.47  | 81.6    | 96.18  | 67.34  | 23.1    |

Tempos das fases ate 1 ms de diferenca. Fica uma diferenca passiva pequena no EDV
do VD (~0.6 mL); a mola do epicardio do pulse age na normal atual, a do cardiax na
de referencia.

## Refazer a referencia e a malha (ambiente `fenicsx-env`)

1. `cd pulse && python complete_cycle_p1_static_dg0.py` (~18 min): usa a geometria de
   `~/Documentos/pulse_examples/results_biv_complete_cycle/geometry`, refaz as fibras
   em DG0 e o prestress, e grava em `pulse/results_dg0/`.
2. Com `R=~/Documentos/pulse_examples/results_biv_complete_cycle`:
   `python scripts/unload_mesh.py --results pulse/results_dg0 --geometry-ref $R/geometry --xml-in $R/malha.xml -o malha_descarregada.xml`
   (`--geometry-ref` casa os nos por coordenada; a geometria regravada tem outra numeracao).
3. `python scripts/fibras_pulse.py --results pulse/results_dg0 -i malha_descarregada.xml -o malha_descarregada_dg0.xml`
4. `cp pulse/results_dg0/traces_p1_static_dg0.npz referencia/traces_pulse_dg0.npz`
