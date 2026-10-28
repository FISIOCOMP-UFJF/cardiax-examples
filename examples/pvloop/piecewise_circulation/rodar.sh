#!/bin/bash
# Reproduz o complete_cycle.py do fenicsx-pulse (elementos P1, 2 batimentos) com o
# cardiac_cycle -pw 1 e compara com a referencia. Ver README.md.
#
# Uso:  ./rodar.sh
# Variaveis opcionais:
#   NUM_THREADS   threads OpenMP (padrao: 8)
#
# Saidas (neste diretorio): saida.log, malha_descarregada_ldrb.xml_piecewise.dat,
# output/ (campos para o ParaView; APAGADO e recriado a cada rodada) e
# comparacao_pulse_p1.png.
set -e
NUM_THREADS="${NUM_THREADS:-8}"


cardiac_cycle -f malha_descarregada_ldrb.xml -num_threads "$NUM_THREADS" \
  -c Kerkoff2003 -m ExplicitEuler \
  -ta_curve ta_bestel_pulse.dat \
  -rr 800 -beats 2 -t 1650 -dt_mech 1 -dt 1 \
  -pw 1 \
  -pwinit 50 -pwedp 70 -pwminphase 10 \
  -pwpds 500    -pwedplv 1000 -pwpla 500 \
  -pwpds_rv 200 -pwedprv 400  -pwpra 200 \
  -pwwk 2 \
  -pwpej 9000    -pwcart 0.011251    -pwrart 146654 -pwrc 3999.66 \
  -pwpej_rv 2500 -pwcart_rv 0.030003 -pwrart_rv 66661 -pwrc_rv 1333.22 \
  -pwfillrate 0.104 -pwfillrate_rv 0.057 \
  -pwdamp 0 -pwff 1 \
  -dumpflags 1 \
  > saida.log 2>&1

echo "Simulacao concluida."
grep -E "VE: V_ED|VD: V_ED" saida.log || true

if python3 -c "import numpy, matplotlib" 2>/dev/null; then
  python3 scripts/compara_graficos.py \
    malha_descarregada_ldrb.xml_piecewise.dat referencia/traces_pulse_p1.npz \
    comparacao_pulse_p1.png --beats 2 --cardiax-label cardiax \
    --title "pulse P1 x cardiax, 2 batimentos" 2>/dev/null
else
  echo "numpy/matplotlib ausentes: comparação com pulse pulada"
fi
