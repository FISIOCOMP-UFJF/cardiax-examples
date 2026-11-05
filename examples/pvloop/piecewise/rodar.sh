#!/bin/bash
# cardiac_cycle -pw 1 (PV loop por fases, quase-estatico), 2 batimentos de 800 ms,
# Ta(t) prescrita (Bestel). Todos os parametros do -pw escritos explicitamente.
# Depois: python3 compara.py  (cardiax x referencia do pulse).
set -e
NUM_THREADS="${NUM_THREADS:-8}"
cardiac_cycle -f malha_descarregada_dg0.xml -num_threads "$NUM_THREADS" \
  -c Kerkoff2003 -m ExplicitEuler \
  -ta_curve ta_bestel_pulse.dat \
  -rr 800 -beats 2 -dt_mech 1 -dt 1 \
  -pw 1 \
  -pwinit 50 -pwedp 70 -pwminphase 10 \
  -pwpds 500    -pwedplv 1000 -pwpla 500 \
  -pwpds_rv 200 -pwedprv 400  -pwpra 200 \
  -pwpej    9000 -pwcart    0.011251 -pwrart    146654 -pwrc    3999.66 \
  -pwpej_rv 2500 -pwcart_rv 0.030003 -pwrart_rv  66661 -pwrc_rv 1333.22 \
  -pwfillrate 0.104 -pwfillrate_rv 0.057 \
  -pwtol 1e-3 \
  -logc 50 -dumpflags 1 \
  > saida.log 2>&1
echo "Simulacao concluida."
tail -8 saida.log
