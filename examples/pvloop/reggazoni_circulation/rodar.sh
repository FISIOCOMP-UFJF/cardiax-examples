#!/bin/bash
# cardiac_cycle -circ 1 com a Ta prescrita do pulse P1 (land_circulation_biv.py)
cd "$(dirname "$0")"
~/Documentos/cardiax/build/app/cardiac_cycle -f malha_biv_3mat.xml -num_threads 8 \
  -c ToRORdLand -circ 1 -beats 4 -rr 1000 -dt_mech 1 \
  -tact 0 -emd 70 -circ_vtot 1048 \
  -ta_curve ta_pulseP1_LV_bat4.dat \
  > saida.log 2>&1
