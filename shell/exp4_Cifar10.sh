#!/bin/bash
# Exp4 - Cifar10: CAMP effectiveness evaluation
# Model: cnn2, lr: 0.04, m: 8, k: 40, alpha: 0.5 (noniid, unbalance, dir)

filters="avg multi-krum median trmean flame FLDetector"
COMMON="--model cnn2 --dataset Cifar10 --lr 0.04 --ld True --exp 4 --device cuda --device_id 0"

mp="none"
for filter in ${filters}; do
    echo "Running: filter=${filter}, mp=${mp}"
    python main.py ${COMMON} --m 0 --filter ${filter} --mp ${mp} &
done
wait
echo "=== none done ==="

mp="min-max"
for filter in ${filters}; do
    echo "Running: filter=${filter}, mp=${mp}"
    python main.py ${COMMON} --m 8 --filter ${filter} --mp ${mp} &
done
wait
echo "=== min-max done ==="

mp="sign_flip"
for filter in ${filters}; do
    echo "Running: filter=${filter}, mp=${mp}"
    python main.py ${COMMON} --m 8 --filter ${filter} --mp ${mp} &
done
wait
echo "=== sign_flip done ==="

mp="LIE"
for filter in ${filters}; do
    echo "Running: filter=${filter}, mp=${mp}"
    python main.py ${COMMON} --m 8 --filter ${filter} --mp ${mp} &
done
wait
echo "=== LIE done ==="

mp="MPAF"
for filter in ${filters}; do
    echo "Running: filter=${filter}, mp=${mp}"
    python main.py ${COMMON} --m 8 --filter ${filter} --mp ${mp} &
done
wait
echo "=== MPAF done ==="

for filter in ${filters}; do
    echo "Running: filter=${filter}, mp=CAMP_clipping"
    python main.py ${COMMON} --m 8 --filter ${filter} --mp CAMP --lamda 2 --pk all --CAMP_mode clipping &
done
wait
echo "=== CAMP clipping done ==="

for filter in ${filters}; do
    echo "Running: filter=${filter}, mp=CAMP_perturbation"
    python main.py ${COMMON} --m 8 --filter ${filter} --mp CAMP --lamda 2 --pk all --CAMP_mode perturbation &
done
wait
echo "=== CAMP perturbation done ==="

echo "All Cifar10 experiments done!"
