#!/bin/bash
# Exp5 - Cifar10: m=4 attackers (vs exp4 m=8)
COMMON="--model cnn2 --dataset Cifar10 --lr 0.04 --ld True --exp 5 --device cuda --device_id 0"
filters="avg multi-krum median trmean flame"

echo "=== Starting none ==="
for filter in ${filters}; do
    python main.py ${COMMON} --m 0 --filter ${filter} --mp none &
done
wait
echo "=== none done ==="

echo "=== Starting sign_flip ==="
for filter in ${filters}; do
    python main.py ${COMMON} --m 4 --filter ${filter} --mp sign_flip &
done
wait
echo "=== sign_flip done ==="

echo "=== Starting LIE ==="
for filter in ${filters}; do
    python main.py ${COMMON} --m 4 --filter ${filter} --mp LIE &
done
wait
echo "=== LIE done ==="

echo "=== Starting MPAF ==="
for filter in ${filters}; do
    python main.py ${COMMON} --m 4 --filter ${filter} --mp MPAF &
done
wait
echo "=== MPAF done ==="

echo "=== Starting CAMP clipping_v8 ==="
for filter in ${filters}; do
    python main.py ${COMMON} --m 4 --filter ${filter} --mp CAMP --lamda 2 --pk all --CAMP_mode clipping_v8 &
done
wait
echo "=== CAMP clipping_v8 done ==="

echo "=== Starting CAMP perturbation_v5 ==="
for filter in ${filters}; do
    python main.py ${COMMON} --m 4 --filter ${filter} --mp CAMP --lamda 2 --pk all --CAMP_mode perturbation_v5 &
done
wait
echo "=== CAMP perturbation_v5 done ==="

echo "All exp5 Cifar10 experiments done!"
