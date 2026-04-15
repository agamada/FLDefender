#!/bin/bash
COMMON="--model cnn --dataset FashionMNIST --lr 0.02 --ld True --exp 4 --device cuda --device_id 0"
FILTERS="avg multi-krum median trmean flame"

wait_batch() {
    while [ $(ps aux | grep "FashionMNIST.*exp 4" | grep -v grep | wc -l) -gt 0 ]; do
        sleep 30
    done
    echo "Batch done at $(date)"
}

# Batch 1: none baselines (5) + sign_flip avg
echo "=== Batch 1: none + sign_flip/avg ==="
for f in $FILTERS; do
  nohup python main.py $COMMON --m 0 --filter $f --mp none > /dev/null 2>&1 &
done
nohup python main.py $COMMON --m 8 --filter avg --mp sign_flip > /dev/null 2>&1 &
wait_batch

# Batch 2: sign_flip (4 remaining) + LIE avg + LIE multi-krum
echo "=== Batch 2: sign_flip rest + LIE ==="
for f in multi-krum median trmean flame; do
  nohup python main.py $COMMON --m 8 --filter $f --mp sign_flip > /dev/null 2>&1 &
done
nohup python main.py $COMMON --m 8 --filter avg --mp LIE > /dev/null 2>&1 &
nohup python main.py $COMMON --m 8 --filter multi-krum --mp LIE > /dev/null 2>&1 &
wait_batch

# Batch 3: LIE (3) + MPAF (3)
echo "=== Batch 3: LIE rest + MPAF ==="
for f in median trmean flame; do
  nohup python main.py $COMMON --m 8 --filter $f --mp LIE > /dev/null 2>&1 &
done
for f in avg multi-krum median; do
  nohup python main.py $COMMON --m 8 --filter $f --mp MPAF > /dev/null 2>&1 &
done
wait_batch

# Batch 4: MPAF (2) + CAMP clipping (4)
echo "=== Batch 4: MPAF rest + CAMP clipping ==="
for f in trmean flame; do
  nohup python main.py $COMMON --m 8 --filter $f --mp MPAF > /dev/null 2>&1 &
done
for f in avg multi-krum median trmean; do
  nohup python main.py $COMMON --m 8 --filter $f --mp CAMP --CAMP_mode clipping --pk all --lamda 2 > /dev/null 2>&1 &
done
wait_batch

# Batch 5: CAMP clipping (1) + CAMP perturbation (5)
echo "=== Batch 5: CAMP clipping flame + CAMP perturbation ==="
nohup python main.py $COMMON --m 8 --filter flame --mp CAMP --CAMP_mode clipping --pk all --lamda 2 > /dev/null 2>&1 &
for f in avg multi-krum median trmean flame; do
  nohup python main.py $COMMON --m 8 --filter $f --mp CAMP --CAMP_mode perturbation --pk all --lamda 2 > /dev/null 2>&1 &
done
wait_batch

echo "=== All FashionMNIST experiments done! ==="
