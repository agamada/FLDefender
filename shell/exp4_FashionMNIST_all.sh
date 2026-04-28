#!/bin/bash
# Exp4 - FashionMNIST:
# Model: cnn, lr: 0.02, m: 8, k: 40, alpha: 0.5 (noniid, unbalanced, dir)
# Filters: avg, multi-krum, median, trmean, flame, maud-norm (w3/5/10), maud-cosine (w3/5/10)
# Attacks: none, sign_flip, LIE, MPAF, CAMP_clipping_v8, CAMP_perturbation_v5
#

# python main.py --model cnn --dataset FashionMNIST --lr 0.02 --ld True --exp 4 --device cuda --device_id 0 --m 8 --filter multi-krum --mp CAMP --CAMP_mode clipping_v8 --lamda 2 --pk all
# python main.py --model cnn --dataset FashionMNIST --lr 0.02 --ld True --exp 4 --device cuda --device_id 0 --m 8 --filter maud-norm --mp CAMP --CAMP_mode clipping_v8 --lamda 2 --pk all --maud_window 5

COMMON="--model cnn --dataset FashionMNIST --lr 0.02 --ld True --exp 4 --device cuda --device_id 0"
BASIC_FILTERS="avg multi-krum median trmean flame"
MAUD_WINDOWS="3 5 10"


# none (no attack, m=0)
echo "=== Phase 1: none (no attack) ==="
for f in $BASIC_FILTERS; do
    nohup python main.py $COMMON --m 0 --filter $f --mp none > /dev/null 2>&1 &
done
for w in $MAUD_WINDOWS; do
    nohup python main.py $COMMON --m 0 --filter maud-norm   --mp none --maud_window $w > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 0 --filter maud-cosine --mp none --maud_window $w > /dev/null 2>&1 &
done
wait
echo "--- Phase 1 done ---"

# sign_flip
echo "=== Phase 2: sign_flip ==="
for f in $BASIC_FILTERS; do
    nohup python main.py $COMMON --m 8 --filter $f --mp sign_flip > /dev/null 2>&1 &
done
for w in $MAUD_WINDOWS; do
    nohup python main.py $COMMON --m 8 --filter maud-norm   --mp sign_flip --maud_window $w > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter maud-cosine --mp sign_flip --maud_window $w > /dev/null 2>&1 &
done
wait
echo "--- Phase 2 done ---"

# LIE
echo "=== Phase 3: LIE ==="
for f in $BASIC_FILTERS; do
    nohup python main.py $COMMON --m 8 --filter $f --mp LIE > /dev/null 2>&1 &
done
for w in $MAUD_WINDOWS; do
    nohup python main.py $COMMON --m 8 --filter maud-norm   --mp LIE --maud_window $w > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter maud-cosine --mp LIE --maud_window $w > /dev/null 2>&1 &
done
wait
echo "--- Phase 3 done ---"

# MPAF
echo "=== Phase 4: MPAF ==="
for f in $BASIC_FILTERS; do
    nohup python main.py $COMMON --m 8 --filter $f --mp MPAF > /dev/null 2>&1 &
done
for w in $MAUD_WINDOWS; do
    nohup python main.py $COMMON --m 8 --filter maud-norm   --mp MPAF --maud_window $w > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter maud-cosine --mp MPAF --maud_window $w > /dev/null 2>&1 &
done
wait
echo "--- Phase 4 done ---"

# CAMP clipping_v8
echo "=== Phase 5: CAMP clipping_v8 ==="
for f in $BASIC_FILTERS; do
    nohup python main.py $COMMON --m 8 --filter $f --mp CAMP --CAMP_mode clipping_v8 --lamda 2 --pk all > /dev/null 2>&1 &
done
for w in $MAUD_WINDOWS; do
    nohup python main.py $COMMON --m 8 --filter maud-norm   --mp CAMP --CAMP_mode clipping_v8 --lamda 2 --pk all --maud_window $w > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter maud-cosine --mp CAMP --CAMP_mode clipping_v8 --lamda 2 --pk all --maud_window $w > /dev/null 2>&1 &
done
wait
echo "--- Phase 5 done ---"

# CAMP perturbation_v5
echo "=== Phase 6: CAMP perturbation_v5 ==="
for f in $BASIC_FILTERS; do
    nohup python main.py $COMMON --m 8 --filter $f --mp CAMP --CAMP_mode perturbation_v5 --lamda 2 --pk all > /dev/null 2>&1 &
done
for w in $MAUD_WINDOWS; do
    nohup python main.py $COMMON --m 8 --filter maud-norm   --mp CAMP --CAMP_mode perturbation_v5 --lamda 2 --pk all --maud_window $w > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter maud-cosine --mp CAMP --CAMP_mode perturbation_v5 --lamda 2 --pk all --maud_window $w > /dev/null 2>&1 &
done
wait
echo "--- Phase 6 done ---"

echo "=== All FashionMNIST exp4 experiments done! ==="
