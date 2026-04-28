#!/bin/bash
# Exp4 - FashionMNIST:
# Model: cnn, lr: 0.02, m: 8, k: 40, alpha: 0.5 (noniid, unbalanced, dir)
# 按 filter × N 分批
#
# Example commands:
# python main.py --model cnn --dataset FashionMNIST --lr 0.02 --ld True --exp 4 --device cuda --device_id 0 --m 8 --filter maud-norm --mp CAMP --CAMP_mode clipping_v8 --lamda 2 --pk all --maud_window 5
# python main.py --model cnn --dataset FashionMNIST --lr 0.02 --ld True --exp 4 --device cuda --device_id 0 --m 8 --filter maud-cosine --mp CAMP --CAMP_mode clipping_v8 --lamda 2 --pk all --maud_window 5

COMMON="--model cnn --dataset FashionMNIST --lr 0.02 --ld True --exp 4 --device cuda --device_id 0"

run_batch() {
    local filter=$1
    local window=$2
    echo "=== ${filter} w${window} ==="
    nohup python main.py $COMMON --m 0 --filter $filter --mp none              --maud_window $window > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter $filter --mp sign_flip         --maud_window $window > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter $filter --mp LIE               --maud_window $window > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter $filter --mp MPAF              --maud_window $window > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter $filter --mp CAMP --CAMP_mode clipping_v8   --lamda 2 --pk all --maud_window $window > /dev/null 2>&1 &
    nohup python main.py $COMMON --m 8 --filter $filter --mp CAMP --CAMP_mode perturbation_v5 --lamda 2 --pk all --maud_window $window > /dev/null 2>&1 &
    wait
    echo "--- ${filter} w${window} done ---"
}

run_batch maud-norm  3
run_batch maud-norm  5
run_batch maud-norm  10
run_batch maud-cosine 3
run_batch maud-cosine 5
run_batch maud-cosine 10

echo "=== All FashionMNIST MAUD experiments done! ==="
