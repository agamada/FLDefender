noise_levels="0.002 0.004 0.006 0.008 0.01"

for level in ${noise_levels}; do
    echo "Running with noise_level=${level}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.2 \
        --ld True \
        --m 8 \
        --dp lf \
        --ls 4 \
        --lt 6 \
        --filter dpd \
        --dpd_mode auto \
        --noise_level ${level} \
        --mp scale \
        --s 4 \
        --exp 2 &
done
wait


noise_levels="0.012 0.014 0.016 0.018 0.02"

for level in ${noise_levels}; do
    echo "Running with noise_level=${level}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.2 \
        --ld True \
        --m 8 \
        --dp lf \
        --ls 4 \
        --lt 6 \
        --filter dpd \
        --dpd_mode auto \
        --noise_level ${level} \
        --mp scale \
        --s 4 \
        --exp 2 &
done
wait

noise_levels="0.022 0.024 0.026 0.028 0.03"

for level in ${noise_levels}; do
    echo "Running with noise_level=${level}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.2 \
        --ld True \
        --m 8 \
        --dp lf \
        --ls 4 \
        --lt 6 \
        --filter dpd \
        --dpd_mode auto \
        --noise_level ${level} \
        --mp scale \
        --s 4 \
        --exp 2 &
done
wait

noise_levels="0.032 0.034 0.036 0.038 0.04"

for level in ${noise_levels}; do
    echo "Running with noise_level=${level}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.2 \
        --ld True \
        --m 8 \
        --dp lf \
        --ls 4 \
        --lt 6 \
        --filter dpd \
        --dpd_mode auto \
        --noise_level ${level} \
        --mp scale \
        --s 4 \
        --exp 2 &
done
wait