noise_levels="0.0005 0.005 0.05 0.5"


for noise_level in ${noise_levels}; do
    python main.py \
        --model cnn2 \
        --dataset Cifar10 \
        --lr 0.04 \
        --ld True \
        --m 8 \
        --dp lf \
        --ls 5 \
        --lt 3 \
        --filter dpd \
        --dpd_mode auto \
        --noise_level ${noise_level} \
        --mp scale \
        --s 4 \
        --exp 2 &
done
wait