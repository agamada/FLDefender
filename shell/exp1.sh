dpd_modes="high auto low none"
ss="1 2 4 6 8"

for mode in ${dpd_modes}; do
    for s in ${ss}; do
        echo "Running with dpd_mode=${mode}, s=${s}"
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
            --dpd_mode ${mode} \
            --noise_level 0 \
            --mp scale \
            --s ${s} \
            --exp 1 &
    done
    wait
done
