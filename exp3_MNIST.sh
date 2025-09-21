filters="avg krum median trmean"
# filters="flame sad FLDetector"

mp="min-max"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset MNIST \
        --lr 0.2 \
        --ld True \
        --m 16 \
        --dp none \
        --filter ${filter} \
        --mp ${mp} \
        --noise_level 0.001\
        --exp 3 &
done

mp="none" # --dp lf
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset MNIST \
        --lr 0.2 \
        --ld True \
        --m 0 \
        --dp lf \
        --ls 4 \
        --lt 6 \
        --filter ${filter} \
        --mp ${mp} \
        --noise_level 0.001\
        --s 4 \
        --exp 3 &
done

mp="sign_flip"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset MNIST \
        --lr 0.2 \
        --ld True \
        --m 16 \
        --dp none \
        --filter ${filter} \
        --mp ${mp} \
        --noise_level 0.001\
        --exp 3 &
done

mp="scale"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset MNIST \
        --lr 0.2 \
        --ld True \
        --m 16 \
        --dp lf \
        --ls 4 \
        --lt 6 \
        --filter ${filter} \
        --mp ${mp} \
        --noise_level 0.001\
        --s 4 \
        --exp 3 &
done