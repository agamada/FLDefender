filters="avg krum median trmean sad"

mp="min-max"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.2 \
        --ld True \
        --m 16 \
        --dp none \
        --filter ${filter} \
        --mp ${mp} \
        --exp 3 &
done
wait

mp="none" # --dp lf
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.2 \
        --ld True \
        --m 0 \
        --dp lf \
        --ls 4 \
        --lt 6 \
        --filter ${filter} \
        --mp scale \
        --s 4 \
        --exp 3 &
done
wait

mp="sign_flip"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.2 \
        --ld True \
        --m 16 \
        --dp none \
        --filter ${filter} \
        --mp ${mp} \
        --exp 3 &
done
wait

mp="scale"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.2 \
        --ld True \
        --m 16 \
        --dp lf \
        --ls 4 \
        --lt 6 \
        --filter ${filter} \
        --mp scale \
        --s 4 \
        --exp 3 &
done
wait

