filters="avg krum median trmean sad"

mp="none"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn2 \
        --dataset Cifar10 \
        --lr 0.04 \
        --ld True \
        --m 16 \
        --dp lf \
        --ls 4 \
        --lt 6 \
        --filter ${filter} \
        --mp ${mp} \
        --s 4 \
        --exp 3 &
done
wait