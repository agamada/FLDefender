filters="avg multi-krum median trmean flame"

# mp="CAMP"
# for filter in ${filters}; do
#     echo "Running with filter=${filter}, mp=${mp}"
#     python main.py \
#         --model cnn2 \
#         --dataset Cifar10 \
#         --lr 0.04 \
#         --ld True \
#         --m 8 \
#         --filter ${filter} \
#         --mp ${mp} \
#         --lamda 1.5 \
#         --pk 'all' \
#         --CAMP_mode 'clipping' \
#         --exp 4 &
# done
# wait

# mp="CAMP"
# for filter in ${filters}; do
#     echo "Running with filter=${filter}, mp=${mp}"
#     python main.py \
#         --model cnn2 \
#         --dataset Cifar10 \
#         --lr 0.04 \
#         --ld True \
#         --m 8 \
#         --filter ${filter} \
#         --mp ${mp} \
#         --lamda 1.5 \
#         --pk 'all' \
#         --CAMP_mode 'perturbation' \
#         --exp 4 &
# done
# wait

mp="sign_flip"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn2 \
        --dataset Cifar10 \
        --lr 0.04 \
        --ld True \
        --m 8 \
        --filter ${filter} \
        --mp ${mp} \
        --exp 4 &
done
wait