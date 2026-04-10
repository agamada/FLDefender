# filters="avg multi-krum median trmean flame FLDetector"
filters="avg  trmean "

# mp="none" # --dp lf
# for filter in ${filters}; do
#     echo "Running with filter=${filter}, mp=${mp}"
#     python main.py \
#         --model cnn \
#         --dataset FashionMNIST \
#         --lr 0.02 \
#         --ld True \
#         --m 0 \
#         --filter ${filter} \
#         --mp none \
#         --exp 4 &
# done
# # # wait

# mp="min-max"
# for filter in ${filters}; do
#     echo "Running with filter=${filter}, mp=${mp}"
#     python main.py \
#         --model cnn \
#         --dataset FashionMNIST \
#         --lr 0.02 \
#         --ld True \
#         --m 8 \
#         --filter ${filter} \
#         --mp ${mp} \
#         --exp 4 &
# done
# # # wait


# mp="sign_flip"
# for filter in ${filters}; do
#     echo "Running with filter=${filter}, mp=${mp}"
#     python main.py \
#         --model cnn \
#         --dataset FashionMNIST \
#         --lr 0.02 \
#         --ld True \
#         --m 8 \
#         --filter ${filter} \
#         --mp ${mp} \
#         --exp 4 &
# done
# # # wait


# mp="LIE"
# for filter in ${filters}; do
#     echo "Running with filter=${filter}, mp=${mp}"
#     python main.py \
#         --model cnn \
#         --dataset FashionMNIST \
#         --lr 0.02 \
#         --ld True \
#         --m 8 \
#         --filter ${filter} \
#         --mp ${mp} \
#         --exp 4 &
# done
# # # wait


# mp="MPAF"
# for filter in ${filters}; do
#     echo "Running with filter=${filter}, mp=${mp}"
#     python main.py \
#         --model cnn \
#         --dataset FashionMNIST \
#         --lr 0.02 \
#         --ld True \
#         --m 8 \
#         --filter ${filter} \
#         --mp ${mp} \
#         --exp 4 &
# done
# # wait


mp="CAMP"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.02 \
        --ld True \
        --m 8 \
        --filter ${filter} \
        --mp ${mp} \
        --lamda 1 \
        --pk 'all' \
        --CAMP_mode 'clipping' \
        --exp 4 &
done
# wait

mp="CAMP"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn \
        --dataset FashionMNIST \
        --lr 0.02 \
        --ld True \
        --m 8 \
        --filter ${filter} \
        --mp ${mp} \
        --lamda 1 \
        --pk 'all' \
        --CAMP_mode 'perturbation' \
        --exp 4 &
done
wait