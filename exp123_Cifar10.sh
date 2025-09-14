filters="avg krum median trmean sad"

mp="min-max"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn2 \
        --dataset Cifar10 \
        --lr 0.04 \
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
        --model cnn2 \
        --dataset Cifar10 \
        --lr 0.04 \
        --ld True \
        --m 0 \
        --dp lf \
        --ls 5 \
        --lt 3 \
        --filter ${filter} \
        --mp ${mp} \
        --s 4 \
        --exp 3 &
done
wait

mp="sign_flip"
for filter in ${filters}; do
    echo "Running with filter=${filter}, mp=${mp}"
    python main.py \
        --model cnn2 \
        --dataset Cifar10 \
        --lr 0.04 \
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
        --model cnn2 \
        --dataset Cifar10 \
        --lr 0.04 \
        --ld True \
        --m 16 \
        --dp lf \
        --ls 5 \
        --lt 3 \
        --filter ${filter} \
        --mp ${mp} \
        --s 4 \
        --exp 3 &
done
wait

dpd_modes="high auto low none"
ss="1 2 4 6 8"

for mode in ${dpd_modes}; do
    for s in ${ss}; do
        echo "Running with dpd_mode=${mode}, s=${s}"
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
            --dpd_mode ${mode} \
            --noise_level 0 \
            --mp scale \
            --s ${s} \
            --exp 1 &
    done
    wait
done
wait

# 1) 生成 20 个噪声点（含首尾）
noise_levels=$(awk 'BEGIN{for(i=0;i<50;i++) printf("%.6f\n", 0.0002+i*0.0002)}')

# 2) 第一次用 GNU parallel 可能会提示引用声明，先执行一次：
# parallel --will-cite

# 3) 并行执行（一次跑 5 个），每个任务单独存日志、带进度条与作业日志
echo "${noise_levels}" | parallel -j 5 \
  'python main.py \
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
      --noise_level {} \
      --mp scale \
      --s 4 \
      --exp 2'
