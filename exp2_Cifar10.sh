# noise_levels="0.0005 0.005 0.05 0.5"


# for noise_level in ${noise_levels}; do
#     python main.py \
#         --model cnn2 \
#         --dataset Cifar10 \
#         --lr 0.04 \
#         --ld True \
#         --m 8 \
#         --dp lf \
#         --ls 5 \
#         --lt 3 \
#         --filter dpd \
#         --dpd_mode auto \
#         --noise_level ${noise_level} \
#         --mp scale \
#         --s 4 \
#         --exp 2 &
# done
# wait

# 1) 生成 20 个噪声点（含首尾）
noise_levels=$(awk 'BEGIN{for(i=0;i<25;i++) printf("%.6f\n", 0.002+i*0.002)}')

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
