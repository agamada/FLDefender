# 1) 生成 20 个噪声点（含首尾）
noise_levels=$(awk 'BEGIN{for(i=0;i<50;i++) printf("%.6f\n", 0.002+i*0.002)}')

# 2) 第一次用 GNU parallel 可能会提示引用声明，先执行一次：
# parallel --will-cite

# 3) 并行执行（一次跑 5 个），每个任务单独存日志、带进度条与作业日志
echo "${noise_levels}" | parallel -j 5 \
  'python main.py \
      --model cnn \
      --dataset MNIST \
      --lr 0.1 \
      --ld True \
      --m 8 \
      --dp lf \
      --ls 4 \
      --lt 6 \
      --filter dpd \
      --dpd_mode auto \
      --noise_level {} \
      --mp scale \
      --s 4 \
      --exp 2'
