# python main.py --model cnn --filter krum --mp min-max --m 4 --dataset FashionMNIST

filters="krum median trmean avg"
mps="min-max LIE none"

for filter in ${filters}; do
    for mp in ${mps}; do
        python main.py --model cnn --filter ${filter} --mp ${mp} --m 4 --dataset FashionMNIST
    done
done