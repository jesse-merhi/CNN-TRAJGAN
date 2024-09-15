for epsilon in 0.1 0.5 1 5; do
    CUDA_VISIBLE_DEVICES=2,3 python3 -m evaluation_framework.eval cnngan geolife -c configs/dp-cnngan_geolife_iwgan_eps.json --gpu 1 --epsilon $epsilon
done