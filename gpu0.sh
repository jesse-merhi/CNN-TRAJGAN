#!/usr/bin/env bash

# ID 1: Done
# python3 -m evaluation_framework.eval cnngan fs -c configs/cnngan_fs_gan.json --gpu 0
# ID 3: Done
# python3 -m evaluation_framework.eval cnngan fs -c configs/cnngan_fs_wgan.json --gpu 0
# ID 5: Done
# python3 -m evaluation_framework.eval cnngan fs -c configs/dp-cnngan_fs_gan.json --gpu 1
# ID 7: Done
# python3 -m evaluation_framework.eval cnngan fs -c configs/dp-cnngan_fs_wgan.json --gpu 0
#ID 9: Done
# python3 -m evaluation_framework.eval cnngan fs -c configs/dp-cnngan_fs_iwgan.json --gpu 0
# ID 11: Done
#python3 -m evaluation_framework.eval ntg fs -c configs/noise_tg_fs.json --gpu 1
#ID 13: Done
#python3 -m evaluation_framework.eval ntg fs -c configs/dp-noise_tg_fs_wgan.json --gpu 0
#ID 14: Running...
python3 -m evaluation_framework.eval ntg geolife -c configs/dp-noise_tg_geolife_wgan.json --gpu 0
#ID 15: Not started
#python3 -m evaluation_framework.eval ntg fs -c configs/dp-noise_tg_fs_gan.json --gpu 0
#ID 16: Not started
#python3 -m evaluation_framework.eval ntg geolife -c configs/dp-noise_tg_geolife_gan.json --gpu 0

# Placeholder
# python3 -m evaluation_framework.eval cnngan fs -c configs/ --gpu 0
