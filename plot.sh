
# DCGAN FS
python -m figure_framework.plot -r results/DCGAN_fs_257_1xD_LR-2e-04_Sched-4000_GFac-1.0_GAN results/DCGAN_fs_257_5xD_LR-1e-04_Sched-4000_GFac-1.0_WGAN-LP results/DCGAN_fs_257_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN-LP -l "DCGAN (old)" "DCGAN(WGAN-LP1e-4)" "DCGAN (WGAN-LP2e-4)" -o metric_dcgan_fs

# DCGAN GEOLIFE
python -m figure_framework.plot -l "DCGAN (old)" "DCGAN(WGAN-LP)" -o metric_dcgan_geolife -r results/DCGAN_geolife_110_1xD_LR-2e-04_Sched-4000_GFac-1.0_GAN results/DCGAN_geolife_110_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN-LP/

# DP-DCGAN FS
python -m figure_framework.plot -l "DP-DCGAN (old)" "DP-DCGAN (GAN)" "DP-DCGAN (WGAN)" "DP-DCGAN (WGAN-LP)" -o metric_dp_dcgan_fs -r results/DP-DCGAN_fs_2570__LR-0.002_Sched-4000_GFac-1.0/ results/DP-DCGAN_fs_2570_1xD_LR-2e-04_Sched-4000_GFac-1.0_GAN results/DP-DCGAN_fs_2570_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN/ results/DP-DCGAN_fs_2570_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN-LP/

# DP-DCGAN GEOLIFE
#python -m figure_framework.plot -l "DP-DCGAN (old)" "DP-DCGAN (WGAN)" "DP-DCGAN (WGAN-LP)" "DP-DCGAN (LP F=10)" -o metric_dp_dcgan_geolife -r results/DP-DCGAN_geolife_1100__LR-0.002_Sched-4000_GFac-1.0/ results/DP-DCGAN_geolife_1100_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN/ results/DP-DCGAN_geolife_1100_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN-LP/ results/DP-DCGAN_geolife_1100_5xD_LR-2e-04_Sched-4000_GFac-10.0_WGAN-LP
python -m figure_framework.plot -l "DP-DCGAN (GAN)" "DP-DCGAN (WGAN)" "DP-DCGAN (WGAN-LP)" "DP-DCGAN (LP F=10)" -o metric_dp_dcgan_geolife -r results/DP-DCGAN_geolife_1100_1xD_LR-2e-04_Sched-4000_GFac-1.0_GAN results/DP-DCGAN_geolife_1100_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN/ results/DP-DCGAN_geolife_1100_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN-LP/ results/DP-DCGAN_geolife_1100_5xD_LR-2e-04_Sched-4000_GFac-10.0_WGAN-LP

# NTG/DP-NTG FS
python -m figure_framework.plot -l "NTG" "DP-NTG" -o metric_ntg_fs -r results/NTG_fs_1e-04_5xD1e-04_L256_N28_B64_WGAN-LP/ results/DP-NTG_fs_1e-04_5xD1e-04_L256_N28_B640_WGAN/

# NTG/DP-NTG GEOLIFE
python -m figure_framework.plot -l "NTG" "DP-NTG" -o metric_ntg_geolife -r results/NTG_geolife_1e-04_5xD1e-04_L256_N28_B64_WGAN-LP/ results/DP-NTG_geolife_1e-04_5xD1e-04_L256_N28_B640_WGAN/

# Paper Plot FS
python -m figure_framework.plot -n results/NTG_fs_1e-04_5xD1e-04_L256_N28_B64_WGAN-LP -c results/DCGAN_fs_257_1xD_LR-2e-04_Sched-4000_GFac-1.0_GAN -dn results/DP-NTG_fs_1e-04_5xD1e-04_L256_N28_B640_WGAN -dc results/DP-DCGAN_fs_2570_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN-LP -o metrics_fs

# Paper Plot Geolife
python -m figure_framework.plot -n results/NTG_geolife_1e-04_5xD1e-04_L256_N28_B64_WGAN-LP -c results/DCGAN_geolife_110_1xD_LR-2e-04_Sched-4000_GFac-1.0_GAN -dn results/DP-NTG_geolife_1e-04_5xD1e-04_L256_N28_B640_WGAN -dc results/DP-DCGAN_geolife_1100_5xD_LR-2e-04_Sched-4000_GFac-1.0_WGAN-LP -o metrics_geolife
