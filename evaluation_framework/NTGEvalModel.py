import logging

import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import time
import config
from conv_gan.datasets import get_dataset, Datasets
from conv_gan.datasets.base_dataset import LAT, LON
from conv_gan.datasets.fs_nyc import CAT, DAY, HOUR, TID, UID
from conv_gan.datasets.padding import ZeroPadding
from conv_gan.models.noise_trajgan import Noise_TrajGAN
from evaluation_framework.EvalModel import EvalModel

ACCOUNTANT = 'prv'  # Default is 'prv', but I found that 'rdp' is more stable

log = logging.getLogger()

class NTGEvalModel(EvalModel):
    dl = None
    ds = None
    def __init__(self):
        return
    

    def eval_with_cross_validation(self, dataset, opt: dict, k: int = 5):
        name = (f'{"DP-" if opt["dp"] else ""}NTG_{str(dataset)}_{opt["lr_g"]:.0e}_{opt["n_critic"]}xD{opt["lr_d"]:.0e}_'
                f'L{opt["latent_dim"]}_N{opt["noise_dim"]}_B{opt["batch_size"]}_{"WGAN" if opt["wgan"] else "GAN"}'
                f'{"-GP" if opt["gradient_penalty"] and not opt["lp"] else ""}{"-LP" if opt["lp"] else ""}')

        if opt["dp"]:
            print("Using DP NoiseTrajGAN")
        else:
            print("Using Standard NoiseTrajGAN")

        if dataset == "geolife":
            path = config.BASE_DIR + "data/geolife/restricted_geolife.csv"
            dtype={TID: str, UID: 'int32', HOUR: 'int32', DAY: 'int32', CAT: 'int32'}
        else:
            path = config.BASE_DIR + "data/fs_nyc/restricted_foursquare.csv"
            dtype={'tid': str,
                   'label': int,
                   LAT: float,
                   LON: float,
                   "day":int,
                   "hour":int
                   }
        data = pd.read_csv(path,dtype=dtype)
        tids = data['tid'].unique()
        kfold = KFold(n_splits=k,shuffle=True,random_state=1)
        count = 0
        for train, test in kfold.split(tids):

            # Initialise a training and testing dataset and dataloader.
            train_tids = tids[train]
            test_tids = tids[test]
            padding_fn_train = ZeroPadding(
                return_len=True,
                return_labels=True,
                feature_first=True,
            )
            padding_fn_test = ZeroPadding(
                return_len=False,
                return_labels=False,
                feature_first=True,
            )
            self.train_ds, self.test_ds = self.load_dataset_ntg(dataset, train_tids, test_tids)
            self.train_dl = DataLoader(self.train_ds, batch_size=opt["batch_size"], collate_fn=padding_fn_train,
                                       num_workers=4, pin_memory=True)
            self.test_dl = DataLoader(self.test_ds, batch_size=opt["batch_size"], collate_fn=padding_fn_test,
                                      num_workers=4, pin_memory=True)

            ntg = Noise_TrajGAN(
                dp= opt["dp"],
                features=self.train_ds.features,
                latent_dim=opt["latent_dim"],
                noise_dim=opt["noise_dim"],
                wgan=opt["wgan"],
                gradient_penalty=opt["gradient_penalty"],
                lipschitz_penalty=opt["lp"],
                lr_g=opt["lr_g"],
                lr_d=opt["lr_d"],
                gpu = opt["gpu"],
                name=name,
                dp_in_dis=opt["dp_in_dis"],
                privacy_accountant=ACCOUNTANT,
            ).to(f"cuda:{opt['gpu']}")

            log.info(f"Using Device:\t\t{ntg.device}")

            # Initialize DP --> Returns DP dataloader
            if opt["dp"]:
                self.train_dl = ntg.init_dp(
                    dataloader=self.train_dl,
                    epochs=opt["epochs"],
                    max_grad_norm=opt["max_grad_norm"],
                    target_epsilon=opt["epsilon"],
                )

            ntg.training_loop(self.train_dl, epochs=opt["epochs"], dataset_name=Datasets.GEOLIFE if dataset == "geolife" else Datasets.FS ,
                              n_critic=opt["n_critic"], plot_freq=200, save_freq=-1, tensorboard=True, notebook=False,
                              test_dataloader=self.test_dl, test_dataset=self.test_ds,
                               eval_freq=opt["eval_freq"], fold = count)
            count +=1 
            time.sleep(5)
            if opt["dp"]:
                log.info(f"Final Epsilon: {ntg.epsilon} @ Delta: {ntg.delta}")
                if ntg.epsilon > opt["epsilon"]:
                    log.error(f"Epsilon ({ntg.epsilon}) exceeded the target epsilon ({opt['epsilon']}). Exiting.")
                    # raise RuntimeError(
                    #     f"Epsilon ({ntg.epsilon}) exceeded the target epsilon ({opt['epsilon']}). Exiting.")

    def is_ntg(self, ):
        return True
    def load_dataset_ntg(self, dataset_name, train_tids, test_tids):
        # Load dataset based on dataset_name
        self.dataset_name = dataset_name
        print(f"Loading dataset for NoiseTrajGAN: {dataset_name}")
        
        match dataset_name:
            case "geolife":
                train_ds = get_dataset(Datasets.GEOLIFE, return_labels=True, tids=train_tids)
                test_ds = get_dataset(Datasets.GEOLIFE, tids=test_tids)

            case "fs":
                train_ds = get_dataset(Datasets.FS, return_labels=True, tids=train_tids)
                test_ds = get_dataset(Datasets.FS, tids=test_tids)
            case _:
                print("Invalid dataset please choose: fs or geolife")
                exit(1)
        return train_ds, test_ds
        
