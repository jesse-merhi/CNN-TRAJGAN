import logging

import pandas as pd
from sklearn.model_selection import KFold
from conv_gan.models.dcgan import DCGan
from evaluation_framework.EvalModel import EvalModel
import time
ACCOUNTANT = 'prv'  # Default is 'prv', but I found that 'rdp' is more stable

log = logging.getLogger()

class CNNEvalModel(EvalModel):
    def __init__(self):
        pass
    def is_ntg(self):
        return False
    def eval_with_cross_validation(self,dataset ,opt:dict,k=5):
        if opt["dp"]:
            print(f"Using DP CNNGAN {opt['dp']}")
        else:
            print("Using CNNGAN")
        self.dataset_name = dataset
        # Load dataset based on dataset_name
        print(f"Loading dataset for CNN-GAN: {dataset}")
        match dataset:
            case "geolife":
                data =  pd.read_csv("data/geolife/restricted_geolife.csv")
            case "fs":
                data =  pd.read_csv("data/fs_nyc/restricted_foursquare.csv")
            case _ :
                print("Invalid dataset please choose: fs or geolife")
                exit(1)
        self.trajectories = [traj.values.tolist()[:144] for tid, traj in data.groupby('tid') ]
        kfold = KFold(n_splits=k,shuffle=True,random_state=1)
        count = 0
        
        for train, test in kfold.split(self.trajectories):
            dcgan = DCGan(
                opt,
                mainData=[self.trajectories[i] for i in train],
                testData=[self.trajectories[i] for i in test],
                fold=count,
                dp=opt["dp"],
                privacy_accountant=ACCOUNTANT,
                gpu=opt["gpu"],
                max_grad_norm=opt["max_grad_norm"],
                epsilon=opt["epsilon"] if opt["dp"] else None,
            )
            dcgan.training_loop()
            count += 1
            time.sleep(5)
            if opt["dp"]:
                log.info(f"Final Epsilon: {dcgan.epsilon} @ Delta: {dcgan.delta}")
                if dcgan.epsilon > opt["epsilon"]:
                    log.error(f"Epsilon ({dcgan.epsilon}) exceeded the target epsilon ({opt['epsilon']}). Exiting.")
                    raise RuntimeError(f"Epsilon ({dcgan.epsilon}) exceeded the target epsilon ({opt['epsilon']}). Exiting.")
        