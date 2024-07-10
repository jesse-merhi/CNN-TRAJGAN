import json
import logging
import os
from pprint import pprint

import config
from conv_gan.utils.parser import load_config_file
from evaluation_framework.parser import get_eval_parser
from .CNNEvalModel import CNNEvalModel
from .NTGEvalModel import NTGEvalModel
from conv_gan.utils.logger import configure_root_loger

log = logging.getLogger()

def load_model(model_name: str, opt: dict):
    # Load model based on model_name
    match model_name:
        case "ntg":
            return NTGEvalModel()
        case "cnngan":
            return CNNEvalModel()
        case _ :
            log.error("Invalid model name. Please choose one of 'ntg' or 'cnngan'.")
            exit(1)
         
def validate_opt(opt: dict):
    if opt["dp"] and opt["dp_in_dis"] and opt["n_critic"] > 1:
        log.error("DP and DP in discriminator shouldn't be used together with n_critic > 1")
        exit(1)
    if opt["wgan"] and opt["n_critic"] == 1:
        log.warning("WGAN should be used with n_critic > 1")
    if opt["dp"] and opt["dp_in_dis"] and opt["wgan"]:
        log.warning("WGAN should be used with DP in Generator only")
    if opt["dp"] and not opt["dp_in_dis"] and not opt["wgan"]:
        log.warning("GAN should apply DP in Discriminator")

def main():
    opt = vars(get_eval_parser().parse_args())
    if 'config' in opt and opt['config'] is not None:
        opt = load_config_file(opt)
    print("Arguments: ")
    pprint(opt)

    validate_opt(opt)

    # Configure Logging
    configure_root_loger(
        logging_level=logging.INFO,
        file=os.path.join(config.LOG_DIR, f"eval_{opt['config'].replace('.json','').replace('configs/', '')}.log")
    )
    # Write parameter dict to log in a nice way
    log.info("Arguments: ")
    log.info(json.dumps(opt, indent=4))
    try:
        model = load_model(opt["model"], opt)
        model.eval_with_cross_validation(opt["dataset"], opt)
    except Exception as e:
        log.error(f"Error occurred:")
        log.error(str(e))
        raise e

if __name__ == "__main__":
    main()
