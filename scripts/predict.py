import hydra
from rosa import predict
from omegaconf import OmegaConf
from glob import  glob



if __name__ == "__main__":

    BASE_DIR = "/home/ec2-user/outputs/2023-03-26/17-15-19" # /home/ec2-user/ or /Users/nsofroniew/Documents/data/rosa/
    config_dir = BASE_DIR + "/.hydra"

    with hydra.initialize_config_dir(config_dir=config_dir):
        config = hydra.compose(config_name="config", overrides=OmegaConf.load(config_dir + "/overrides.yaml"))

        # chkpts = BASE_DIR + "/checkpoints/epoch=*.ckpt"
        chkpts = BASE_DIR + "/checkpoints/last.ckpt"
        chkpt = glob(chkpts)[0]
        predict(config, chkpt)