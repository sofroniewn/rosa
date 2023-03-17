import hydra
from rosa import predict
from omegaconf import OmegaConf
from glob import  glob



if __name__ == "__main__":

    BASE_DIR = "/home/ec2-user/outputs/2023-03-17/07-06-31" # /home/ec2-user/ or /Users/nsofroniew/Documents/data/rosa/
    config_dir = BASE_DIR + "/.hydra"

    with hydra.initialize_config_dir(config_dir=config_dir):
        config = hydra.compose(config_name="config", overrides=OmegaConf.load(config_dir + "/overrides.yaml"))

        # chkpts = BASE_DIR + "/checkpoints/epoch=*.ckpt"
        chkpts = BASE_DIR + "/checkpoints/last.ckpt"
        chkpt = glob(chkpts)[0]

        # config.device = 'cpu'
        # config.num_devices = 32
        # config.preprocessing.bulk_data.sample_col = 'donor_id'
        predict(config, chkpt)
        
        # adata, rdm, rlm = predict(config, chkpt)

        # output_path = str(rdm.adata_path.with_name(rdm.adata_path.stem + '__preprocessed.h5ad'))

        # print(output_path)
        # adata.write_h5ad(output_path)

        # # display(adata)
        # print(chkpt)