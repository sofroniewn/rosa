import hydra
from hydra.core.config_store import ConfigStore
from rosa import RosaConfig, train


cs = ConfigStore.instance()
cs.store(name="rosa_config", node=RosaConfig)

CONFIG_PATH = "../conf"
CONFIG_NAME = "config"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: RosaConfig) -> None:
    import os
    print("Working directory : {}".format(os.getcwd()))
    train(cfg)


if __name__ == "__main__":
    main()
