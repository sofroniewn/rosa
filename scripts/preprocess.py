import hydra
from hydra.core.config_store import ConfigStore
from rosa import RosaConfig
from rosa.data import preprocess

cs = ConfigStore.instance()
cs.store(name="rosa_config", node=RosaConfig)

CONFIG_PATH = "../conf"
CONFIG_NAME = "config"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(config: RosaConfig) -> None:
    preprocess(config)


if __name__ == "__main__":
    main()
