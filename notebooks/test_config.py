from src.training.utils import load_config

config = load_config()

print(config["training"]["epochs"])
