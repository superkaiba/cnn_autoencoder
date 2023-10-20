import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--architecture", type=str, default="original")
    parser.add_argument("--load_dir", type=str, default="NONE")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--no_wandb", type=bool, default=True)
    args = parser.parse_args()
    return args

# def get_args_eval():
#     parser = argparse.ArgumentParser(description="Train Autoencoder")
#     parser.add_argument("--log_dir", type=str, default="./logs")
#     parser.add_argument("--architecture", type=str, default="original")
#     parser.add_argument("--load_dir", type=str, default="NONE")
#     args = parser.parse_args()
#     return args