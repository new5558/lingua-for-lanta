import argparse
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp

def main(consolidated_checkpoint_path, output_path):
    torch_save_to_dcp(consolidated_checkpoint_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chkpt", type=str)
    parser.add_argument("output", type=str)

    args = parser.parse_args()

    main(args.chkpt, args.output)
