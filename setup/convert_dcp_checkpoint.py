import argparse

import torch
import torch.distributed.checkpoint as load_state_dict


def main(input_path, output_path):
    state_dict = torch.load(input_path)
    
    state_dict = torch.save(state_dict['model'], output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chkpt", type=str)
    parser.add_argument("output", type=str)

    args = parser.parse_args()

    main(args.chkpt, args.output)
