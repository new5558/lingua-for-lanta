import argparse
import os

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save



def main(input_path, output_path):
    save_path = os.path.join(output_path, 'consolidated.pth')
    final_path = os.path.join(output_path, 'consolidated.00.pth')
    
    dcp_to_torch_save(input_path, save_path)
    state_dict = torch.load(save_path)
    state_dict = torch.save(state_dict['model'], final_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chkpt", type=str)
    parser.add_argument("output", type=str)

    args = parser.parse_args()

    main(args.chkpt, args.output)
