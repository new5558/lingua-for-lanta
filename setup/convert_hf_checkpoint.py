import argparse

import torch
import torch.distributed.checkpoint as dcp


def main(consolidated_checkpoint_path, output_path):
    state_dict = torch.load(
        consolidated_checkpoint_path, 
        map_location=torch.device('cpu'), 
        weights_only = True
    )
    
    converted_state_dict = {"model": state_dict}
    dcp.state_dict_saver._save_state_dict(
        converted_state_dict, 
        storage_writer=dcp.FileSystemWriter(output_path),
        no_dist=True
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chkpt", type=str)
    parser.add_argument("output", type=str)

    args = parser.parse_args()

    main(args.chkpt, args.output)
