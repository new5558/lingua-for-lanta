import argparse
import os

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file


def map_state_dict_qwen(source_state_dict):
    mapped_state_dict = {}
    for source_key, source_tensor in source_state_dict.items():
        if 'embed_tokens.weight' in source_key:
            mapped_state_dict['tok_embeddings.weight'] = source_tensor
            mapped_state_dict['output.weight'] = source_tensor # weight tying
        
        elif "layers" in source_key and "weight" in source_key:
            layer_num = int(source_key.split(".")[2])
            if 'input_layernorm.weight' in source_key:
                  mapped_state_dict[f'layers.{layer_num}.attention_norm.weight'] = source_tensor
            elif 'post_attention_layernorm.weight' in source_key:
                  mapped_state_dict[f'layers.{layer_num}.ffn_norm.weight'] = source_tensor
            elif 'mlp.down_proj.weight' in source_key:
                mapped_state_dict[f'layers.{layer_num}.feed_forward.w2.weight'] = source_tensor
            elif 'mlp.gate_proj.weight' in source_key:
                 mapped_state_dict[f'layers.{layer_num}.feed_forward.w1.weight'] = source_tensor
            elif 'mlp.up_proj.weight' in source_key:
                mapped_state_dict[f'layers.{layer_num}.feed_forward.w3.weight'] = source_tensor
            elif 'self_attn.q_proj.weight' in source_key:
                mapped_state_dict[f'layers.{layer_num}.attention.wq.weight'] = source_tensor
            elif 'self_attn.k_proj.weight' in source_key:
                mapped_state_dict[f'layers.{layer_num}.attention.wk.weight'] = source_tensor
            elif 'self_attn.v_proj.weight' in source_key:
                mapped_state_dict[f'layers.{layer_num}.attention.wv.weight'] = source_tensor

            elif 'self_attn.o_proj.weight' in source_key:
                mapped_state_dict[f'layers.{layer_num}.attention.wo.weight'] = source_tensor
        elif 'norm.weight' in source_key:
             mapped_state_dict['norm.weight'] = source_tensor

        elif "layers" in source_key and "bias" in source_key:

            layer_num = int(source_key.split(".")[2])
            if 'self_attn.q_proj.bias' in source_key:
                 mapped_state_dict[f'layers.{layer_num}.attention.wq.bias'] = source_tensor
            elif 'self_attn.k_proj.bias' in source_key:
                 mapped_state_dict[f'layers.{layer_num}.attention.wk.bias'] = source_tensor
            elif 'self_attn.v_proj.bias' in source_key:
                 mapped_state_dict[f'layers.{layer_num}.attention.wv.bias'] = source_tensor

        else:
             print(f'skipping {source_key}')

    return mapped_state_dict


def main(consolidated_checkpoint_path: str, output_path: str, model_family: str):
    if model_family == 'llama3':
        checkpoint_path = os.path.join(consolidated_checkpoint_path, 'original/consolidated.00.pth')
        state_dict = torch.load(
            checkpoint_path, 
            map_location=torch.device('cpu'), 
            weights_only = True
        )

    elif model_family == 'qwen2':
        checkpoint_path = os.path.join(consolidated_checkpoint_path, 'model.safetensors')
        state_dict = load_file(checkpoint_path)
        state_dict = map_state_dict_qwen(state_dict)
    else: 
        raise NotImplementedError(f"{model_family} model family is not implemented")
    
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
    parser.add_argument("model_family", type=str)

    args = parser.parse_args()

    main(args.chkpt, args.output, args.model_family)
