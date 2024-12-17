import argparse
import os

import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer

def reverse_map_state_dict_qwen(mapped_state_dict):
    source_state_dict = {}
    
    for mapped_key, mapped_tensor in mapped_state_dict.items():
        if mapped_key == 'tok_embeddings.weight':
            source_state_dict['model.embed_tokens.weight'] = mapped_tensor
        
        if mapped_key == 'output.weight':
            source_state_dict['lm_head.weight'] = mapped_tensor
            
        elif "layers" in mapped_key and "weight" in mapped_key:
            layer_num = int(mapped_key.split(".")[1])
            if 'attention_norm.weight' in mapped_key:
                source_state_dict[f'model.layers.{layer_num}.input_layernorm.weight'] = mapped_tensor
            elif 'ffn_norm.weight' in mapped_key:
                source_state_dict[f'model.layers.{layer_num}.post_attention_layernorm.weight'] = mapped_tensor
            elif 'feed_forward.w2.weight' in mapped_key:
                source_state_dict[f'model.layers.{layer_num}.mlp.down_proj.weight'] = mapped_tensor
            elif 'feed_forward.w1.weight' in mapped_key:
                source_state_dict[f'model.layers.{layer_num}.mlp.gate_proj.weight'] = mapped_tensor
            elif 'feed_forward.w3.weight' in mapped_key:
                source_state_dict[f'model.layers.{layer_num}.mlp.up_proj.weight'] = mapped_tensor
            elif 'attention.wq.weight' in mapped_key:
                source_state_dict[f'model.layers.{layer_num}.self_attn.q_proj.weight'] = mapped_tensor
            elif 'attention.wk.weight' in mapped_key:
                source_state_dict[f'model.layers.{layer_num}.self_attn.k_proj.weight'] = mapped_tensor
            elif 'attention.wv.weight' in mapped_key:
                source_state_dict[f'model.layers.{layer_num}.self_attn.v_proj.weight'] = mapped_tensor
            elif 'attention.wo.weight' in mapped_key:
                source_state_dict[f'model.layers.{layer_num}.self_attn.o_proj.weight'] = mapped_tensor
        
        elif 'norm.weight' in mapped_key:
            source_state_dict['model.norm.weight'] = mapped_tensor

        elif "layers" in mapped_key and "bias" in mapped_key:

            layer_num = int(mapped_key.split(".")[1])

            if 'attention.wq.bias' in mapped_key:
                 source_state_dict[f'model.layers.{layer_num}.self_attn.q_proj.bias'] = mapped_tensor
            elif 'attention.wk.bias' in mapped_key:
                 source_state_dict[f'model.layers.{layer_num}.self_attn.k_proj.bias'] = mapped_tensor
            elif 'attention.wv.bias' in mapped_key:
                 source_state_dict[f'model.layers.{layer_num}.self_attn.v_proj.bias'] = mapped_tensor

        else:
            print(f'skipping {mapped_key}')

    return source_state_dict


def main(
        consolidated_checkpoint_path: str,
        output_path: str, 
        original_path: str,
    ):
    checkpoint_path = os.path.join(consolidated_checkpoint_path, 'consolidated.00.pth')
    state_dict = torch.load(checkpoint_path)
    
    state_dict = reverse_map_state_dict_qwen(state_dict)


    model = Qwen2ForCausalLM.from_pretrained(original_path)
    original_state_dict = model.state_dict()
    
    vocab_size = len(state_dict['model.embed_tokens.weight'])
    original_state_dict['model.embed_tokens.weight'][:vocab_size] = state_dict['model.embed_tokens.weight']
    original_state_dict['lm_head.weight'][:vocab_size] = state_dict['lm_head.weight']

    state_dict['model.embed_tokens.weight'] = original_state_dict['model.embed_tokens.weight']
    state_dict['lm_head.weight'] = original_state_dict['lm_head.weight']
    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(consolidated_checkpoint_path)

    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chkpt", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("original_path", type=str)

    args = parser.parse_args()

    main(args.chkpt, args.output, args.original_path)
