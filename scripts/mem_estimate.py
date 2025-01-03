from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5)

from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

def get_models(name: str):
    model = load_flow_model2(name, device="cpu")
    return model

def main():
    dit = get_models(name="flux-dev")
    for n, param in dit.named_parameters():
        if 'txt_attn' not in n:
            param.requires_grad = False
    estimate_zero3_model_states_mem_needs_all_live(dit, num_gpus_per_node=1, num_nodes=1)

if __name__ == "__main__":
    main()
