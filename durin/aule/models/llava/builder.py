#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration, AutoProcessor
import torch
from aule.models.llava import *
from aule.models.llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from aule.models.llava.multimodal_encoder.builder import build_vision_tower
from aule.models.llava.multimodal_projector.builder import build_vision_projector

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import transformers
import copy
import re
from typing import List, Dict, Any, Optional

from aule.groundingdino.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from aule.groundingdino.util.misc import coordinate_to_encoding
from aule.groundingdino.models.GroundingDINO.utils import gen_sineembed_for_position, MLP
from aule.models.llava.language_model.llava_qwen import LlavaQwenForCausalLM
from aule.models.llava.multimodal_projector.builder import vision_projector_with_pos_proj
import math

@dataclass
class DefaultLLMArguments:
    huggingface_llava: bool = False
    huggingface_model: dict[str, str] | None = None
    lmm_max_token_length: int = 1600
    num_region_caption: int = 16
    lmm_layers: int = 1
    lmm_connector: str | None = "ef21d35ec3004407a1740453640bce3e"
    lmm_connector_prefix: str | None = "mm_projector"
    use_plora: bool = False
    use_lora: bool = True
    lmm_new_layer_insert_type: str = "all"
    use_p5_input: bool = False
    use_p4_input: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: int = 0
    num_lmm_new_layers: int = 6
    use_lmm_cross_attn: bool = True
    use_image_level_cross_attn: bool = False
    use_query_input: bool = False
    feature_map_size: int = 27
    use_constrast_conv: bool = False
    vis: bool = False
    use_pretrained_projector: bool = False
    lmm_region_loss_weight: float = 1.0
    lmm_image_loss_weight: float = 1.0
    huggingface_llava: bool = False

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            if 'qwen' in model_name.lower():
                from aule.models.llava.language_model.llava_qwen import LlavaQwenConfig
                lora_cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            else:
                from aule.models.llava.language_model.llava_llama import LlavaConfig
                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            if lora_cfg_pretrained.load_full_model:
                from aule.models.llava.multimodal_projector.builder import vision_projector_with_pos_proj
                model.model.mm_projector = vision_projector_with_pos_proj(model.model.config.hidden_size, [model.model.mm_projector])
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            msg = model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
        elif model_base is not None:
            # this may be mm projector only
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            elif 'qwen' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            elif 'qwen' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len



def load_mix_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    # Load LLaVA model
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
    if 'lora' in model_name.lower() and model_base is not None:
        if 'qwen' in model_name.lower():
            from aule.models.llava.language_model.llava_qwen import LlavaQwenConfig
            lora_cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        model.config.mm_vision_tower = lora_cfg_pretrained.mm_vision_tower
        model.config.vision_tower_weight_path = lora_cfg_pretrained.vision_tower_weight_path
        model.config.load_ram = lora_cfg_pretrained.load_ram
        model.config.grounding_dino_config = lora_cfg_pretrained.grounding_dino_config
        model.config.bert_base_path = lora_cfg_pretrained.bert_base_path
        model.config.repeat_num = lora_cfg_pretrained.repeat_num
        model.config.plain_projector = lora_cfg_pretrained.plain_projector
        model.config.use_mm_proj = True
        model.config.mm_projector_type = lora_cfg_pretrained.mm_projector_type
        model.config.mm_hidden_size = lora_cfg_pretrained.mm_hidden_size
        model.config.mm_vision_select_layer = lora_cfg_pretrained.mm_vision_select_layer
        model.config.mm_vision_select_feature = lora_cfg_pretrained.mm_vision_select_feature
        model.config.mm_patch_merge_type = lora_cfg_pretrained.mm_patch_merge_type
        model.config.load_full_model = lora_cfg_pretrained.load_full_model

        if model.config.mm_vision_tower == 'grounding_dino_mixed':
            model.model.vision_tower = build_vision_tower(model.config, model.model.vision_tower)
            model.model.mm_projector = build_vision_projector(model.config,)

        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        if lora_cfg_pretrained.load_full_model:
            from aule.models.llava.multimodal_projector.builder import vision_projector_with_pos_proj
            model.model.mm_projector = vision_projector_with_pos_proj(model.model.config.hidden_size, [model.model.mm_projector])
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        msg = model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        
    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

def gen_sineembed_for_position_2d(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, 0] * scale
    y_embed = pos_tensor[:, 1] * scale
    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=1)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, 2] * scale
        pos_w = w_embed[:, None] / dim_t
        pos_w = torch.stack((pos_w[:, 0::2].sin(), pos_w[:, 1::2].cos()), dim=2).flatten(1)

        h_embed = pos_tensor[:, 3] * scale
        pos_h = h_embed[:, None] / dim_t
        pos_h = torch.stack((pos_h[:, 0::2].sin(), pos_h[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=1)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class LlavaModel(torch.nn.Module):
    def __init__(self, llm_args: dict, d_model: int = 256, num_feature_levels: int = 4):
        super().__init__()
        if llm_args.huggingface_model is None:
            llm_args.huggingface_model = {
                "pretrained_model_name_or_path": "fushh7/LLMDet",
                "subfolder": "my_llava-onevision-qwen2-0.5b-ov-2",
            }
        self.lmm_args = llm_args
        self.lmm = LlavaQwenForCausalLM.from_pretrained(**llm_args.huggingface_model, mm_vision_tower="google/siglip-so400m-patch14-384")

        self.lmm.requires_grad_(False)
        self.lmm.config.use_cache = False

        if llm_args.use_constrast_conv:
            self.img_sep = torch.nn.Embedding(1, self.lmm.config.hidden_size)

        if llm_args.use_plora and hasattr(self.lmm.model, 'build_plora'):
            self.lmm.model.build_plora(llm_args.num_lmm_new_layers, llm_args.lmm_new_layer_insert_type, llm_args.lora_r, llm_args.lora_alpha, llm_args.lora_dropout)
        if llm_args.use_lora and hasattr(self.lmm.model, 'build_lora'):
            self.lmm.model.build_lora(llm_args.lora_r, llm_args.lora_alpha, llm_args.lora_dropout)
        if (llm_args.use_lmm_cross_attn and llm_args.num_region_caption > 0) or llm_args.use_image_level_cross_attn:
            if hasattr(self.lmm.model, 'build_cross_attention'):
                 self.lmm.model.build_cross_attention(llm_args.num_lmm_new_layers, llm_args.lmm_new_layer_insert_type, num_feature_levels)
            self.ref_point_head = MLP(d_model * 2, d_model, d_model, 2)

        self.lmm_tokenizer = transformers.AutoTokenizer.from_pretrained(
            **llm_args.huggingface_model,
            cache_dir=None, 
            model_max_length=llm_args.lmm_max_token_length,
            padding_side="right")
        
        self.IGNORE_INDEX = IGNORE_INDEX
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.lmm_tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.image_token_index = self.lmm_tokenizer.convert_tokens_to_ids("<image>")
        im_start, im_end = self.lmm_tokenizer.additional_special_tokens_ids
        self.unmask_tokens_idx =  [198, im_start, im_end]

        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        self.lmm_tokenizer.chat_template = chat_template
        self.roles = {"human": "user", "gpt": "assistant"}
        
        self.cache_input_id, self.cache_target = None, None
        
        if self.lmm_tokenizer.pad_token_id is None:
            self.lmm_tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.

        if hasattr(self.lmm, 'model') and hasattr(self.lmm.model, 'mm_projector'):
            del self.lmm.model.mm_projector
        if hasattr(self.lmm, 'model') and hasattr(self.lmm.model, 'vision_tower'):
            del self.lmm.model.vision_tower
        
        self.lmm.config.tokenizer_padding_side = self.lmm_tokenizer.padding_side
        self.lmm.config.tokenizer_model_max_length = llm_args.lmm_max_token_length

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', 'mlp2x_gelu')
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(256, self.lmm.config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.lmm.config.hidden_size, self.lmm.config.hidden_size))
            vision_projector = nn.Sequential(*modules)
        
        vision_projector = vision_projector_with_pos_proj(self.lmm.config.hidden_size, [vision_projector])
        if llm_args.use_pretrained_projector:
            mm_projector_weights = torch.load(llm_args.lmm_connector, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if llm_args.lmm_connector_prefix in k}
            vision_projector.load_state_dict(get_w(mm_projector_weights, llm_args.lmm_connector_prefix))

        if llm_args.use_p5_input or llm_args.use_p4_input:
            self.connector = vision_projector
        if llm_args.num_region_caption > 0 or llm_args.use_query_input:
            self.region_connector = copy.deepcopy(vision_projector)
        
        fm_size = llm_args.feature_map_size # default value
        yv, xv = torch.meshgrid([torch.arange(0, 1, 1/fm_size), torch.arange(0, 1, 1/fm_size)], indexing='ij')
        grid = torch.stack((xv, yv), 2).view(fm_size, fm_size, 2)
        self.grid_box = torch.cat([grid.reshape(-1, 2), grid.reshape(-1, 2)], dim=-1).flatten(0, 1) # Simplified construction            

        if llm_args.use_p4_input:
            self.image_seperate = nn.Embedding(1, self.lmm.config.hidden_size)
            yv, xv = torch.meshgrid([torch.arange(0, 1, 1/20), torch.arange(0, 1, 1/20)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(21, 21, 2)
            self.p5_grid_box = torch.cat([grid[:-1, :-1], grid[1:, 1:]], dim=-1).flatten(0, 1)
        
    def _prepare_region_caption_inputs(
        self,
        decoder_inputs_dict: Dict[str, Any],
        targets: List[Any],
        hidden_states: torch.Tensor | None = None,
        matching_bbox_preds: torch.Tensor | None = None,
        assign_results: List | None = None,
        selected_boxes: torch.Tensor | None = None,
        selected_queries: torch.Tensor | None = None,
        query_lens: list[int] | None = None,
        get_label: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Prepares the dictionary of inputs for the LMM for region-level captions."""
        if selected_boxes is None and selected_queries is None:
            matching_bbox_preds = box_cxcywh_to_xyxy(matching_bbox_preds)
            selected_queries, selected_boxes, query_lens = [], [], []
            for j in range(matching_bbox_preds.shape[0]):
                if not targets[j]['region_conversations']['conversations']:
                    query_lens.append(0)
                    continue
                
                gt_bboxes = targets[j]['boxes']
                device = gt_bboxes.device
                
                index_mapping = torch.zeros(len(gt_bboxes), device=device, dtype=torch.long)
                gt_inds = assign_results[j]["gt_inds"]
                valid_gt_mask = gt_inds > 0
                index_mapping[gt_inds[valid_gt_mask] - 1] = torch.arange(len(gt_inds), device=device)[valid_gt_mask]
                
                query_index = index_mapping[targets[j]['region_conversations']['box_index']]
                
                selected_boxes.append(matching_bbox_preds[j][query_index])
                selected_queries.append(hidden_states[j][query_index])
                query_lens.append(len(query_index))

            if not selected_queries:
                return None

            selected_boxes = torch.cat(selected_boxes).unsqueeze(1)
            selected_queries = torch.cat(selected_queries).unsqueeze(1)
        
        # Prepare text inputs
        region_conversations = [conv for sample in targets for conv in sample['region_conversations']['conversations']]
        input_ids = [torch.tensor(conv['input_id'], dtype=torch.long) for conv in region_conversations]
        max_len = self.lmm_args.lmm_max_token_length
        if get_label:
            labels = [torch.tensor(conv['label'], dtype=torch.long) for conv in region_conversations]
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            labels = labels[:, :max_len]
        else:
            labels = torch.Tensor()


        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.lmm_tokenizer.pad_token_id)
        
        # Truncate
        input_ids = input_ids[:, :max_len]
        attention_mask = input_ids.ne(self.lmm_tokenizer.pad_token_id)
        
        # Prepare visual features
        features = self.region_connector([selected_queries])
        features = features + self.region_connector.forward_pos(gen_sineembed_for_position(selected_boxes))
        
        # Prepare cross-attention inputs if needed
        cross_attention_input = None
        if self.lmm_args.use_lmm_cross_attn:
            repeat_num = torch.tensor(query_lens, device=selected_queries.device)
            new_memory = torch.repeat_interleave(decoder_inputs_dict['memory'], repeat_num, dim=0)
            new_valid_ratios = torch.repeat_interleave(decoder_inputs_dict['valid_ratios'], repeat_num, dim=0)
            new_memory_mask = torch.repeat_interleave(decoder_inputs_dict['memory_mask'], repeat_num, dim=0) if decoder_inputs_dict.get('memory_mask') is not None else None
            
            reference_points = box_xyxy_to_cxcywh(selected_boxes)
            reference_points_input = reference_points[:, :, None] * torch.cat([new_valid_ratios, new_valid_ratios], -1)[:, None]
            query_sine_embed = coordinate_to_encoding(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            
            cross_attention_input = {
                'query_pos': query_pos, 'memory': new_memory, 'memory_mask': new_memory_mask,
                'spatial_shapes': decoder_inputs_dict['spatial_shapes'], 'level_start_index': decoder_inputs_dict['level_start_index'],
                'reference_points_input': reference_points_input, 'half': False,
            }

        return dict(
            input_ids=input_ids.to(features.device),
            labels=labels.to(features.device) if get_label else None,
            attention_mask=attention_mask.to(features.device),
            image_queries=features,
            query_masks=torch.ones((len(input_ids), 1), device=features.device, dtype=torch.bool),
            cross_attention_input=cross_attention_input,
        )

    def _prepare_image_caption_inputs(
        self,
        decoder_inputs_dict: Dict[str, Any],
        targets: List[Any]
    ) -> Dict[str, Any]:
        """Prepares the dictionary of inputs for the LMM for image-level descriptions."""
        
        # Prepare text inputs
        input_ids = [torch.tensor(s['conversations']['input_id'], dtype=torch.long) for s in targets]
        labels = [torch.tensor(s['conversations']['label'], dtype=torch.long) for s in targets]

        if self.lmm_args.use_constrast_conv:
            input_ids += [torch.tensor(s['contrast_conv'][1]['input_id'], dtype=torch.long) for s in targets if 'contrast_conv' in s]
            labels += [torch.tensor(s['contrast_conv'][1]['label'], dtype=torch.long) for s in targets if 'contrast_conv' in s]

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.lmm_tokenizer.pad_token_id)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        max_len = self.lmm_args.lmm_max_token_length
        input_ids = input_ids[:, :max_len]
        labels = labels[:, :max_len]
        attention_mask = input_ids.ne(self.lmm_tokenizer.pad_token_id)
        
        # Prepare visual features
        image_queries, query_masks = [], []
        memory = decoder_inputs_dict['memory']
        for i in range(len(memory)):
            valid_H_p5, valid_W_p5 = (decoder_inputs_dict['spatial_shapes'][-2] * decoder_inputs_dict['valid_ratios'][i, -2]).round().int()
            p5_map = memory[i][decoder_inputs_dict['level_start_index'][-2]: decoder_inputs_dict['level_start_index'][-1]]
            p5_map = p5_map.reshape(*decoder_inputs_dict['spatial_shapes'][-2], memory.shape[-1]).permute(2, 0, 1)
            p5_map = F.interpolate(p5_map.unsqueeze(0), size=(20, 20), mode='bilinear')[0]
            p5_map = self.connector([p5_map.permute(1, 2, 0)]).flatten(0, 1) + self.connector.forward_pos(gen_sineembed_for_position_2d(self.p5_grid_box.to(memory.device)))

            if self.lmm_args.use_p4_input:
                valid_H_p4, valid_W_p4 = (decoder_inputs_dict['spatial_shapes'][-3] * decoder_inputs_dict['valid_ratios'][i, -3]).round().int()
                p4_map = memory[i][decoder_inputs_dict['level_start_index'][-3]: decoder_inputs_dict['level_start_index'][-2]]
                p4_map = p4_map.reshape(*decoder_inputs_dict['spatial_shapes'][-3], memory.shape[-1]).permute(2, 0, 1)
                fm_size = self.lmm_args.feature_map_size
                p4_map = F.interpolate(p4_map.unsqueeze(0), size=(fm_size, fm_size), mode='bilinear')[0]
                p4_map = self.connector([p4_map.permute(1, 2, 0)]).flatten(0, 1) + self.connector.forward_pos(gen_sineembed_for_position_2d(self.grid_box.to(memory.device)))
                
                final_query = torch.cat([p5_map, self.image_seperate.weight, p4_map])
            else:
                final_query = p5_map

            image_queries.append(final_query)
            query_masks.append(torch.ones(len(final_query), device=memory.device, dtype=torch.bool))

        cross_attention_input = None
        if self.lmm_args.use_image_level_cross_attn:
            ref_points_input = self.grid_box.clone().unsqueeze(0).repeat(len(memory), 1, 1).to(memory.dtype).to(memory.device)
            ref_points_input = ref_points_input.unsqueeze(2).repeat(1, 1, decoder_inputs_dict['valid_ratios'].shape[1], 1)
            query_sine_embed = coordinate_to_encoding(ref_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            cross_attention_input = {
                'query_pos': query_pos, 'memory': memory, 'memory_mask': decoder_inputs_dict['memory_mask'],
                'spatial_shapes': decoder_inputs_dict['spatial_shapes'], 'level_start_index': decoder_inputs_dict['level_start_index'],
                'reference_points_input': ref_points_input, 'half': False,
            }
        
        return dict(
            input_ids=input_ids.to(memory.device),
            labels=labels.to(memory.device),
            attention_mask=attention_mask.to(memory.device),
            image_queries=image_queries,
            query_masks=query_masks,
            cross_attention_input=cross_attention_input
        )
    
    @torch.autocast('cuda', dtype=torch.half)
    def forward(
        self,
        query_predictions: torch.Tensor,
        decoder_inputs_dict: Dict[str, Any],
        all_layers_matching_bbox_preds: torch.Tensor,
        all_stage_assign_result: List[List],
        targets: List[Any],
    ) -> Dict[str, Any]:
        """
        Prepares inputs and runs the LMM forward pass.

        Args:
            query_predictions (torch.Tensor): Predicted queries.
            decoder_inputs_dict (Dict): Dictionary with inputs from the decoder.
                                        Expected to contain 'memory', 'valid_ratios', etc.
            all_layers_matching_bbox_preds (torch.Tensor): Predicted bboxes from all decoder layers.
            all_stage_assign_result (List[List]): Assigner results for all decoder layers.
            targets (List[Any]): List of data samples for the batch.

        Returns:
            Dict[str, Any]: A dictionary containing the outputs from the LMM for region and/or
                            image level captions. The outputs themselves are objects that
                            typically contain the loss.
        """
        outputs = {}
        self.lmm.eval() # As per original code

        # 1. Process Region-level Captions
        if self.lmm_args.num_region_caption > 0:
            outputs['region_lmm_outputs'] = []
            num_layers = len(query_predictions)
            for i in range(1, num_layers + 1):
                lmm_input_dict = self._prepare_region_caption_inputs(
                    matching_bbox_preds=all_layers_matching_bbox_preds[-i].clone().detach(),
                    assign_results=all_stage_assign_result[-i],
                    hidden_states=query_predictions[-i],
                    decoder_inputs_dict=decoder_inputs_dict,
                    targets=targets,
                )
                
                if lmm_input_dict is None:
                    continue

                lmm_output = self.lmm.detection_forward(**lmm_input_dict)
                outputs['region_lmm_outputs'].append(lmm_output)
        
        # 2. Process Image-level Captions
        if self.lmm_args.use_p5_input or self.lmm_args.use_p4_input:
            lmm_input_dict = self._prepare_image_caption_inputs(
                decoder_inputs_dict=decoder_inputs_dict,
                targets=targets
            )
            
            lmm_output = self.lmm.detection_forward(**lmm_input_dict)
            outputs['image_lmm_output'] = lmm_output
            
        return outputs

    def loss(self, lmm_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Calculates the final weighted loss from the LMM outputs.

        Args:
            lmm_outputs (Dict[str, Any]): The dictionary returned by the forward pass.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of computed loss values.
        """
        losses = {}

        # 1. Region Loss
        if 'region_lmm_outputs' in lmm_outputs:
            region_loss_weight = self.lmm_args.lmm_region_loss_weight
            num_outputs = len(lmm_outputs['region_lmm_outputs'])
            for i, lmm_output in enumerate(lmm_outputs['region_lmm_outputs']):
                # The loop in forward was from 1 to num_layers, and we stored them in order.
                # The original code used `d{i}` where i went from 1 to num_layers, so we replicate that.
                layer_idx = num_outputs - i 
                losses[f'd{layer_idx}.loss_lmm_region'] = lmm_output.loss * region_loss_weight

        # 2. Image Loss
        if 'image_lmm_output' in lmm_outputs:
            image_loss_weight = self.lmm_args.lmm_image_loss_weight
            losses['loss_lmm_image'] = lmm_outputs['image_lmm_output'].loss * image_loss_weight
            
        return losses
    
    def _create_prompt(self, prompt: str) -> dict[str, Any]:
        input_id = []
        input_id += self.lmm_tokenizer.apply_chat_template([{"role" : "system", "content" : "You are a helpful assistant."}])
        role = self.roles.get("human", "user")
        conv = [{"role": role, "content": prompt}]
        encode_id = self.lmm_tokenizer.apply_chat_template(conv)
        input_id += encode_id
        
        return {"input_id": input_id}

    
    def predict_region_level(
        self,
        query_embeddings: torch.Tensor,
        query_boxes_xyxy: torch.Tensor,
        prompts: Optional[List[List[str]]] = None,
        targets: Optional[List[Any]] = None,
        decoder_inputs_dict: Optional[Dict[str, Any]] = None,
    ) -> List[List[str]]:
        """
        Generates text descriptions for a batch of query embeddings and their corresponding boxes.

        Args:
            query_embeddings (torch.Tensor): A tensor of query embeddings with shape (batch_size, num_queries, d_model).
            query_boxes_xyxy (torch.Tensor): A tensor of corresponding bounding boxes in [x1, y1, x2, y2]
                                             format, normalized to [0, 1]. Shape: (batch_size, num_queries, 4).
            prompts (List[str]): A list of text prompts, one for each query.
            decoder_inputs_dict (Optional[Dict[str, Any]]): Dictionary with decoder memory and context,
                                                            required if using cross-attention. Assumed to be
                                                            for a single image context for all queries.
            generation_kwargs (Optional[Dict[str, Any]]): A dictionary of arguments for the `generate`
                                                          function (e.g., `max_new_tokens`, `num_beams`).

        Returns:
            List[List[str]]: A list of lists of generated answers. The outer list corresponds to the input
                             queries, and each inner list contains one or more generated strings based on
                             `num_return_sequences`.
        """
        assert len(query_embeddings) == len(query_boxes_xyxy), \
            "Mismatch in number of queries, boxes, and prompts."
            
        if prompts:
            assert len(query_boxes_xyxy) == len(prompts), \
                "Mismatch in number of queries, boxes, and prompts."
                
        if targets:
            assert len(query_boxes_xyxy) == len(targets), \
                "Mismatch in number of queries, boxes, and targets."
                    
        if prompts is None and targets is None:
            raise ValueError("Either prompts or targets must be provided.")

        with torch.inference_mode():
            targets = [
                {
                    "region_conversations": {
                        "conversations": [self._create_prompt(prompt) for prompt in prompts[idx]],
                    },
                } for idx, _ in enumerate(prompts)
            ] if targets is None else targets
            
            num_region_conversations = sum([len(_["region_conversations"]["conversations"]) for _ in targets])
            
            assert num_region_conversations == query_embeddings.shape[0]*query_embeddings.shape[1], \
                "Mismatch in number of queries, boxes, and prompts."
            
            lmm_input_dict = self._prepare_region_caption_inputs(
                selected_boxes=query_boxes_xyxy.reshape(-1, 4).unsqueeze(1).clone().detach(),
                selected_queries=query_embeddings.reshape(-1, query_embeddings.shape[-1]).unsqueeze(1),
                decoder_inputs_dict=decoder_inputs_dict,
                targets=targets,
                query_lens=[len(_) for _ in query_embeddings],
                get_label=False,
            )
            lmm_output = self.lmm.generate(
                **lmm_input_dict,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
                top_p = None,
                top_k = None,
                temperature=None,
                use_cache=True,
            )
        lmm_output = [res.split("assistant\n")[-1].strip().strip('.') for res in self.lmm_tokenizer.batch_decode(lmm_output, skip_special_tokens=True)]
        return lmm_output


class LLavaProcessor:    
    def __init__(self, huggingface_model: dict, ignore_index: int = IGNORE_INDEX):
        self.processor = AutoProcessor.from_pretrained(**huggingface_model)
        self.image_token = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        self.ignore_index = ignore_index
        
    def _expand_image_tokens(
        self,
        text: torch.Tensor,
        attention_mask: torch.Tensor,
        image_sizes,
        height: int,
        width: int,
        special_token: str,
        num_frames: int = 1,
        labels: torch.Tensor | None = None,
    ):
        prompt_strings = []
        attention_mask = [att_mask.tolist() if isinstance(att_mask, torch.Tensor) else att_mask for att_mask in attention_mask]
        if labels:
            labels = [lab.tolist() if isinstance(lab, torch.Tensor) else lab for lab in labels]
        for b_idx, sample in enumerate([t.tolist() if isinstance(t, torch.Tensor) else t for t in text]):
            while special_token in sample:
                image_size_list = next(image_sizes)
                original_size = image_size_list[0] if num_frames != 1 else image_size_list
                if not isinstance(original_size, (list, tuple)):
                    # cast to list to avoid numerical precision errors when calculating unpadding
                    original_size = original_size.tolist()
                orig_height, orig_width = original_size
                num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height, width)
                if self.processor.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                for idx, token_id in enumerate(sample):
                    if token_id == special_token:
                        break
                else:
                    idx = -1
                if idx >= 0:
                    for _ in range(num_image_tokens*num_frames):
                        sample.insert(idx+1, -999)
                        attention_mask[b_idx].insert(idx+1, 1)
                        if labels is not None:
                            labels[b_idx].insert(idx+1, self.ignore_index)
                    sample.pop(idx)
                    attention_mask[b_idx].pop(idx)
                    if labels is not None:
                        labels[b_idx].pop(idx)
            prompt_strings.append(sample)
        text = [[token_id if token_id != -999 else special_token for token_id in sample] for sample in prompt_strings]
        max_dim_pad = max([len(value) for value in text])
        text = torch.stack([torch.nn.functional.pad(torch.tensor(value), (0, max_dim_pad - len(value)), value=self.processor.tokenizer.pad_token_id) for value in text])
        attention_mask = torch.stack([torch.nn.functional.pad(torch.tensor(value), (0, max_dim_pad - len(value)), value=0) for value in attention_mask])
        if labels is not None:
            labels = torch.stack([torch.nn.functional.pad(torch.tensor(value), (0, max_dim_pad - len(value)), value=self.ignore_index) for value in labels])
        return text, attention_mask, labels
        
    def __call__(self, images: torch.Tensor | list[torch.Tensor], image_sizes: list[tuple[int, int]], attention_mask: torch.Tensor = None, input_ids: torch.Tensor = None, text: list[str] = None, labels: list[int] | None = None, **kwargs):
        if text:
            return self.processor(images=images, text=text, image_sizes=image_sizes, **kwargs)

        image_sizes_iter = iter(image_sizes)
        height, width = images.shape[-2], images.shape[-1]
        text, attention_mask, labels = self._expand_image_tokens(input_ids, attention_mask, image_sizes_iter, height, width, self.image_token, labels=labels)
        text_inputs = {
            "input_ids": text.to(images.device),
            "attention_mask": attention_mask.to(images.device)
        }
        if labels is not None:
            text_inputs["labels"] = labels.to(images.device)
        return transformers.BatchFeature(data={"pixel_values": images, "image_sizes": image_sizes, **text_inputs, **kwargs})
            
    
def build_llm(
    args = None,
    d_model: int = 256,
    num_feature_levels: int = 4,
    config=None,
) -> LlavaModel:
    if args is None:
        args = {}
    llm_args = DefaultLLMArguments(
        **args
    )
    
    if not llm_args.huggingface_llava and not config:
        return LlavaModel(
            llm_args,
            d_model=d_model,
            num_feature_levels=num_feature_levels,
        )
    if config:
        return LlavaOnevisionForConditionalGeneration(config)
    return LlavaOnevisionForConditionalGeneration.from_pretrained(
        **llm_args.huggingface_model,
    )
    
def build_processor(args = None):
    if args is None:
        args = {"pretrained_model_name_or_path": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"}
    return LLavaProcessor(huggingface_model=args)