#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .modeling_qwen2 import Qwen2Config, Qwen2Model, Qwen2ForCausalLM


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def init_weights(self):
        pass

    def get_model(self):
        return self.model

    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     images: Optional[torch.FloatTensor] = None,
    #     image_sizes: Optional[List[List[int]]] = None,
    #     tags: Optional[List[str]] = None,
    #     return_dict: Optional[bool] = None,
    #     modalities: Optional[List[str]] = ["image"],
    #     dpo_forward: Optional[bool] = False,
    #     cache_position=None,
    #     im_mask=None,
    #     cross_attention_input=None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     if inputs_embeds is None:
    #         (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes, tags)

    #     if dpo_forward:
    #         outputs = self.model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    #         hidden_states = outputs[0]
    #         logits = self.lm_head(hidden_states)
    #         return logits, labels

    #     else:
    #         return super().forward(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             labels=labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #             im_mask=im_mask,
    #             cross_attention_input=cross_attention_input,
    #         )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_queries: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        query_masks=None,
        cross_attention_input=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, new_im_mask_padded) = self.prepare_inputs_labels_for_multimodal_detection(input_ids.clone(), position_ids, attention_mask, past_key_values, labels, image_queries, query_masks)
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_mask=new_im_mask_padded,
            cross_attention_input=cross_attention_input,
        )
        
    def detection_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_queries: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        query_masks=None,
        cross_attention_input=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, new_im_mask_padded) = self.prepare_inputs_labels_for_multimodal_detection(input_ids, position_ids, attention_mask, past_key_values, labels, image_queries, query_masks)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_mask=new_im_mask_padded,
            cross_attention_input=cross_attention_input,
        )
        
        outputs.cross_attention_input = cross_attention_input
        
        return outputs
    
    @torch.no_grad()
    def detection_generate(
        self,
        input_ids: torch.LongTensor = None,
        image_queries: Optional[torch.FloatTensor] = None,
        query_masks=None,
        cross_attention_input=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        (input_ids, position_ids, attention_mask, _, inputs_embeds, labels, new_im_mask_padded) = self.prepare_inputs_labels_for_multimodal_detection(input_ids, position_ids, attention_mask, None, None, image_queries, query_masks)

        return super().generate(position_ids=position_ids, inputs_embeds=inputs_embeds, im_mask=new_im_mask_padded, cross_attention_input=cross_attention_input, **kwargs)

        
    @torch.no_grad()
    def custom_detection_generate(
        self,
        input_ids: torch.LongTensor = None,
        image_queries: Optional[torch.FloatTensor] = None,
        query_masks=None,
        cross_attention_input=None,
        max_new_tokens: int | None = 32,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        end_token_reached = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        new_tokens = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)
        for _ in range(max_new_tokens):
            outputs = self.detection_forward(input_ids, image_queries=image_queries, query_masks=query_masks, cross_attention_input=cross_attention_input, **kwargs)
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            new_tokens = torch.cat([new_tokens, next_token.unsqueeze(-1)], dim=-1)
            
            if "attention_mask" in kwargs:
                attention_mask = kwargs["attention_mask"]
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                kwargs["attention_mask"] = attention_mask
            
            for i in range(input_ids.shape[0]):
                if next_token[i] == self.config.eos_token_id:
                    end_token_reached[i] = True
            if end_token_reached.all():
                break
        return new_tokens

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[tuple] = None,
        **kwargs
    ) -> dict:
        """
        This method is called by `.generate()` at each step.
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]
            kwargs.pop("cross_attention_input", None)
            kwargs.pop("query_masks", None)
            kwargs.pop("image_queries", None)

        model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": kwargs.get("attention_mask"),
            "image_queries": kwargs.get("image_queries"),
            "query_masks": kwargs.get("query_masks"),
            "cross_attention_input": kwargs.get("cross_attention_input"),
        })


        return model_inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
