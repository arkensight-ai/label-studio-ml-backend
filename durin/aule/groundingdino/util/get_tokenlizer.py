import os

from transformers import (AutoTokenizer, BertModel, BertTokenizer, AutoConfig,
                          RobertaModel)


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type, config_file: str | None = None):
    config = None
    if config_file is not None:
        config = AutoConfig.from_pretrained(config_file, local_files_only=True)
    if text_encoder_type == "bert-base-uncased" or (
        os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)
    ):
        if config:
            return BertModel._from_config(config)
        return BertModel.from_pretrained(text_encoder_type, add_pooling_layer=False)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
