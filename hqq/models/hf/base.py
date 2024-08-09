# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import transformers
from accelerate import init_empty_weights
from ..base import BaseHQQModel, BasePatch


class BaseHQQHFModel(BaseHQQModel):
    # Save model architecture
    @classmethod
    def cache_model(cls, model, save_dir):
        model.config.save_pretrained(save_dir)

    # Create empty model from config
    @classmethod
    def create_model(cls, save_dir, kwargs):
        model_kwargs = {}
        for key in ["attn_implementation"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        config = transformers.AutoConfig.from_pretrained(
            cls.get_config_file(save_dir)
        )

        auto_class = transformers.AutoModel

        # Todo: add support for other auto models
        archs = config.architectures
        if len(archs) == 1 and ("CausalLM" in archs[0]):
            auto_class = transformers.AutoModelForCausalLM
        elif len(archs) == 1 and ("Seq2SeqLM" in archs[0]):
            auto_class = transformers.AutoModelForSeq2SeqLM
        elif len(archs) == 1 and ("M2M100ForConditionalGeneration" in archs[0]):
            auto_class = transformers.AutoModelForSeq2SeqLM

        with init_empty_weights():
            model = auto_class.from_config(config, **model_kwargs)

        return model


# Auto class used for HF models if no architecture was manually setup
class AutoHQQHFModel(BaseHQQHFModel, BasePatch):
    pass
