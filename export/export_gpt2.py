from transformers import AutoTokenizer, GPT2LMHeadModel

from exporters.coreml import export
from exporters.coreml.models import GPT2CoreMLConfig

model_ckpt = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_ckpt, torchscript=True)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

coreml_config = GPT2CoreMLConfig(model.config, "text-generation")
mlmodel = export(tokenizer, model, coreml_config)
mlmodel.save("../exported/GPT2.mlpackage")
