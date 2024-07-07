# usage: python export/test_coreml_gpt2.py --prompt=<PROMPT_STR>

import argparse

import coremltools as ct
import numpy as np
from scipy.special import softmax
from transformers import GPT2Tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)

    args = parser.parse_args()

    return args


def load_model():
    gpt2_coreml_model = ct.models.MLModel("../exported/GPT2.mlpackage")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return gpt2_coreml_model, gpt2_tokenizer


def resize_to_128(input_tensor, max_length=128):
    """Resize input tensor to shape (1x128).
    We need to resize the input tensor, otherwise we get `MultiArray shape (1 x N) does not match the shape (1 x 128) specified in the model description` as GPT2 expects a (1 x 128) input.

    input: tensor, shape=(1 x N)
    output: tensor, shape=(1 x 128)
    """
    assert len(input_tensor.shape) == 2
    assert input_tensor.shape[0] == 1

    tensor_length = input_tensor.shape[-1]

    if tensor_length > max_length:
        # Truncate to `max_length`
        resized_tensor = input_tensor[:max_length]
    else:
        # Pad to `max_length`
        resized_tensor = np.pad(
            input_tensor,
            ((0, 0), (0, max_length - tensor_length)),
            mode="constant",
            constant_values=0,
        )

    return resized_tensor


def generate_next_token(model, tokenizer, inputs):
    # CoreML GPT2 requires input size (1, 128).
    # Cast to f64 fixes `Error: value type not convertible`.
    input_ids_f64 = resize_to_128(inputs["input_ids"].astype(np.float64))
    attention_mask = resize_to_128(inputs["attention_mask"].astype(np.float64))
    input_dict = {"input_ids": input_ids_f64, "attention_mask": attention_mask}

    output = model.predict(input_dict)

    # Next token prediction
    # TODO: Why is the last token logit not equal to current length?
    last_token_logits = output["logits"][0, inputs["input_ids"].shape[1] - 1, :]
    last_token_softmax = softmax(last_token_logits, axis=-1)
    next_token_id = np.argmax(last_token_softmax)
    next_token = tokenizer.decode(next_token_id)

    return next_token


def continue_generating(inputs, max_length=128):
    # TODO: How do we determine EOS termination for GPT2? Currently, it will repeat some chunks of text until `max_length` is reached.
    return inputs["input_ids"].shape[-1] < max_length


if __name__ == "__main__":
    args = get_args()
    model, tokenizer = load_model()

    current_prompt = args.prompt
    inputs = tokenizer(current_prompt, return_tensors="np")
    while continue_generating(inputs):
        # Autoregressive update
        next_token = generate_next_token(model, tokenizer, inputs)
        current_prompt += next_token
        print(current_prompt)

        inputs = tokenizer(current_prompt, return_tensors="np")
