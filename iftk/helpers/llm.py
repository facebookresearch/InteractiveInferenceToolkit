# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from asyncio import to_thread
from typing import AsyncIterator

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TextIteratorStreamer)


async def llm_stream(
    model_id: str, messages: list, quantize: bool = True, max_new_tokens: int = 50
) -> AsyncIterator[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    streamer = TextIteratorStreamer(tokenizer=tokenizer)
    if quantize:
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenized_messages = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    await to_thread(
        model.generate,
        tokenized_messages,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
    )
    for new_text in streamer:
        if new_text:
            yield new_text
