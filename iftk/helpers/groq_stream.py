# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import AsyncIterator
import groq


async def GroqProcessor(llm_stream: groq._client.AsyncStream) -> AsyncIterator:
    """An AsyncIterator wrapper for the groq generation stream.

    Args:
        llm_stream (groq._client.AsyncStream): The generation stream from the groq library.

    Yields:
        sentence (str): A sentence of text.
    """
    sentence = ""
    async for token in llm_stream:
        new_text = token.choices[0].delta.content
        if new_text is None:
            continue
        if new_text != "" and "." not in new_text:
            sentence += new_text
        else:
            sentence += new_text
            yield sentence
            sentence = ""
