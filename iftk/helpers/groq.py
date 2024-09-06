# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import AsyncIterator, Callable

import groq

from iftk.channel import AsyncChannel
from iftk.system import System


async def groq_sentence_stream(llm_stream: groq._client.AsyncStream) -> AsyncIterator:
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
            if sentence:
                yield sentence
            sentence = ""


class GroqChannel(AsyncChannel):
    def __init__(
        self,
        sentence_stream: groq._client.AsyncStream,
        notify_readable: Callable[[None], None] = None,
    ) -> None:
        super().__init__(notify_readable)
        self.sentence_stream = sentence_stream
        self.groq_stream = groq_sentence_stream(sentence_stream=sentence_stream)

    async def read(self) -> AsyncIterator[bytes]:
        yield await anext(self.groq_stream)


class ElevenlabsSystem(System):
    async def create_async_channel(
        self,
        sentence_stream: groq._client.AsyncStream,
        notify_readable: Callable[[None], None] = None,
        **kwargs,
    ) -> AsyncChannel:
        groq_channel = GroqChannel(sentence_stream=sentence_stream)
        return groq_channel
