# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from collections import deque
from typing import Any, AsyncIterator, Callable, Iterator

from context import iftk
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs.play import stream

from iftk.channel import DequeChannel
from iftk.helpers import deepgram, elevenlabs, groq, pyaudio
from iftk.system import System

DOTENV = dotenv_values(".env")
GROQ_API_KEY = DOTENV["GROQ_API_KEY"]
DEEPGRAM_API_KEY = DOTENV["DEEPGRAM_API_KEY"]
ELEVENLABS_API_KEY = DOTENV["ELEVEN_API_KEY"]
CHUNK = 512
RATE = 16000
model_id = "llama-3.1-8b-instant"
messages = [{"role": "system", "content": "Answer to the user in a few sentences."}]


class RemoteChannel(DequeChannel):
    def __init__(
        self,
        notify_readable: Callable[[None], None] = None,
    ) -> None:
        super().__init__(notify_readable)
        self.groq_client = groq.groq.AsyncClient(api_key=GROQ_API_KEY)
        self.elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.q = deque()

    async def read_to_stream(self) -> AsyncIterator[bytes]:
        """Convert the deque into an AsyncIterator."""
        while True:
            while not self.q:
                continue

            while self.q:
                try:
                    message = self.q.popleft()
                    if message is not None:
                        yield message
                except IndexError:
                    continue

    async def read(self) -> AsyncIterator[Iterator[bytes]]:
        self.deepgram_stream = deepgram.deepgram_stream(
            key=DEEPGRAM_API_KEY, audio_stream=self.read_to_stream()
        )
        async for user_message in self.deepgram_stream:
            messages.append({"role": "user", "content": user_message})
            llm_stream = await self.groq_client.chat.completions.create(
                messages=messages, model=model_id, stream=True
            )
            groq_sentence_stream: AsyncIterator = groq.groq_sentence_stream(
                llm_stream=llm_stream
            )
            async for sentence in elevenlabs.eleven_stream(
                sentences=groq_sentence_stream, eleven_client=self.elevenlabs_client
            ):
                yield sentence


class RemoteInferenceSystem(System):
    async def create_async_channel(
        self, notify_readable: Callable[[None], None] = None, **kwargs
    ) -> iftk.AsyncChannel:
        super().__init__()
        return RemoteChannel()


async def main():
    mic_stream: AsyncIterator = pyaudio.microphone(rate=RATE, frames_per_buffer=CHUNK)
    system = RemoteInferenceSystem()
    channel = await system.create_async_channel()
    async for audio_chunk in mic_stream:
        channel.write(audio_chunk)
        async for output in channel.read():
            if output:
                await asyncio.to_thread(stream, output)


if __name__ == "__main__":
    asyncio.run(main())
