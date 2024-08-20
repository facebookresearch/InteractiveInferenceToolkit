# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from typing import AsyncIterator

from context import iftk
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs.play import stream

from iftk.helpers import deepgram, elevenlabs, groq, pyaudio

DOTENV = dotenv_values(".env")
GROQ_API_KEY = DOTENV["GROQ_API_KEY"]
DEEPGRAM_API_KEY = DOTENV["DEEPGRAM_API_KEY"]
ELEVENLABS_API_KEY = DOTENV["ELEVEN_API_KEY"]
CHUNK = 512
RATE = 16000
model_id = "llama-3.1-8b-instant"
messages = [{"role": "system", "content": "Answer to the user in a few sentences."}]


async def main():
    groq_client = groq.groq.AsyncClient(api_key=GROQ_API_KEY)
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    mic_stream: AsyncIterator = pyaudio.microphone(rate=RATE, frames_per_buffer=CHUNK)
    deepgram_stream: AsyncIterator = deepgram.deepgram_stream(
        key=DEEPGRAM_API_KEY, audio_stream=mic_stream
    )
    async for user_message in deepgram_stream:
        if user_message:
            messages.append({"role": "user", "content": user_message})
            llm_stream = await groq_client.chat.completions.create(
                messages=messages, model=model_id, stream=True
            )
            groq_sentence_stream: AsyncIterator = groq.groq_sentence_stream(
                llm_stream=llm_stream
            )
            async for sentence in elevenlabs.eleven_stream(
                sentences=groq_sentence_stream, eleven_client=elevenlabs_client
            ):
                await asyncio.to_thread(stream, sentence)


asyncio.run(main())
