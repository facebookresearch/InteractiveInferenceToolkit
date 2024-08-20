# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from typing import AsyncIterator

import pyaudio


async def microphone(rate: int, frames_per_buffer: int) -> AsyncIterator[bytes]:
    """An AsyncIterator-friendly PyAudio microphone stream.

    Args:
        rate (int): Sampling rate for the microphone recording.
        frames_per_buffer (int): Chunk size of microphone recordings.

    Yields:
        bytes: A recorded audio chunk.
    """
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    pa = pyaudio.PyAudio()

    def put(in_data, frame_count, time_info, status):
        loop.call_soon_threadsafe(queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    async def get():
        while True:
            yield await queue.get()

    stream = pa.open(
        stream_callback=put,
        rate=rate,
        format=pyaudio.paInt16,
        channels=1,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )
    stream.start_stream()
    async for chunk in get():
        yield chunk
