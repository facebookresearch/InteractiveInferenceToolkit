# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pyaudio


async def microphone(
    format: int,
    rate: int,
    frames_per_buffer: int,
    callback: pyaudio._StreamCallback,
):
    """A streaming-friendly PyAudio microphone interface.

    Args:
        format (int): The audio format, commonly pyaudio.paInt16
        rate (int): The audio recording sampling rate.
        frames_per_buffer (int): An arbitrarily chosen number of frames for the signals to be split into.
        callback (pyaudio._StreamCallback): A callback function, commonly in the form of an asyncio.Queue() thread-safe wrapper.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=format,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
        stream_callback=callback,
    )

    stream.start_stream()
    while stream.is_active():
        await asyncio.sleep(0.1)

    stream.stop_stream()
    stream.close()
