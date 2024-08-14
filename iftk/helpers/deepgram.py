# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json

import pyaudio
import websockets
from .microphone import microphone_async

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
DEEPGRAM_CHUNK = 8000
DEEPGRAM_RATE = 16000
SECONDS = 2
WSS_URL = "wss://api.deepgram.com/v1/listen?endpointing=500&encoding=linear16&sample_rate=16000&channels=1&interim_results=false"


async def deepgram_stream(key: str, transcription_queue: asyncio.Queue) -> None:
    """Put ASR results from Deepgram streaming API in an asyncio.Queue.

    Args:
        key (str): Your Deepgram API key
        transcription_queue (asyncio.Queue): A queue that stores the transcriptions as a
        "speech_final" message is detected from Deepgram.
    """
    extra_headers = {"Authorization": f"Token {key}"}
    audio_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(in_data, frame_count, time_info, status) -> tuple[None, int]:
        """A basic callback for Deepgram's stream."""
        loop.call_soon_threadsafe(audio_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    async with websockets.connect(
        WSS_URL,
        extra_headers=extra_headers,
    ) as ws:

        async def keep_alive(websocket: websockets.WebSocketClientProtocol):
            while True:
                keep_alive_msg = json.dumps({"type": "KeepAlive"})
                websocket.send(keep_alive_msg)
                await asyncio.sleep(3)

        keep_alive_task = asyncio.create_task(keep_alive(websocket=ws))

        async def sender(ws):
            while True:
                data = await audio_queue.get()
                await ws.send(data)

        async def receiver(ws):
            transcript = ""
            async for msg in ws:
                msg = json.loads(msg)

                if len(msg["channel"]["alternatives"][0]["transcript"]) > 0:
                    if transcript:
                        transcript += " "
                    transcript += msg["channel"]["alternatives"][0]["transcript"]
                if msg["speech_final"]:
                    yield transcript
                    transcript = ""

        microphone_task = asyncio.create_task(
            microphone_async(
                format=FORMAT,
                rate=RATE,
                frames_per_buffer=DEEPGRAM_CHUNK,
                callback=callback,
            )
        )
        sender_task = asyncio.create_task(sender(ws))

        async for update in receiver(ws):
            transcription_queue.put_nowait(update)

        tasks = [microphone_task, sender_task, keep_alive_task]
        for task in tasks:
            task.cancel()
