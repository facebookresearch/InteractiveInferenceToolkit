# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
from typing import Any, AsyncIterator, Callable

import websockets

from iftk.channel import AsyncChannel
from iftk.system import System

WSS_URL = "wss://api.deepgram.com/v1/listen?endpointing=500&encoding=linear16&sample_rate=16000&channels=1&interim_results=false"


async def deepgram_stream(
    key: str, audio_stream: AsyncIterator[bytes]
) -> AsyncIterator[str]:
    """An AsyncIterator-friendly Deepgram wrapper for streaming inputs/outputs.

    Args:
        key (str): Your Deepgram API key
        audio_stream (AsyncIterator[bytes]): An AsyncIterator-compatible audio iterator, commonly in the form of streaming file or audio input.

    Yields:
        str: A sentence of text.
    """
    extra_headers = {"Authorization": f"Token {key}"}

    async with websockets.connect(
        WSS_URL,
        extra_headers=extra_headers,
    ) as ws:

        async def keep_alive(websocket: websockets.WebSocketClientProtocol):
            while True:
                keep_alive_msg = json.dumps({"type": "KeepAlive"})
                await websocket.send(keep_alive_msg)
                await asyncio.sleep(3)

        keep_alive_task = asyncio.create_task(keep_alive(websocket=ws))

        async def sender(ws):
            while True:
                data = await anext(audio_stream)
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

        sender_task = asyncio.create_task(sender(ws))

        async for update in receiver(ws):
            if update:
                yield update

        tasks = [sender_task, keep_alive_task]
        for task in tasks:
            task.cancel()


class DeepgramChannel(AsyncChannel):
    def __init__(
        self,
        deepgram_key: str,
        input_stream: AsyncIterator[bytes],
        notify_readable: Callable[[None], None] = None,
    ) -> None:
        super().__init__(notify_readable)
        self.deepgram_stream = deepgram_stream(deepgram_key, input_stream)

    async def read(self) -> AsyncIterator[str]:
        yield await anext(self.deepgram_stream)


class DeepgramSystem(System):
    async def create_async_channel(
        self,
        deepgram_key: str,
        input_stream: AsyncIterator[bytes],
        notify_readable: Callable[[None], None],
        **kwargs,
    ) -> AsyncChannel:
        deepgram_channel = DeepgramChannel(
            deepgram_key=deepgram_key,
            input_stream=input_stream,
            notify_readable=notify_readable,
        )
        return deepgram_channel
