# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
from typing import AsyncIterator

import websockets

WSS_URL = "wss://api.deepgram.com/v1/listen?endpointing=500&encoding=linear16&sample_rate=16000&channels=1&interim_results=false"


async def deepgram_stream(
    key: str, audio_stream: AsyncIterator[bytes]
) -> AsyncIterator[str]:
    """An AsyncIterator-friendly Deepgram wrapper for streaming inputs/outputs.

    Args:
        key (str): Your Deepgram API key
        audio_stream (AsyncIterator[bytes]): An AsyncIterator-compatible audio iterator, commonly in the form of streaming file or audio input.
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
            yield update

        tasks = [sender_task, keep_alive_task]
        for task in tasks:
            task.cancel()
