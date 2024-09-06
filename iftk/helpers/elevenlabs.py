# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, AsyncIterator, Callable, Iterator, Optional

from elevenlabs.client import ElevenLabs

from iftk.channel import AsyncChannel
from iftk.system import System


async def eleven_stream(
    sentences: AsyncIterator[str],
    eleven_client: ElevenLabs,
    voice: Optional[str] = "Jessica",
) -> AsyncIterator[Iterator[bytes]]:
    """An AsyncIterator wrapper for the 11Labs TTS generation stream.

    Args:
        sentences (AsyncIterator): A sentence AsyncIterator
        eleven_client (ElevenLabs): The 11Labs third-party TTS client.
        messages (dict, optional): The message history of the dialogue system.

    Yields:
        Iterator[bytes]: A bytes iterator for playing audio using 11Labs stream audio playing function.
    """
    async for sentence in sentences:
        if sentence:
            audio_stream = eleven_client.generate(
                text=sentence, stream=True, voice=voice
            )
            yield audio_stream


class ElevenlabsChannel(AsyncChannel):
    def __init__(
        self,
        sentence_stream: AsyncChannel,
        eleven_client: ElevenLabs,
        voice: Optional[str] = "Jessica",
        notify_readable: Callable[[None], None] = None,
    ) -> None:
        super().__init__(notify_readable)
        self.sentence_stream = sentence_stream
        self.eleven_stream = eleven_stream(
            sentences=self.write(), eleven_client=eleven_client, voice=voice
        )

    async def read(self) -> AsyncIterator[bytes]:
        yield await anext(self.eleven_stream)

    async def write(self) -> AsyncIterator[str]:
        async for sentence in self.sentence_stream:
            yield sentence


class ElevenlabsSystem(System):
    async def create_async_channel(
        self,
        sentence_stream: AsyncChannel,
        eleven_client: ElevenLabs,
        notify_readable: Callable[[None], None] = None,
        voice: Optional[str] = "Jessica",
        **kwargs,
    ) -> AsyncChannel:
        eleven_channel = ElevenlabsChannel(
            sentence_stream=sentence_stream,
            eleven_client=eleven_client,
            voice=voice,
            notify_readable=notify_readable,
        )
        return eleven_channel
