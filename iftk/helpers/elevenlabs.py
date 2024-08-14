# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import AsyncIterator, Iterator, Optional

from elevenlabs.client import ElevenLabs


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
        Iterator[AsyncIterator]: A bytes iterator for playing audio using 11Labs stream audio playing function.
    """
    async for sentence in sentences:
        if sentence:
            audio_stream = eleven_client.generate(
                text=sentence, stream=True, voice=voice
            )
            yield audio_stream
