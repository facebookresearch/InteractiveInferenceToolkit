# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from asyncio import to_thread
from typing import AsyncIterator

import numpy as np
import whisper


async def whisper_stream(
    audio_feed: AsyncIterator[bytes], model_size: str = "tiny", language: str = "en"
) -> AsyncIterator[str]:
    """A stream-friendly implementation of Whisper for ASR transcript generation.

    Args:
        audio_feed (AsyncIterator[bytes]): Audio bytes AsyncIterator to generate the transcription on. Note that the chunk size
                                           should be big enough to generate streaming transcriptions.
        model_size (str, optional): The model size tag. Defaults to "tiny".
        language (str, optional): The model language to generate the transcription on. Defaults to "en".

    Yields:
        AsyncIterator[str]: The text result of the transcription.
    """
    model = await to_thread(whisper.load_model, name=model_size)
    async for chunk in audio_feed:
        buffer = (
            np.frombuffer(buffer=chunk, dtype=np.int16).flatten().astype(np.float32)
            / 32768.0
        )
        if not buffer.any():
            break
        result = await to_thread(model.transcribe, buffer, language=language)
        if result["text"]:
            yield result["text"]
