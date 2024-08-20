# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import sys
import unittest
import wave
from math import ceil
from typing import AsyncIterator

from dotenv import dotenv_values

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from iftk.helpers.deepgram import deepgram_stream

DOTENV = dotenv_values(".env")
DEEPGRAM_KEY = DOTENV["DEEPGRAM_API_KEY"]
AUDIO_FILE = "tests/helpers/audio_test.wav"
CHUNK = 512


async def stream_file(filepath: str, chunk: int) -> AsyncIterator[bytes]:
    w = wave.open(filepath, "rb")
    for _ in range(ceil(w.getnframes() / CHUNK)):
        data = w.readframes(chunk)
        yield data
        await asyncio.sleep(0.0)


class TestDeepgram(unittest.IsolatedAsyncioTestCase):
    async def test_deepgram_outputs(self):
        deepgram_iterator = deepgram_stream(
            DEEPGRAM_KEY, stream_file(AUDIO_FILE, CHUNK)
        )
        result = await anext(deepgram_iterator)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
