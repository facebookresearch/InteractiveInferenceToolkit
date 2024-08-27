# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import unittest

from .test_deepgram_stream import stream_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from iftk.helpers.silero_vad import silero_vad_stream

AUDIO_FILE = "tests/helpers/audio_test.wav"
CHUNK = 512 * 8
RATE = 16000


class TestSileroVAD(unittest.IsolatedAsyncioTestCase):
    async def test_vad(self):
        async for timestamp in silero_vad_stream(stream_file(AUDIO_FILE, chunk=CHUNK)):
            self.assertIsInstance(timestamp, list)


if __name__ == "__main__":
    unittest.main()
