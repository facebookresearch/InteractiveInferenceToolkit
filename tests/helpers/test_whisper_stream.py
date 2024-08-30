# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import unittest

from .test_deepgram_stream import stream_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from iftk.helpers.whisper import whisper_stream

AUDIO_FILE = "tests/helpers/audio_test.wav"
CHUNK = 512 * 100
RATE = 16000


class TestWhisper(unittest.IsolatedAsyncioTestCase):
    async def test_whisper(self):
        async for output in whisper_stream(stream_file(AUDIO_FILE, CHUNK)):
            self.assertIsInstance(output, str)
            self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
