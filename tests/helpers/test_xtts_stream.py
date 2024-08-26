# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from iftk.helpers.xtts import xtts_stream


class TestXTTS(unittest.IsolatedAsyncioTestCase):
    async def test_xtts_chunks(self):
        sample_message = "Hello there!"
        async for chunk in xtts_stream(message=sample_message):
            assert type(chunk) == torch.Tensor


if __name__ == "__main__":
    unittest.main()
