# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import unittest
from typing import AsyncIterator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from iftk.helpers.transformers import transformer_stream

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
messages = [{"role": "user", "content": "How are you doing?"}]


class TestLLM(unittest.IsolatedAsyncioTestCase):
    async def test_llm(self):
        async for token in transformer_stream(model_id=MODEL_ID, messages=messages):
            self.assertIsInstance(token, str)
