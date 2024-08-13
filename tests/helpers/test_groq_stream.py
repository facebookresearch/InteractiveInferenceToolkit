# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import unittest
from groq import AsyncGroq
from dotenv import dotenv_values

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from iftk.helpers.groq_stream import GroqProcessor

DOTENV = dotenv_values(".env")
GROQ_API_KEY = DOTENV["GROQ_API_KEY"]
messages = [{"role": "user", "content": "How are you doing?"}]
model_id = "llama-3.1-70b-versatile"
client = AsyncGroq()


class TestGroq(unittest.IsolatedAsyncioTestCase):
    async def test_sentences(self):
        llm_stream = await client.chat.completions.create(
            messages=messages, model=model_id, stream=True
        )
        async for sentence in GroqProcessor(llm_stream):
            if sentence:
                assert type(sentence) == str
                assert (
                    sentence[-1] == "." or sentence[-1] == "?"
                )  # sentences must end on period or question mark


if __name__ == "__main__":
    unittest.main()
