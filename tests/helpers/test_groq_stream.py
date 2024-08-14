# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import unittest

from dotenv import dotenv_values
from groq import AsyncGroq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from iftk.helpers.groq import groq_sentence_stream

DOTENV = dotenv_values(".env")
GROQ_API_KEY = DOTENV["GROQ_API_KEY"]
messages = [{"role": "user", "content": "How are you doing?"}]
model_id = "llama-3.1-70b-versatile"


class TestGroq(unittest.IsolatedAsyncioTestCase):
    async def test_sentences(self):
        client = AsyncGroq()
        llm_stream = await client.chat.completions.create(
            messages=messages, model=model_id, stream=True
        )
        async for sentence in groq_sentence_stream(llm_stream):
            assert sentence
            assert type(sentence) == str
            assert (
                sentence[-1] == "." or sentence[-1] == "?"
            )  # sentences must end on period or question mark


if __name__ == "__main__":
    unittest.main()
