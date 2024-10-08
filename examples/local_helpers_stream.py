# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from asyncio import to_thread
from typing import AsyncIterator

import numpy as np
import sounddevice as sd
import torch
from context import iftk

from iftk.helpers import pyaudio, silero_vad, transformers, whisper, xtts

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK = 512 * 100
RATE = 16000
messages = [{"role": "system", "content": "Respond in a few sentences."}]


async def main():
    microphone_stream: AsyncIterator = pyaudio.microphone(
        rate=RATE, frames_per_buffer=CHUNK
    )
    vad_stream: AsyncIterator = silero_vad.silero_vad_stream(
        microphone_stream, sample_rate=RATE
    )
    whisper_stream: AsyncIterator = whisper.whisper_stream(microphone_stream)
    turn_transcription_parts = []
    async for output in whisper_stream:
        turn_transcription_parts.append(output)
        if not anext(
            vad_stream
        ):  # Detected silence in microphone chunk (around 2 seconds of silence) - ends user turn
            turn_transcription = " ".join(turn_transcription_parts)
            messages.append({"role": "user", "content": turn_transcription})
            transformer_stream: AsyncIterator = transformers.transformer_stream(
                model_id=MODEL_ID, messages=messages
            )
            assistant_turn_parts = []
            async for token in transformer_stream:
                if "." in token:  # Ended sentence, do XTTS chunk generation
                    xtts_stream: AsyncIterator = xtts.xtts_stream(
                        " ".join(assistant_turn_parts)
                    )
                    wav_chunks = []
                    async for chunk in xtts_stream:
                        wav_chunks.append(chunk)
                    wav = torch.cat(wav_chunks, dim=0)
                    wav = wav.squeeze().unsqueeze(0).cpu().numpy()
                    wav /= np.max(
                        np.abs(wav), axis=0
                    )  # Scale Tensor to -1 to 1 to be played by sounddevice
                    await to_thread(sd.play, data=wav, blocking=True)
                assistant_turn_parts.append(token)
            messages.append(
                {"role": "assistant", "content": " ".join(assistant_turn_parts)}
            )
            assistant_turn_parts = []
            turn_transcription_parts = []


asyncio.run(main())
