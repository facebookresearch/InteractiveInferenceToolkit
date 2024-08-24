# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
from typing import AsyncIterator

import numpy as np
from scipy.io import wavfile
from asyncio import to_thread
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad


async def silero_vad_stream(audio_feed: AsyncIterator[bytes], sample_rate: int = 16000) -> AsyncIterator[bool]:
    model = load_silero_vad()
    async for chunk in audio_feed:
        wav_bytes_io = io.BytesIO()
        raw_data = np.frombuffer(buffer=chunk, dtype=np.int16)
        wavfile.write(wav_bytes_io, sample_rate, raw_data)
        data, sr = torchaudio.load(wav_bytes_io)
        transform = torchaudio.transforms.Resample(sr, sample_rate)
        data = transform(data)
        data = data.squeeze(0)
        coro = to_thread(get_speech_timestamps, data, model)
        yield await coro
        model.reset_states()