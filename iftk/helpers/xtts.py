# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import sys
from asyncio import to_thread
from typing import AsyncIterator

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from examples.iter_utils import iter_on_thread


async def xtts_stream(
    message: str, speaker: str = "Luis Moray", language: str = "en"
) -> AsyncIterator[torch.Tensor]:
    config = XttsConfig()
    model_path = os.path.join(
        get_user_data_dir("tts"), "tts_models--multilingual--multi-dataset--xtts_v2"
    )
    config.load_json(os.path.join(model_path, "config.json"))
    xtts_model = Xtts.init_from_config(config)
    xtts_model.load_checkpoint(config, checkpoint_dir=model_path)
    xtts_model.cuda()
    gpt_cond_latent, speaker_embedding = xtts_model.speaker_manager.speakers[
        speaker
    ].values()
    async for chunk in iter_on_thread(
        xtts_model.inference_stream(
            message,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            language=language,
        )
    ):
        yield chunk
