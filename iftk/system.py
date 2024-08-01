# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from .channel import AsyncChannel


class System:
    def load(self):
        pass

    async def create_async_channel(
        self,
        notify_readable: Callable[[None], None] = None,
        **kwargs,
    ) -> AsyncChannel:
        raise NotImplementedError()
