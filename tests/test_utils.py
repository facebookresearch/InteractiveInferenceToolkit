# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import sys
from typing import Any, TextIO

from .context import iftk

logger = logging.getLogger(__name__)


class QueueSubscriber(iftk.Subscriber):
    """Subscriber that collects all events into the `queue` property"""

    def __init__(self, pubsub: iftk.PubSub) -> None:
        super().__init__(pubsub)
        self.queue = asyncio.Queue()

    async def on_event(self, evt: Any) -> None:
        await self.queue.put(evt)


class PrintSubscriber(iftk.Subscriber):
    """Subscriber that prints all events to `file`."""

    def __init__(
        self,
        pubsub: iftk.PubSub,
        skip: tuple[Any] | None = None,
        file: TextIO = sys.stderr,
        maxlen: int = 200,
    ) -> None:
        super().__init__(pubsub)
        self.skip = tuple(skip)
        self.file = file
        self.maxlen = maxlen

    async def on_event(self, event: Any) -> None:
        if self.skip and isinstance(event, self.skip):
            return

        s = str(event)
        # Truncate long events (e.g., containing audio data)
        if len(s) > self.maxlen:
            halfmax = self.maxlen // 2
            s = f"{s[:halfmax]}...{s[-halfmax:]}"
        print(s, file=self.file)
