# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import concurrent.futures
import logging
from typing import Any, Callable

from .channel import AsyncChannel, DequeChannel

logger = logging.getLogger(__name__)


class BaseSubscriber:
    publishes = None
    subscribes_to = None

    async def on_event(self, event: Any) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class PubSub:
    def __init__(self, loop: asyncio.AbstractEventLoop = None) -> None:
        self.loop = loop if loop is not None else asyncio.get_running_loop()
        self.subscribers: list[BaseSubscriber] = []

    def subscribe(self, subscriber: BaseSubscriber) -> None:
        self.subscribers.append(subscriber)

    async def publish(self, event) -> None:
        # logger.debug(f"publishing {event} to {len(self._subscribers)} subscribers")
        for subscriber in self.subscribers:
            subscribes_to = getattr(subscriber, "subscribes_to", None)
            if subscribes_to is None or isinstance(event, subscribes_to):
                await subscriber.on_event(event)

    def publish_threadsafe(self, event) -> concurrent.futures.Future:
        """Run publish() on the event loop this instance was created on. Thread-safe.

        Return a concurrent.futures.Future to cancel the publish from another OS thread.

        See also: https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading
        """
        return asyncio.run_coroutine_threadsafe(self.publish(event), self.loop)

    async def shutdown(self):
        for subscriber in self.subscribers:
            await subscriber.shutdown()


class Subscriber(BaseSubscriber):
    publishes = None
    subscribes_to = None

    def __init__(self, pubsub: PubSub):
        self.pubsub = pubsub
        pubsub.subscribe(self)

    async def publish(self, event: Any) -> None:
        if not isinstance(event, self.__class__.publishes):
            logger.warn(f"publish() unregistered event: {type(event)} in {type(self)}")
        await self.pubsub.publish(event)

    def publish_threadsafe(self, event: Any) -> concurrent.futures.Future:
        if not isinstance(event, self.__class__.publishes):
            logger.warn(
                f"publish_threadsafe() unregistered event: {type(event)} in {type(self)}"
            )
        return self.pubsub.publish_threadsafe(event)


class PubSubChannel(AsyncChannel):
    def __init__(
        self,
        pubsub: PubSub,
        notify_readable: Callable[[None], None] = None,
    ):
        super().__init__()
        self.output_channel = DequeChannel(notify_readable)
        self.pubsub: PubSub = pubsub
        pubsub.subscribe(self)

    async def close(self) -> None:
        await super().close()
        await self.pubsub.shutdown()

    async def read(self) -> Any | None:
        return self.output_channel.read()

    async def write(self, event: Any) -> None:
        await super().write(event)
        await self.pubsub.publish(event)
