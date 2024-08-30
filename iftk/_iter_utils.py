# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import TypeVar

T = TypeVar("T")


_sentinel = object()


async def iter_on_thread(iterator: Iterator[T]) -> AsyncIterator[T]:
    """Request the next iterator value within asyncio.to_thread."""
    read = lambda: next(iterator, _sentinel)
    while True:
        v = await asyncio.to_thread(read)
        if v is _sentinel:
            break
        yield v


async def queue_to_iter(queue: asyncio.Queue[T]) -> AsyncIterator[T]:
    while True:
        yield await queue.get()


async def iter_to_queue(iterator: AsyncIterator[T], queue: asyncio.Queue[T]) -> None:
    async for x in iterator:
        await queue.put(x)
