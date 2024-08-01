# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from typing import Any, Callable


class ChannelClosedEvent(RuntimeError):
    pass


class Channel:
    """Channel models a (possibly asymmetric) two-way stream of typed events.

    The interface is intentionally simple:

    - Call write() to append an input
    - Call read() to pop an output (may return None)
    - Call close() to indicate no further writes. It is an error to write() afterwards.
    - Optionally, provide notify_readable to be notified when read() might be available
    """

    def __init__(self, notify_readable: Callable[[None], None] = None) -> None:
        self.notify_readable = notify_readable
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def read(self) -> Any | None:
        pass

    def write(self, x: Any) -> None:
        if self._closed:
            raise ChannelClosedEvent()


class AsyncChannel(Channel):
    """A Channel with async close(), read(), and write() methods."""

    def __init__(self, notify_readable: Callable[[None], None] = None) -> None:
        super().__init__(notify_readable)

    async def close(self) -> None:
        super().close()

    async def read(self) -> Any | None:
        pass

    async def write(self, x: Any) -> None:
        super().write(x)


class DequeChannel(Channel):
    """A simple symmetric channel that wraps a deque, writing to the right and reading from the left."""

    def __init__(self, notify_readable: Callable[[None], None] = None) -> None:
        super().__init__(notify_readable)
        self.q = deque()

    def read(self) -> Any | None:
        try:
            return self.q.popleft()
        except IndexError:
            return None

    def write(self, x: Any) -> None:
        super().write(x)
        self.q.append(x)
        if self.notify_readable:
            self.notify_readable()
