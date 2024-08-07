# InteractiveInferenceToolkit, a library for building interactive inference systems

:exclamation: This library is currently in the pre-alpha stage and is not yet
published. All interfaces and examples are subject to change.

The InteractiveInferenceToolkit (IFTK, iftk) encapsulates code and patterns to
assist in building flexible, interactive systems. Its genesis was in speech-
and video-based demonstrations of machine learning models.

# Key Concepts

At a high-level, an interactive inference-based system can be thought of as a
communication channel and a protocol for communicating over that channel.

The `Channel` is the limited but powerful code interface at the heart of IFTK.
A channel may be written to at any time, and may be read from at any time,
although there is no guarantee of a result. On creation, a channel accepts a
notification callback that will be called when the channel is ready to be read
(note that there is still no guarantee of a result when the channel is read
from).

The user of a channel must define the protocol for communication -- `Channel`
will accept any type of object to read or write. This can be as simple as a
convention of primitive types, or more flexible and complicated such as a type
hierarchy built from Python's `TypedDict` or pydantic's `BaseModel`.

Moving into the implementation behind the channel, IFTK defines interfaces to
separate model loading from channel creation, and to coordinate activity
between different stages of computation.

The `System` interface helps to organize model loading, which often needs to
happen early on, from channel creation, which can happen once, e.g., in the
case of a CLI, or repeatedly throughout the lifetime of a program, e.g., in the
case of an interactive service backend. Dynamic configuration may be passed to
the channel on creation.

To coordinate computation, the `PubSub` and `Subscriber` interfaces define an
event bus where any subscriber may listen to or publish events. This usually
requires another protocol which is implicitly defined by what each subscriber
listens to and publishes. See `PubSubChannel` for an example of how to fit a
`PubSub` instance to the `Channel` interface.

A useful alternative to `PubSub` is the concept of a `Stream`, which in
practice is often defined as a function that accepts Python's `AsyncIterator`
and is itself an `AsyncIterator`.

Finally, at the lowest level of an interactive inference-based system is the
inference. This is often either locally-run models or API calls for remote
processing. If there is not already an API wrapper defined, it is common to
define an API for inference that enables convenient re-use within a
`Subscriber` or `Stream`-type function which will be concerned with state
management and lifecycle within the overall system.

# Code Organization

* iftk/ - the core library code, this is what would be packaged for distribution on PyPI.
* tests/ - unit test code to validate the core library code.
* examples/ - sample usage of the iftk library.

# Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

# License

InteractiveInferenceToolkit is MIT licensed, as found in the LICENSE file.
