"""Thin registry of named Channel instances."""

from src.bus.channel import Channel


class MessageBus:
    """Registry that holds named Channel instances.

    Components declare which channels they read from and write to by name.
    The bus itself does no routing — it only stores and hands out Channel
    references. Each Channel owns its own buffers, listeners, and dirty sets,
    so publishes on one channel never wake listeners on another.

    Channels must be created at startup via create_channel before any component
    tries to access them.
    """

    def __init__(self) -> None:
        """Initialise with an empty channel registry."""
        self._channels: dict[str, Channel] = {}

    def create_channel(self, name: str, capacity: int) -> Channel:
        """Create and register a new channel.

        Args:
            name: Unique name for this channel (e.g. "market_data").
            capacity: Capacity of each per-symbol RingBuffer on this channel.

        Returns:
            The newly created Channel.

        Raises:
            ValueError: If a channel with this name already exists.
        """
        if name in self._channels:
            raise ValueError(f"Channel '{name}' already exists.")
        ch = Channel(capacity)
        self._channels[name] = ch
        return ch

    def channel(self, name: str) -> Channel:
        """Return the Channel registered under name.

        Args:
            name: The channel name passed to create_channel.

        Returns:
            The Channel instance.

        Raises:
            KeyError: If no channel with this name has been created.
        """
        if name not in self._channels:
            raise KeyError("This channel has not been created")
        
        return self._channels[name]

    @property
    def channel_names(self) -> set[str]:
        """Return the set of registered channel names."""
        return set(self._channels)
