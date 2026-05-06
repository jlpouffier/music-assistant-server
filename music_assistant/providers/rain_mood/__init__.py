"""Rainy Mood Plugin Provider for Music Assistant.

Mixes looping rain sounds from rainymood.com transparently into whatever the
player is already playing from its queue.  The queue is never touched: the
rain audio is injected into the PCM stream that the streams controller serves
to the player, by overriding get_player_overlay() on the PluginProvider base.

A persistent FFmpeg subprocess is kept alive per player so that rain audio
continues seamlessly across track transitions and seeks.  The subprocess is
only restarted when explicitly stopped and re-enabled.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.config_entries import ConfigEntry, ConfigValueType, ProviderConfig
from music_assistant_models.enums import ConfigEntryType, EventType, PlaybackState, ProviderFeature

from music_assistant.models.plugin import PluginProvider

if TYPE_CHECKING:
    from music_assistant_models.event import MassEvent
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

RAIN_URL = "https://media.rainymood.com/0.mp3"

CONF_RAIN_RATIO = "rain_ratio"

SUPPORTED_FEATURES: set[ProviderFeature] = set()


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return RainyMoodPlugin(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """
    Return Config entries to setup this provider.

    :param mass: MusicAssistant instance.
    :param instance_id: ID of an existing provider instance (None if new instance setup).
    :param action: Optional action key called from config entries UI.
    :param values: The (intermediate) raw values for config entries sent with the action.
    """
    return (
        ConfigEntry(
            key=CONF_RAIN_RATIO,
            type=ConfigEntryType.INTEGER,
            range=(0, 200),
            default_value=100,
            label="Rain Volume Ratio (%)",
            description="Rain loudness relative to the music. 100 % = equally loud, 0 % = inaudible, 200 % = twice as loud.",
        ),
    )


class RainBuffer:
    """Persistent FFmpeg subprocess that streams looping rain PCM (f32le/48000/2ch)."""

    def __init__(self) -> None:
        """Initialize the RainBuffer."""
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Start the FFmpeg rain process."""
        await self.stop()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-stream_loop",
            "-1",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_on_http_error",
            "5xx,429",
            "-reconnect_delay_max",
            "10",
            "-i",
            RAIN_URL,
            "-f",
            "f32le",
            "-ar",
            "48000",
            "-ac",
            "2",
            "pipe:1",
        ]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def stop(self) -> None:
        """Kill the FFmpeg process."""
        proc, self._proc = self._proc, None
        if proc is not None:
            with suppress(Exception):
                proc.kill()

    async def read(self, n: int) -> bytes | None:
        """
        Read exactly n bytes of rain PCM.

        :param n: Number of bytes to read.
        :returns: Raw PCM bytes, or None if the process has ended.
        """
        if self._proc is None or self._proc.stdout is None or self._proc.returncode is not None:
            return None
        try:
            return await self._proc.stdout.readexactly(n)
        except (asyncio.IncompleteReadError, Exception):
            return None


class RainyMoodPlugin(PluginProvider):
    """Rainy Mood Plugin Provider.

    When enabled for a player, returns an overlay via get_player_overlay() so
    the streams controller can mix rain into the regular queue PCM output.
    A persistent RainBuffer ensures rain audio continues seamlessly across
    track transitions and seeks without restarting from the beginning.
    """

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set[ProviderFeature],
    ) -> None:
        """Initialize the Rainy Mood plugin."""
        super().__init__(mass, manifest, config, supported_features)
        self._active_players: set[str] = set()
        self._rain_buffers: dict[str, RainBuffer] = {}
        self._unregister_handles: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Provider lifecycle
    # ------------------------------------------------------------------

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        self._unregister_handles.append(
            self.mass.register_api_command("rain_mood/enable", self._enable)
        )
        self._unregister_handles.append(
            self.mass.register_api_command("rain_mood/disable", self._disable)
        )
        self._unregister_handles.append(
            self.mass.register_api_command("rain_mood/status", self._status)
        )
        self._unregister_handles.append(
            self.mass.subscribe(self._on_queue_updated, EventType.QUEUE_UPDATED)
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        for handle in self._unregister_handles:
            handle()
        self._unregister_handles.clear()
        for buf in self._rain_buffers.values():
            await buf.stop()
        self._rain_buffers.clear()
        self._active_players.clear()

    # ------------------------------------------------------------------
    # Overlay interface (called by the streams controller)
    # ------------------------------------------------------------------

    def get_player_overlay(self, player_id: str) -> tuple[Any, float] | None:
        """
        Return the rain overlay reader for this player if active.

        :param player_id: The player for which an overlay is requested.
        :returns: (read_callable, rain_volume_0_to_1) if rain is enabled, else None.
        """
        if player_id not in self._active_players:
            return None
        buf = self._rain_buffers.get(player_id)
        if buf is None:
            return None
        rain_vol = float(cast("int", self.config.get_value(CONF_RAIN_RATIO))) / 100.0
        return (buf.read, rain_vol)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_queue_updated(self, event: MassEvent) -> None:
        """Auto-disable when the queue finishes playing."""
        queue = event.data
        if queue.queue_id not in self._active_players:
            return
        if queue.items == 0 and queue.state == PlaybackState.IDLE:
            self._active_players.discard(queue.queue_id)
            await self._stop_rain_buffer(queue.queue_id)
            self.logger.info("Rainy Mood auto-disabled for player %s (queue ended)", queue.queue_id)

    # ------------------------------------------------------------------
    # API commands (called from the frontend)
    # ------------------------------------------------------------------

    async def _enable(self, player_id: str) -> dict[str, Any]:
        """
        Enable rain overlay for the given player.

        :param player_id: The player on which to activate rain.
        :returns: Status dict.
        """
        if player_id in self._active_players:
            return {"active": True}
        self._active_players.add(player_id)
        await self._start_rain_buffer(player_id)
        self.logger.info("Rainy Mood enabled for player %s", player_id)
        queue = self.mass.player_queues.get(player_id)
        if queue and queue.state == PlaybackState.PLAYING:
            await self._restart_stream(player_id)
        return {"active": True}

    async def _disable(self, player_id: str) -> dict[str, Any]:
        """
        Disable rain overlay for the given player.

        :param player_id: The player on which to deactivate rain.
        :returns: Status dict.
        """
        if player_id not in self._active_players:
            return {"active": False}
        self._active_players.discard(player_id)
        await self._stop_rain_buffer(player_id)
        self.logger.info("Rainy Mood disabled for player %s", player_id)
        queue = self.mass.player_queues.get(player_id)
        if queue and queue.state == PlaybackState.PLAYING:
            await self._restart_stream(player_id)
        return {"active": False}

    async def _start_rain_buffer(self, player_id: str) -> None:
        buf = RainBuffer()
        await buf.start()
        self._rain_buffers[player_id] = buf

    async def _stop_rain_buffer(self, player_id: str) -> None:
        buf = self._rain_buffers.pop(player_id, None)
        if buf:
            await buf.stop()

    async def _restart_stream(self, player_id: str) -> None:
        """Resume playback from the current position to force a stream re-request."""
        try:
            await self.mass.player_queues.resume(player_id)
        except Exception as err:
            self.logger.debug("Could not restart stream for %s: %s", player_id, err)

    async def _status(self, player_id: str) -> dict[str, Any]:
        """
        Return whether rain is currently active for the given player.

        :param player_id: The player to query.
        :returns: Status dict with 'active' bool.
        """
        return {"active": player_id in self._active_players}
