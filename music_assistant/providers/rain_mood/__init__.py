"""Rainy Mood Plugin Provider for Music Assistant.

Mixes looping rain sounds from rainymood.com transparently into whatever the
player is already playing from its queue.  The queue is never touched: the
rain audio is injected into the PCM stream that the streams controller serves
to the player, by implementing the AUDIO_OVERLAY plugin interface.

A persistent FFmpeg subprocess is kept alive per player so that rain audio
continues seamlessly across track transitions and seeks.  The subprocess is
only restarted when explicitly stopped and re-enabled, or when the PCM format
changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from music_assistant_models.config_entries import ConfigEntry, ConfigValueType, ProviderConfig
from music_assistant_models.enums import ConfigEntryType, EventType, PlaybackState, ProviderFeature

from music_assistant.models.plugin import PluginProvider

if TYPE_CHECKING:
    from music_assistant_models.event import MassEvent
    from music_assistant_models.media_items.audio_format import AudioFormat
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

RAIN_URL = "https://media.rainymood.com/0.mp3"

CONF_RAIN_RATIO = "rain_ratio"

SUPPORTED_FEATURES: set[ProviderFeature] = {ProviderFeature.AUDIO_OVERLAY}


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
    """Persistent FFmpeg subprocess that streams looping rain PCM."""

    def __init__(self) -> None:
        """Initialize the RainBuffer."""
        self._proc: asyncio.subprocess.Process | None = None
        self._pcm_format: AudioFormat | None = None

    async def start(self, pcm_format: AudioFormat) -> None:
        """Start the FFmpeg rain process configured to output pcm_format."""
        await self.stop()
        self._pcm_format = pcm_format
        fmt = pcm_format.content_type.value
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
            fmt,
            "-ar",
            str(pcm_format.sample_rate),
            "-ac",
            str(pcm_format.channels),
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

    async def ensure_format(self, pcm_format: AudioFormat) -> None:
        """Restart the subprocess if the requested format differs from the current one."""
        if self._pcm_format != pcm_format:
            await self.start(pcm_format)

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

    async def scaled_stream(
        self, rain_vol: float, pcm_format: AudioFormat
    ) -> AsyncGenerator[bytes, None]:
        """
        Yield rain PCM chunks scaled by rain_vol.

        :param rain_vol: Volume multiplier (1.0 = same level as music).
        :param pcm_format: PCM format for dtype selection.
        """
        fmt = pcm_format.content_type.value
        dtype: Any = np.float32 if "f32" in fmt else np.int16
        clip_min: float = -1.0 if dtype == np.float32 else -32768
        clip_max: float = 1.0 if dtype == np.float32 else 32767
        chunk_size = 4096
        while True:
            data = await self.read(chunk_size)
            if data is None:
                return
            arr = np.frombuffer(data, dtype=dtype)
            scaled = np.clip(arr * rain_vol, clip_min, clip_max).astype(dtype)
            yield scaled.tobytes()


class RainyMoodPlugin(PluginProvider):
    """Rainy Mood Plugin Provider.

    When enabled for a player, implements the AUDIO_OVERLAY interface so the
    streams controller can mix rain into the regular queue PCM output.
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
    # AUDIO_OVERLAY interface (called by the streams controller)
    # ------------------------------------------------------------------

    def is_overlay_active(self, player_id: str) -> bool:
        """
        Return whether the rain overlay is currently active for this player.

        :param player_id: The player to check.
        """
        return player_id in self._active_players

    async def get_overlay_stream(
        self,
        player_id: str,
        pcm_format: AudioFormat,
    ) -> AsyncGenerator[bytes, None] | None:
        """
        Return a volume-adjusted PCM rain stream matching pcm_format.

        :param player_id: The player for which the overlay is requested.
        :param pcm_format: The PCM format the overlay must be produced in.
        :returns: Async generator of raw PCM bytes, or None if overlay is not active.
        """
        if player_id not in self._active_players:
            return None
        buf = self._rain_buffers.get(player_id)
        if buf is None:
            return None
        await buf.ensure_format(pcm_format)
        rain_vol = float(cast("int", self.config.get_value(CONF_RAIN_RATIO))) / 100.0
        return buf.scaled_stream(rain_vol, pcm_format)

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
        self._rain_buffers[player_id] = RainBuffer()
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
