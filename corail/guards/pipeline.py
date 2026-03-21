"""GuardPipeline — runs all guards in sequence on input/output."""

import logging

from corail.events.bus import EventBus
from corail.events.types import Event, EventType
from corail.guards.base import Guard, GuardDirection, GuardResult

logger = logging.getLogger(__name__)


class GuardPipeline:
    """Runs registered guards in order. Emits events for each check.

    If any guard blocks, the pipeline stops and returns the block result.
    If a guard returns sanitized content, subsequent guards check the sanitized version.
    """

    def __init__(self, guards: list[Guard] | None = None, event_bus: EventBus | None = None) -> None:
        self._guards = guards or []
        self._event_bus = event_bus

    def add(self, guard: Guard) -> None:
        self._guards.append(guard)

    @property
    def guard_names(self) -> list[str]:
        return [g.name for g in self._guards]

    async def check_input(self, content: str, user_id: str = "", session_id: str = "") -> GuardResult:
        return await self._run(content, GuardDirection.INPUT, user_id, session_id)

    async def check_output(self, content: str, user_id: str = "", session_id: str = "") -> GuardResult:
        return await self._run(content, GuardDirection.OUTPUT, user_id, session_id)

    async def _run(self, content: str, direction: GuardDirection, user_id: str, session_id: str) -> GuardResult:
        current = content

        for guard in self._guards:
            # Skip guards that don't apply to this direction
            if guard.direction != GuardDirection.BOTH and guard.direction != direction:
                continue

            result = await guard.check(current, direction)
            result.guard_name = guard.name

            # Emit event
            event_type = (
                EventType.GUARD_BLOCKED
                if not result.allowed
                else (
                    EventType.GUARD_INPUT_CHECKED
                    if direction == GuardDirection.INPUT
                    else EventType.GUARD_OUTPUT_CHECKED
                )
            )
            if self._event_bus:
                await self._event_bus.emit(
                    Event(
                        type=event_type,
                        user_id=user_id,
                        session_id=session_id,
                        data={
                            "guard": guard.name,
                            "direction": direction.value,
                            "allowed": result.allowed,
                            "reason": result.reason,
                        },
                    )
                )

            if not result.allowed:
                logger.warning("Guard %s BLOCKED (%s): %s", guard.name, direction.value, result.reason)
                return result

            # Use sanitized content for next guard
            if result.sanitized:
                current = result.sanitized

        return GuardResult(allowed=True, sanitized=current if current != content else "")
