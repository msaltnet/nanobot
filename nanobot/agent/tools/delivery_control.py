"""Versioned, per-turn delivery ownership for trusted exclusive tools.

Ownership only suppresses generic delivery; it is never a delivery acknowledgment.
A dedicated sender must persist its own external acknowledgments. Call claim before
any external work, and treat DeliveryClaimError as a fail-closed result.
"""
from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4

from nanobot.agent.tools.context import RequestContext

DELIVERY_CONTROL_API_VERSION = 1
DELIVERY_OWNED_META = "_delivery_owned"


class DeliveryClaimError(RuntimeError):
    """The current execution cannot safely acquire delivery ownership."""


@dataclass(frozen=True, slots=True)
class DeliveryOwnership:
    """Typed terminal outcome, not evidence that any message was delivered."""

    owner_tool: str
    channel: str
    chat_id: str
    thread_id: str | None
    turn_id: str


@dataclass(slots=True)
class _ToolDispatch:
    name: str
    exclusive: bool
    claim_allowed: bool
    active: bool = True


_DISPATCH: ContextVar[_ToolDispatch | None] = ContextVar("delivery_tool_dispatch", default=None)
_CURRENT: ContextVar[TurnDeliveryControl | None] = ContextVar("turn_delivery_control", default=None)


class TurnDeliveryControl:
    """Shared mutable arbitration state inherited by a turn's child tasks.

    Methods are synchronous so checking and claiming are atomic on the event loop.
    Only the runtime creates a scope. Retaining its handle cannot extend its life.
    """

    api_version = DELIVERY_CONTROL_API_VERSION

    def __init__(
        self, request: RequestContext,
        on_claim: Callable[[DeliveryOwnership], None] | None = None,
    ) -> None:
        self._on_claim = on_claim
        self._channel = request.channel
        self._chat_id = request.chat_id
        thread = request.metadata.get("message_thread_id", request.metadata.get("thread_id"))
        self._thread_id = str(thread) if thread is not None else None
        self._turn_id = request.turn_id or uuid4().hex
        self._ownership: DeliveryOwnership | None = None
        self._generic_started = False
        self._side_effect_dispatch_started = False
        self._accepted_injection = False
        self._active = True

    @property
    def ownership(self) -> DeliveryOwnership | None:
        return self._ownership

    def claim_current_target(self, *, owner_tool: str) -> DeliveryOwnership:
        """Claim once from an active exclusive tool, before any external work.

        A prior side-effect-capable tool, generic send attempt (including unknown
        callbacks), accepted input injection, nonexclusive dispatch or expired scope denies
        the claim. Repeated claims are rejected, including by the same tool.
        """
        dispatch = _DISPATCH.get()
        if not self._active or _CURRENT.get() is not self:
            raise DeliveryClaimError("No active delivery-control turn")
        if not dispatch or not dispatch.active or not dispatch.exclusive or dispatch.name != owner_tool:
            raise DeliveryClaimError("Delivery ownership requires the active exclusive tool")
        if self._ownership is not None:
            raise DeliveryClaimError("Turn delivery is already owned")
        if not dispatch.claim_allowed:
            raise DeliveryClaimError("A prior side-effect-capable tool already started")
        if self._generic_started:
            raise DeliveryClaimError("Generic delivery already started; its outcome may be unknown")
        if self._accepted_injection:
            raise DeliveryClaimError("Turn already accepted follow-up input")
        self._ownership = DeliveryOwnership(
            owner_tool, self._channel, self._chat_id, self._thread_id, self._turn_id,
        )
        if self._on_claim is not None:
            self._on_claim(self._ownership)
        return self._ownership

    def begin_generic_delivery(
        self, *, channel: str, chat_id: str, thread_id: str | None = None,
    ) -> bool:
        """Atomically guard/register a generic send BEFORE invoking its callback.

        False means suppressed, never acknowledged. Other targets are unaffected.
        Expired inherited tasks cannot send to an owned turn's target.
        """
        if (channel, chat_id, thread_id) != (self._channel, self._chat_id, self._thread_id):
            return True
        if self._ownership is not None:
            return False
        if not self._active:
            return True
        self._generic_started = True
        return True

    def note_tool_dispatch(self, *, read_only: bool) -> bool:
        """Runtime dispatch barrier for opaque plugins that may send directly.

        A late claim after any earlier side-effect-capable tool fails closed,
        even if that earlier tool was exclusive or returned an error/timeout.
        """
        claim_allowed = not self._side_effect_dispatch_started
        if not read_only:
            self._side_effect_dispatch_started = True
        return claim_allowed

    def note_accepted_injection(self) -> None:
        """Runtime admission: future ownership must not replay accepted input."""
        self._accepted_injection = True

    def close(self) -> None:
        """End runtime scope; retained ownership still blocks late generic sends."""
        self._active = False


def current_turn_delivery_control() -> TurnDeliveryControl | None:
    """Return this task's scope, or None; claims reject expired child scopes too."""
    return _CURRENT.get()


@contextmanager
def delivery_control_scope(
    request: RequestContext, *, on_claim: Callable[[DeliveryOwnership], None] | None = None,
) -> Generator[TurnDeliveryControl, None, None]:
    """Runtime boundary: establish a fresh scope and close all inherited handles."""
    control = TurnDeliveryControl(request, on_claim)
    token = _CURRENT.set(control)
    try:
        yield control
    finally:
        control.close()
        _CURRENT.reset(token)


@contextmanager
def delivery_tool_dispatch(
    name: str, *, exclusive: bool, read_only: bool,
) -> Generator[None, None, None]:
    """Runtime boundary: claims are valid only while this dispatch is executing."""
    control = current_turn_delivery_control()
    claim_allowed = control.note_tool_dispatch(read_only=read_only) if control is not None else False
    dispatch = _ToolDispatch(name, exclusive, claim_allowed)
    token = _DISPATCH.set(dispatch)
    try:
        yield
    finally:
        dispatch.active = False
        _DISPATCH.reset(token)
