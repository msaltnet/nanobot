"""Delivery ownership at the real AgentLoop/provider and MessageTool boundaries."""
from __future__ import annotations

import asyncio
import socket
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest


@pytest.fixture(autouse=True)
async def no_external_network(monkeypatch):
    # Patch only after pytest-asyncio has created its Windows event-loop sockets.
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not contact live services")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


class OwnerTool(Tool):
    name = "owned_delivery"
    description = "Deliver using a dedicated sender."
    parameters = {"type": "object", "properties": {}}
    exclusive = True
    # Deliberately override the normal hint: executor must enforce exclusive itself.
    concurrency_safe = True

    def __init__(self, outcome="sent", action=None):
        self.outcome = outcome
        self.action = action
        self.external_calls = 0
        self.control = None

    async def execute(self, **kwargs):
        from nanobot.agent.tools.delivery_control import current_turn_delivery_control

        self.control = current_turn_delivery_control()
        assert self.control is not None
        self.control.claim_current_target(owner_tool=self.name)
        self.external_calls += 1
        if self.action:
            await self.action()
        if self.outcome == "exception":
            raise RuntimeError("synthetic storage failure")
        if self.outcome == "cancelled":
            raise asyncio.CancelledError()
        return self.outcome


def call(name, **arguments):
    return ToolCallRequest(id=f"call-{name}", name=name, arguments=arguments)


def make_loop(tmp_path, responses, owner):
    provider = MagicMock()
    provider.get_default_model.return_value = "offline"
    provider.generation = GenerationSettings()
    requests = []

    async def model(**kwargs):
        requests.append(kwargs["messages"])
        assert responses, "unexpected finalization/model call"
        item = responses.pop(0)
        return item

    provider.chat_stream_with_retry = model
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="offline")
    loop.tools.register(owner)
    sent = []

    async def send(message):
        sent.append(message)

    loop.tools.get("message").set_send_callback(send)
    return loop, requests, sent


def inbound(text="news", **kwargs):
    return InboundMessage(channel="telegram", sender_id="user", chat_id="123", content=text,
                          metadata={"message_thread_id": 7}, **kwargs)


@pytest.mark.parametrize("outcome", ["sent", "failed", "unknown", "partial", "", "duplicate", "already_sent", "exception", "cancelled"])
async def test_owned_outcome_stops_same_batch_and_model_finalization(tmp_path, outcome):
    owner = OwnerTool(outcome)
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name), call("message", content="COPY")]),
        LLMResponse(content="COPY"),
    ], owner)
    result = await loop._process_message(inbound(), ephemeral=True)
    assert owner.external_calls == 1
    assert sent == []
    assert result is None
    assert len(requests) == 1


@pytest.mark.parametrize("callback_result", ["sent", "timeout"])
async def test_started_generic_delivery_denies_later_claim_even_on_timeout(tmp_path, callback_result):
    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call("message", content="FIRST"), call(owner.name)]),
        LLMResponse(content="done"),
    ], owner)

    async def send(message):
        sent.append(message)
        if callback_result == "timeout":
            raise TimeoutError("delivery unknown")

    loop.tools.get("message").set_send_callback(send)
    await loop._process_message(inbound(), ephemeral=True)
    assert len(sent) == 1
    assert owner.external_calls == 0


async def test_pending_input_is_processed_once_in_fresh_turn(tmp_path):
    pending = asyncio.Queue()
    followup = inbound("track this")
    owner = OwnerTool(action=lambda: pending.put(followup))
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name)]),
        LLMResponse(content="tracked"),
    ], owner)
    result = await loop._process_message(inbound(), pending_queue=pending, ephemeral=True)
    assert result is None
    assert len(requests) == 1
    assert pending.qsize() == 1
    next_result = await loop._process_message(pending.get_nowait(), ephemeral=True)
    assert next_result.content == "tracked"
    assert len(requests) == 2
    assert pending.empty()
    assert sent == []

async def test_builtin_guard_shared_with_child_and_other_target_unaffected(tmp_path):
    from nanobot.agent.tools.delivery_control import DeliveryClaimError
    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name)]),
    ], owner)
    observed = []

    async def action():
        message = loop.tools.get("message")
        observed.append(await asyncio.create_task(message.execute(content="BLOCKED")))
        await message.execute(content="OTHER", chat_id="456")
        with pytest.raises(DeliveryClaimError):
            owner.control.claim_current_target(owner_tool=owner.name)

    owner.action = action
    assert await loop._process_message(inbound(), ephemeral=True) is None
    assert [msg.content for msg in sent] == ["OTHER"]
    assert "not delivered" in observed[0]


async def test_expired_child_scope_cannot_reopen_delivery(tmp_path):
    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [LLMResponse(content=None, tool_calls=[call(owner.name)])], owner)
    release = asyncio.Event()
    children = []

    async def late_send():
        await release.wait()
        return await loop.tools.get("message").execute(content="LATE COPY")

    async def action():
        children.append(asyncio.create_task(late_send()))

    owner.action = action
    await loop._process_message(inbound(), ephemeral=True)
    release.set()
    await children[0]
    assert sent == []


@pytest.mark.parametrize("outcome", ["sent", "exception", "cancelled"])
async def test_system_background_response_suppressed(tmp_path, outcome):
    owner = OwnerTool(outcome)
    loop, requests, sent = make_loop(tmp_path, [LLMResponse(content=None, tool_calls=[call(owner.name)])], owner)
    msg = InboundMessage(channel="system", sender_id="subagent", chat_id="telegram:123", content="news")
    assert await loop._process_message(msg, ephemeral=True) is None
    assert len(requests) == 1
    assert owner.external_calls == 1
    assert sent == []


@pytest.mark.parametrize("outcome", ["sent", "failed", "unknown", "partial", "", "duplicate", "exception", "cancelled"])
async def test_bound_cron_records_owned_suppression_not_delivery_ack(tmp_path, outcome):
    from nanobot.cron.bound_runner import run_bound_cron_job
    from nanobot.cron.types import CronJob, CronPayload

    owner = OwnerTool(outcome)
    loop, requests, sent = make_loop(tmp_path, [LLMResponse(content=None, tool_calls=[call(owner.name)])], owner)
    records = []
    recorder = MagicMock()
    recorder.write_run_record.side_effect = lambda run_id, record: records.append(record)
    job = CronJob(id="news", name="news", payload=CronPayload(
        message="news", session_key="telegram:123", origin_channel="telegram",
        origin_chat_id="123", origin_metadata={"message_thread_id": 7},
    ))
    result = await run_bound_cron_job(job, agent=loop, cron=recorder)
    assert result.response == ""
    assert records[-1]["delivery"] == "owned"
    assert len(requests) == 1
    assert sent == []
    outgoing = []
    while not loop.bus.outbound.empty():
        outgoing.append(loop.bus.outbound.get_nowait())
    assert not [msg for msg in outgoing if msg.event is None and msg.content]


async def test_dispatch_republishes_followups_before_during_after_claim_once(tmp_path):
    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name)]),
        LLMResponse(content="first"), LLMResponse(content="second"), LLMResponse(content="third"),
    ], owner)
    real_model = loop.provider.chat_stream_with_retry
    first_request = True

    async def model(**kwargs):
        nonlocal first_request
        if first_request:
            first_request = False
            await loop._pending_queues["telegram:123"].put(inbound("before claim"))
        return await real_model(**kwargs)

    async def action():
        await loop._pending_queues["telegram:123"].put(inbound("during claim"))

    loop.provider.chat_stream_with_retry = model
    owner.action = action
    await loop._dispatch(inbound())
    await loop.bus.publish_inbound(inbound("after claim"))
    assert len(requests) == 1
    followups = []
    while not loop.bus.inbound.empty():
        msg = loop.bus.inbound.get_nowait()
        followups.append(msg.content)
        await loop._dispatch(msg)
    assert followups == ["before claim", "during claim", "after claim"]
    assert len(requests) == 4
    assert owner.external_calls == 1


async def test_accepted_injection_denies_late_owner_without_replaying_side_effect(tmp_path):
    class Tracking(Tool):
        name = "track"
        description = "track"
        parameters = {"type": "object", "properties": {}}
        count = 0

        async def execute(self, **kwargs):
            self.count += 1
            return "tracked"

    tracker = Tracking()
    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call("track")]),
        LLMResponse(content=None, tool_calls=[call("track"), call(owner.name)]),
        LLMResponse(content="tracked followup"),
    ], owner)
    loop.tools.register(tracker)
    pending = asyncio.Queue()
    await pending.put(inbound("track followup"))
    result = await loop._process_message(inbound(), pending_queue=pending, ephemeral=True)
    assert result.content == "tracked followup"
    assert tracker.count == 2
    assert owner.external_calls == 0
    assert pending.empty()


async def test_streamed_content_before_claim_denies_external_owner_work(tmp_path):
    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name)]), LLMResponse(content="done"),
    ], owner)
    real_model = loop.provider.chat_stream_with_retry
    streamed = False

    async def model(**kwargs):
        nonlocal streamed
        if not streamed:
            streamed = True
            await kwargs["on_content_delta"]("already streamed")
        return await real_model(**kwargs)

    loop.provider.chat_stream_with_retry = model
    msg = inbound()
    msg.metadata["_wants_stream"] = True
    await loop._dispatch(msg)
    assert len(requests) == 2
    assert owner.control is not None
    assert owner.external_calls == 0


async def test_failure_after_owned_runner_never_publishes_generic_error(tmp_path):
    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [LLMResponse(content=None, tool_calls=[call(owner.name)])], owner)
    original_save = loop.sessions.save

    def save(session):
        if owner.external_calls:
            raise OSError("synthetic session write failure")
        return original_save(session)

    loop.sessions.save = save
    await loop._dispatch(inbound())
    outgoing = []
    while not loop.bus.outbound.empty():
        outgoing.append(loop.bus.outbound.get_nowait())
    assert not [msg for msg in outgoing if msg.event is None and msg.content]
    assert owner.external_calls == 1

async def test_parallel_other_chat_and_thread_are_not_owned(tmp_path):
    owner = OwnerTool()
    entered, release = asyncio.Event(), asyncio.Event()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name)]),
        LLMResponse(content=None, tool_calls=[call("message", content="other thread")]),
        LLMResponse(content="other thread final"),
        LLMResponse(content=None, tool_calls=[call("message", content="other chat")]),
        LLMResponse(content="other chat final"),
    ], owner)

    async def action():
        entered.set()
        await release.wait()

    owner.action = action
    owned = asyncio.create_task(loop._process_message(inbound(), ephemeral=True))
    await asyncio.wait_for(entered.wait(), 5)
    other_thread = inbound("ordinary")
    other_thread.metadata["message_thread_id"] = 8
    other_thread.session_key_override = "telegram:123:thread:8"
    await loop._process_message(other_thread, ephemeral=True)
    other_chat = inbound("ordinary")
    other_chat.chat_id = "456"
    await loop._process_message(other_chat, ephemeral=True)
    release.set()
    assert await owned is None
    assert [(msg.chat_id, msg.metadata["message_thread_id"]) for msg in sent] == [("123", 8), ("456", 7)]
    assert len(requests) == 5


async def test_real_task_cancellation_produces_terminal_and_preserves_input(tmp_path):
    from nanobot.agent.tools.delivery_control import current_turn_delivery_control
    owner = OwnerTool()
    entered = asyncio.Event()
    pending = asyncio.Queue()
    loop, requests, sent = make_loop(tmp_path, [LLMResponse(content=None, tool_calls=[call(owner.name)])], owner)

    async def action():
        await pending.put(inbound("next"))
        entered.set()
        await asyncio.Future()

    owner.action = action
    task = asyncio.create_task(loop._process_message(inbound(), pending_queue=pending, ephemeral=True))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    assert await task is None
    assert pending.get_nowait().content == "next"
    assert pending.empty()
    assert sent == []
    assert len(requests) == 1
    assert current_turn_delivery_control() is None


async def test_nonexclusive_tool_cannot_claim_even_when_accidentally_serial(tmp_path):
    owner = OwnerTool()
    owner.exclusive = False
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name)]), LLMResponse(content="denied"),
    ], owner)
    result = await loop._process_message(inbound(), ephemeral=True)
    assert result.content == "denied"
    assert owner.external_calls == 0


async def test_shared_arbitration_denies_claim_while_child_callback_in_flight(tmp_path):
    from nanobot.agent.tools.delivery_control import (
        DeliveryClaimError,
        current_turn_delivery_control,
    )
    started, release = asyncio.Event(), asyncio.Event()
    observed = []

    class RaceOwner(OwnerTool):
        async def execute(self, **kwargs):
            child = asyncio.create_task(loop.tools.get("message").execute(content="FIRST"))
            await started.wait()
            control = current_turn_delivery_control()
            with pytest.raises(DeliveryClaimError):
                control.claim_current_target(owner_tool=self.name)
            observed.append("denied")
            release.set()
            await child
            return "denied"

    owner = RaceOwner()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name)]), LLMResponse(content="done"),
    ], owner)

    async def send(message):
        sent.append(message)
        started.set()
        await release.wait()
        raise TimeoutError("unknown after POST")

    loop.tools.get("message").set_send_callback(send)
    await loop._process_message(inbound(), ephemeral=True)
    assert observed == ["denied"]
    assert owner.external_calls == 0
    assert [message.content for message in sent] == ["FIRST"]


async def test_ordinary_expired_child_keeps_original_message_behavior(tmp_path):
    release = asyncio.Event()
    children = []

    class Ordinary(OwnerTool):
        exclusive = False

        async def execute(self, **kwargs):
            async def delayed():
                await release.wait()
                return await loop.tools.get("message").execute(content="ordinary child")
            children.append(asyncio.create_task(delayed()))
            return "scheduled"

    owner = Ordinary()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name)]), LLMResponse(content="normal"),
    ], owner)
    result = await loop._process_message(inbound(), ephemeral=True)
    assert result.content == "normal"
    release.set()
    await children[0]
    assert [message.content for message in sent] == ["ordinary child"]

async def test_direct_stream_callback_blocks_late_claim(tmp_path):
    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name)]), LLMResponse(content="done"),
    ], owner)
    real_model = loop.provider.chat_stream_with_retry
    chunks = []

    async def stream(chunk):
        chunks.append(chunk)

    async def model(**kwargs):
        if not requests:
            await kwargs["on_content_delta"]("already delivered to direct client")
        return await real_model(**kwargs)

    loop.provider.chat_stream_with_retry = model
    await loop._process_message(inbound(), on_stream=stream, ephemeral=True)
    assert chunks == ["already delivered to direct client"]
    assert owner.control is not None
    assert owner.external_calls == 0


async def test_runner_returns_typed_terminal_without_ack_or_extra_model(tmp_path):
    from nanobot.agent.context import TranscriptInput
    from nanobot.agent.tools.context import RequestContext
    from nanobot.agent.tools.delivery_control import DeliveryClaimError, DeliveryOwnership

    owner = OwnerTool("unknown")
    first, second = call(owner.name), call(owner.name)
    second.id = "duplicate-owner-call"
    loop, requests, sent = make_loop(tmp_path, [LLMResponse(content=None, tool_calls=[first, second])], owner)
    result = await loop._run_agent_loop(
        TranscriptInput(history=[], current_message="news"),
        runtime=loop.llm_runtime(), ephemeral=True,
        request_context=RequestContext(channel="telegram", chat_id="123", turn_id="test-turn", metadata={"message_thread_id": 7}),
    )
    assert result.stop_reason == "delivery_owned"
    assert isinstance(result.delivery_owned, DeliveryOwnership)
    assert result.delivery_owned.turn_id == "test-turn"
    assert result.delivery_owned.thread_id == "7"
    assert result.final_content is None
    assert len(requests) == 1
    assert owner.external_calls == 1
    with pytest.raises(DeliveryClaimError):
        owner.control.claim_current_target(owner_tool=owner.name)

async def test_fresh_turn_receives_matched_history_after_skipped_message(tmp_path):
    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call(owner.name), call("message", content="COPY")]),
        LLMResponse(content="ordinary next turn"),
    ], owner)
    assert await loop._process_message(inbound()) is None
    result = await loop._process_message(inbound("ordinary"))
    assert result.content == "ordinary next turn"
    assert len(requests) == 2
    tool_ids = [tc["id"] for msg in requests[1] for tc in msg.get("tool_calls", [])]
    observations = [msg["tool_call_id"] for msg in requests[1] if msg.get("role") == "tool"]
    assert sorted(tool_ids) == sorted(observations) == ["call-message", "call-owned_delivery"]
    assert sent == []

@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("exclusive", [False, True])
async def test_uninstrumented_side_effect_tool_before_owner_fails_closed(tmp_path, unknown, exclusive):
    external = []

    class DirectSender(Tool):
        name = "direct_sender"
        description = "Existing plugin using an external sender."
        parameters = {"type": "object", "properties": {}}

        @property
        def exclusive(self):
            return exclusive

        async def execute(self, **kwargs):
            external.append("prior POST")
            if unknown:
                raise TimeoutError("unknown prior send")
            return "sent"

    owner = OwnerTool()
    loop, requests, sent = make_loop(tmp_path, [
        LLMResponse(content=None, tool_calls=[call("direct_sender"), call(owner.name)]),
        LLMResponse(content="prior delivery handled"),
    ], owner)
    sender = DirectSender()
    loop.tools.register(sender)
    await loop._process_message(inbound(), ephemeral=True)
    assert external == ["prior POST"]
    assert owner.external_calls == 0


@pytest.mark.parametrize("inject", [False, True])
async def test_read_only_predecessor_allows_claim_unless_input_already_accepted(tmp_path, inject):
    class ReadOnly(Tool):
        name = "lookup"
        description = "Local read."
        parameters = {"type": "object", "properties": {}}
        read_only = True

        async def execute(self, **kwargs):
            return "read"

    owner = OwnerTool()
    responses = (
        [LLMResponse(content=None, tool_calls=[call("lookup")]),
         LLMResponse(content=None, tool_calls=[call(owner.name)]),
         LLMResponse(content="followup handled")]
        if inject else
        [LLMResponse(content=None, tool_calls=[call("lookup"), call(owner.name)])]
    )
    loop, requests, sent = make_loop(tmp_path, responses, owner)
    loop.tools.register(ReadOnly())
    pending = asyncio.Queue()
    if inject:
        await pending.put(inbound("followup"))
    result = await loop._process_message(inbound(), pending_queue=pending, ephemeral=True)
    if inject:
        assert result.content == "followup handled"
        assert owner.external_calls == 0
        assert len(requests) == 3
    else:
        assert result is None
        assert owner.external_calls == 1
        assert len(requests) == 1
    assert pending.empty()
