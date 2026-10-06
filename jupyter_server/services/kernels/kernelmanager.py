"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import asyncio
import json
import os
import pathlib  # noqa: TC003
import sys
import time
import typing as t
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial, wraps

import zmq
from jupyter_client.ioloop.manager import AsyncIOLoopKernelManager
from jupyter_client.multikernelmanager import AsyncMultiKernelManager, MultiKernelManager
from jupyter_client.session import Session
from jupyter_core.paths import exists
from jupyter_core.utils import ensure_async
from jupyter_events import EventLogger
from jupyter_events.schema_registry import SchemaRegistryException

from jupyter_server.services.sessions.lease import compute_kernel_identity

if sys.version_info >= (3, 12):
    from typing import override
else:
    from overrides import overrides as override
from tornado import web
from tornado.concurrent import Future
from tornado.ioloop import IOLoop, PeriodicCallback
from traitlets import (
    Any,
    Bool,
    CaselessStrEnum,
    Dict,
    Float,
    Instance,
    Integer,
    List,
    TraitError,
    Unicode,
    default,
    validate,
)

from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH
from jupyter_server._tz import isoformat, utcnow
from jupyter_server.prometheus.metrics import KERNEL_CURRENTLY_RUNNING_TOTAL
from jupyter_server.utils import ApiPath, import_item, to_os_path


class LeaseOwnershipLost(RuntimeError):
    """Raised when a kernel shutdown is blocked because this process no
    longer owns the kernel's generationed lease.

    The caller must not retry the kill: another server/generation is
    responsible for the kernel now.
    """


class MappingKernelManager(MultiKernelManager):
    """项目内部接口说明。"""

    @default("kernel_manager_class")
    def _default_kernel_manager_class(self):
        return "jupyter_client.ioloop.IOLoopKernelManager"

    kernel_argv = List(Unicode())

    transport_encryption = CaselessStrEnum(
        ["disabled", "auto", "required"],
        default_value="disabled",
        config=True,
        help=(
            "Transport encryption policy for manager-provisioned CurveZMQ keys for all managed kernels. "
            "'disabled' (default) does not provision Curve credentials, 'auto' provisions when the kernelspec "
            "declares support, and 'required' enforces provisioning and fails kernel startup if encryption "
            "cannot be applied."
        ),
    )

    root_dir = Unicode(config=True)

    _kernel_connections = Dict()

    _kernel_ports: dict[str, list[int]] = Dict()  # type: ignore[assignment]

    #: lazily-created asyncio zmq context for recovery heartbeat probes
    _probe_context: t.Any = None

    _culler_callback = None

    _initialized_culler = False

    @default("root_dir")
    def _default_root_dir(self):
        if not self.parent:
            return os.getcwd()
        return self.parent.root_dir

    @validate("root_dir")
    def _update_root_dir(self, proposal):
        """项目内部接口说明。"""
        value = proposal["value"]
        if not os.path.isabs(value):
            # If we receive a non-absolute path, make it absolute.
            value = os.path.abspath(value)
        if not exists(value) or not os.path.isdir(value):
            raise TraitError("kernel root dir %r is not a directory" % value)
        return value

    cull_idle_timeout = Integer(
        0,
        config=True,
        help="""Timeout (in seconds) after which a kernel is considered idle and ready to be culled.
        Values of 0 or lower disable culling. Very short timeouts may result in kernels being culled
        for users with poor network connections.""",
    )

    cull_interval_default = 300  # 5 minutes
    cull_interval = Integer(
        cull_interval_default,
        config=True,
        help="""The interval (in seconds) on which to check for idle kernels exceeding the cull timeout value.""",
    )

    cull_connected = Bool(
        False,
        config=True,
        help="""Whether to consider culling kernels which have one or more connections.
        Only effective if cull_idle_timeout > 0.""",
    )

    cull_busy = Bool(
        False,
        config=True,
        help="""Whether to consider culling kernels which are busy.
        Only effective if cull_idle_timeout > 0.""",
    )

    buffer_offline_messages = Bool(
        True,
        config=True,
        help="""Whether messages from kernels whose frontends have disconnected should be buffered in-memory.

        When True (default), messages are buffered and replayed on reconnect,
        avoiding lost messages due to interrupted connectivity.

        Disable if long-running kernels will produce too much output while
        no frontends are connected.
        """,
    )

    kernel_info_timeout = Float(
        60,
        config=True,
        help="""Timeout for giving up on a kernel (in seconds).

        On starting and restarting kernels, we check whether the
        kernel is running and responsive by sending kernel_info_requests.
        This sets the timeout in seconds for how long the kernel can take
        before being presumed dead.
        This affects the MappingKernelManager (which handles kernel restarts)
        and the ZMQChannelsHandler (which handles the startup).
        """,
    )

    kernel_lease_probe_timeout = Float(
        5.0,
        config=True,
        help="""Timeout (in seconds) for the heartbeat probe used while taking
        over an orphaned kernel after a server crash.

        The probe is the kernel's *proof of life*: a persisted session record
        is only reclaimed once the kernel behind it either fails this probe or
        cannot present the connection file at all.  Wall-clock lease expiry by
        itself never triggers a kill.""",
    )

    _kernel_buffers = Any()

    #: sha256 connection-file fingerprints, keyed by kernel_id
    _kernel_identities: dict[str, str]

    #: kernel ids managed without owning the process (taken over on recovery)
    _kernel_adopted: set[str]

    #: kernel_id -> gate() coroutine/function returning True while *this*
    #: process is the lease owner allowed to destroy the kernel
    _kernel_lease_gates: dict[str, t.Any]

    #: optional global gate(kernel_id) consulted when no per-kernel gate
    #: is registered
    _default_lease_gate: t.Any = None

    #: optional callback(kernel_id, new_identity) fired after a restart
    #: rewrites the connection file
    _identity_changed_callback: t.Any = None

    def set_default_lease_gate(self, gate: t.Any) -> None:
        """Install a process-wide ownership predicate for kernel kills.

        ``gate(kernel_id)`` returns True (possibly via a coroutine) while
        this process may destroy the kernel.  It is consulted for every
        terminal shutdown unless a more specific gate was registered with
        :meth:`set_lease_gate`.
        """
        self._default_lease_gate = gate

    @default("_kernel_buffers")
    def _default_kernel_buffers(self):
        return defaultdict(lambda: {"buffer": [], "session_key": "", "channels": {}})

    last_kernel_activity = Instance(
        datetime,
        help="The last activity on any kernel, including shutting down a kernel",
    )

    def __init__(self, **kwargs):
        """项目内部接口说明。"""
        self.pinned_superclass = MultiKernelManager
        self._pending_kernel_tasks = {}
        self._kernel_identities = getattr(self, "_kernel_identities", None) or {}
        self._kernel_adopted = getattr(self, "_kernel_adopted", None) or set()
        self._kernel_lease_gates = getattr(self, "_kernel_lease_gates", None) or {}
        self.pinned_superclass.__init__(self, **kwargs)
        self.last_kernel_activity = utcnow()

    allowed_message_types = List(
        trait=Unicode(),
        config=True,
        help="""White list of allowed kernel message types.
        When the list is empty, all message types are allowed.
        """,
    )

    allow_tracebacks = Bool(
        True, config=True, help=("Whether to send tracebacks to clients on exceptions.")
    )

    traceback_replacement_message = Unicode(
        "An exception occurred at runtime, which is not shown due to security reasons.",
        config=True,
        help=("Message to print when allow_tracebacks is False, and an exception occurs"),
    )

    # -------------------------------------------------------------------------
    # Methods for managing kernels and sessions
    # -------------------------------------------------------------------------

    def _handle_kernel_died(self, kernel_id):
        """项目内部接口说明。"""
        self.log.warning("Kernel %s died, removing from map.", kernel_id)
        self.remove_kernel(kernel_id)
        self._kernel_adopted.discard(kernel_id)
        self._kernel_identities.pop(kernel_id, None)
        self._kernel_ports.pop(kernel_id, None)
        self._kernel_lease_gates.pop(kernel_id, None)

    def cwd_for_path(self, path, **kwargs):
        """项目内部接口说明。"""
        os_path = to_os_path(path, self.root_dir)
        # in the case of documents and kernels not being on the same filesystem,
        # walk up to root_dir if the paths don't exist
        while not os.path.isdir(os_path) and os_path != self.root_dir:
            os_path = os.path.dirname(os_path)
        return os_path

    def _kernel_start_kwargs(self, **kwargs: t.Any) -> dict[str, t.Any]:
        """项目内部接口说明。"""
        launch_kwargs = dict(kwargs)
        if self.transport_encryption != "disabled":
            launch_kwargs["transport_encryption"] = self.transport_encryption
        return launch_kwargs

    async def _remove_kernel_when_ready(self, kernel_id, kernel_awaitable):
        """项目内部接口说明。"""
        await super()._remove_kernel_when_ready(kernel_id, kernel_awaitable)
        self._kernel_connections.pop(kernel_id, None)
        self._kernel_ports.pop(kernel_id, None)
        self._kernel_identities.pop(kernel_id, None)
        self._kernel_lease_gates.pop(kernel_id, None)
        self._kernel_adopted.discard(kernel_id)

    # TODO: DEC 2022: Revise the type-ignore once the signatures have been changed upstream
    # https://github.com/jupyter/jupyter_client/pull/905
    async def _async_start_kernel(  # type:ignore[override]
        self, *, kernel_id: str | None = None, path: ApiPath | None = None, **kwargs: t.Any
    ) -> str:
        """项目内部接口说明。"""
        if kernel_id is None or kernel_id not in self:
            kwargs = self._kernel_start_kwargs(**kwargs)
            if path is not None:
                kwargs["cwd"] = self.cwd_for_path(path, env=kwargs.get("env", {}))
            if kernel_id is not None:
                assert kernel_id is not None, "Never Fail, but necessary for mypy "
                kwargs["kernel_id"] = kernel_id
            kernel_id = await self.pinned_superclass._async_start_kernel(self, **kwargs)
            self._kernel_connections[kernel_id] = 0

            # add busy/activity markers:
            kernel = self.get_kernel(kernel_id)
            kernel.execution_state = "starting"  # type:ignore[attr-defined]
            kernel.reason = ""  # type:ignore[attr-defined]
            kernel.last_activity = utcnow()  # type:ignore[attr-defined]
            self.log.info("Kernel started: %s", kernel_id)
            self.log.debug(
                "Kernel args (excluding env): %r", {k: v for k, v in kwargs.items() if k != "env"}
            )
            env = kwargs.get("env")
            if env and isinstance(env, dict):  # type:ignore[unreachable]
                self.log.debug("Kernel argument 'env' passed with: %r", list(env.keys()))  # type:ignore[unreachable]

            task = asyncio.create_task(self._finish_kernel_start(kernel_id))
            if not getattr(self, "use_pending_kernels", None):
                await task
            else:
                self._pending_kernel_tasks[kernel_id] = task

            # Increase the metric of number of kernels running
            # for the relevant kernel type by 1
            KERNEL_CURRENTLY_RUNNING_TOTAL.labels(type=self._kernels[kernel_id].kernel_name).inc()

        else:
            self.log.info("Using existing kernel: %s", kernel_id)

        # Initialize culling if not already
        if not self._initialized_culler:
            self.initialize_culler()
        assert kernel_id is not None
        return kernel_id

    # see https://github.com/jupyter-server/jupyter_server/issues/1165
    # this assignment is technically incorrect, but might need a change of API
    # in jupyter_client.
    start_kernel = _async_start_kernel  # type:ignore[assignment]

    # ------------------------------------------------------------------
    # generationed lease support: identity, adoption, ownership gate
    # ------------------------------------------------------------------
    def _connection_file_path(self, kernel_id: str) -> str:
        return os.path.join(self.connection_dir, "kernel-%s.json" % kernel_id)

    def _load_connection_info(self, kernel_id: str) -> dict[str, t.Any] | None:
        """Read and parse a kernel connection file, or None if absent/bad."""
        path = self._connection_file_path(kernel_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return t.cast("dict[str, t.Any]", json.load(f))
        except (OSError, json.JSONDecodeError):
            self.log.warning("Unreadable kernel connection file for %s: %s", kernel_id, path)
            return None

    def read_kernel_identity(self, kernel_id: str) -> str | None:
        """Return the persisted identity of a kernel, or None if no
        connection file exists (kernel truly gone).

        The identity is a hash of the five ZMQ ports and the signing key
        recorded in ``kernel-<id>.json``; see
        :func:`jupyter_server.services.sessions.lease.compute_kernel_identity`.
        """
        info = self._load_connection_info(kernel_id)
        if info is None:
            return None
        return compute_kernel_identity(info)

    def current_kernel_identity(self, kernel_id: str) -> str | None:
        """Identity of a kernel this manager holds, computed live."""
        km = self._kernels[kernel_id]
        try:
            return compute_kernel_identity(km.get_connection_info())
        except Exception:
            return self.read_kernel_identity(kernel_id)

    async def probe_kernel(self, kernel_id: str, timeout: float | None = None) -> bool:
        """Send a single heartbeat ping to an unattached kernel.

        This is the *proof of life* used during recovery: success means
        someone (a live peer server, a client) may still be talking to the
        kernel, so an expired wall-clock fence must not be acted upon.
        """
        info = self._load_connection_info(kernel_id)
        if info is None:
            return False
        try:
            transport = info.get("transport", "tcp")
            ip = info.get("ip", "127.0.0.1")
            address = f"{transport}://{ip}:{info['hb_port']}"
            # Use an asyncio ZMQ context so send/poll/recv are awaitable;
            # reuse the multi-kernel-manager's context when available.
            context = getattr(self, "context", None)
            if context is None or not isinstance(context, zmq.asyncio.Context):
                if MappingKernelManager._probe_context is None:
                    MappingKernelManager._probe_context = zmq.asyncio.Context()
                context = MappingKernelManager._probe_context
            socket = context.socket(zmq.REQ)
            socket.linger = 100
            try:
                socket.connect(address)
                await socket.send(b"ping")
                if await socket.poll((timeout or self.kernel_lease_probe_timeout) * 1000):
                    await socket.recv()
                    return True
                return False
            finally:
                socket.close(linger=100)
        except Exception as e:
            self.log.debug("Heartbeat probe failed for kernel %s: %s", kernel_id, e)
            return False

    async def adopt_kernel(self, kernel_id: str, *, kernel_name: str | None = None) -> bool:
        """Attach to an already-running kernel without launching a process.

        Used when this server wins the lease take-over during recovery.
        The kernel manager is constructed in non-owning mode
        (``owns_kernel=False``): we can talk to the kernel and will delete
        its session, but jupyter_client will never kill a process we did
        not start.  Ownership of the process can only be proven by the
        successful heartbeat probe performed by the caller.
        """
        if kernel_id in self:
            return True
        path = self._connection_file_path(kernel_id)
        file_info = self._load_connection_info(kernel_id)
        if file_info is None:
            return False
        constructor_kwargs: dict[str, t.Any] = {}
        if self.kernel_spec_manager:
            constructor_kwargs["kernel_spec_manager"] = self.kernel_spec_manager
        kernel_name = file_info.get("kernel_name") or kernel_name or "python3"
        km = self.kernel_manager_factory(
            connection_file=path,
            parent=self,
            log=self.log,
            kernel_name=kernel_name or "python3",
            owns_kernel=False,
            **constructor_kwargs,
        )
        try:
            km.load_connection_file()
        except Exception as e:
            self.log.warning("Cannot adopt kernel %s: bad connection file: %s", kernel_id, e)
            return False
        # This manager never launched the process; mark its ready future
        # done so the websocket/startup paths treat it as a live kernel.
        ready = getattr(km, "ready", None)
        if ready is not None and not ready.done():
            if isinstance(ready, asyncio.Future):
                ready.set_result(None)
            else:
                ready.set_result(None)
        km.execution_state = "idle"  # type:ignore[attr-defined]
        km.reason = ""  # type:ignore[attr-defined]
        km.last_activity = utcnow()  # type:ignore[attr-defined]
        km._jupyter_server_adopted = True  # type:ignore[attr-defined]
        self._kernels[kernel_id] = km
        self._kernel_adopted.add(kernel_id)
        self._kernel_connections[kernel_id] = 0
        self._kernel_identities[kernel_id] = self.read_kernel_identity(kernel_id) or ""
        self.log.info("Took over orphaned kernel %s without restarting it.", kernel_id)
        return True

    async def _adopted_shutdown(self, km, *, now: bool, restart: bool) -> None:
        """Shutdown a kernel adopted without a provisioner.

        The kernel manager for an adopted kernel never launched the process
        and has no provisioner, so the jupyter_client shutdown path (which
        asserts on a provisioner and signals the process) cannot run.  Ask
        the kernel to exit over the control channel instead; never signal a
        foreign process.  If the kernel ignores the request it is left
        running -- we only drop our attachment and remove the connection
        file we were holding.
        """
        client = None
        try:
            client = km.client()
            # Fire the shutdown_request on the control channel only.  The
            # blocking client uses daemon threads and start/stop here are
            # non-blocking; we poll the heartbeat asynchronously afterwards.
            client.start_channels(shell=False, iopub=False, stdin=False, hb=False)
            client.shutdown(restart=restart)
        except Exception as e:
            self.log.debug("Control-channel shutdown of adopted kernel failed: %s", e)
        finally:
            if client is not None:
                try:
                    client.stop_channels()
                except Exception:
                    pass
        gone = await self._wait_for_kernel_exit(km.kernel_id, timeout=1.0 if now else 5.0)
        if not gone:
            self.log.warning(
                "Adopted kernel %s did not confirm shutdown; leaving its process alone.",
                km.kernel_id,
            )
        if not restart:
            try:
                km.cleanup_connection_file()
            except Exception:
                pass
        try:
            km.cleanup_ipc_files()
        except Exception:
            pass

    async def _wait_for_kernel_exit(self, kernel_id: str, timeout: float = 5.0) -> bool:
        """Poll the heartbeat port until the kernel stops answering."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            if not await self.probe_kernel(kernel_id, timeout=0.5):
                return True
        return False

    def abandon_kernel(self, kernel_id: str) -> None:
        """Detach from an adopted kernel without signalling it.

        Used when recovery decides not to take ownership (quarantine or a
        lost race): the live kernel keeps running for whoever actually owns
        it.
        """
        km = self._kernels.pop(kernel_id, None)
        self._kernel_adopted.discard(kernel_id)
        self._kernel_connections.pop(kernel_id, None)
        self._kernel_ports.pop(kernel_id, None)
        self._kernel_identities.pop(kernel_id, None)
        self._kernel_lease_gates.pop(kernel_id, None)
        if km is not None:
            # Adopted managers never launch a process and, at adoption
            # time, have opened no channels; just release the activity
            # monitor if recovery happened to start one.  We deliberately
            # do not call cleanup_resources(): on async managers it is a
            # coroutine, and it also deletes the connection file, which
            # still belongs to the live kernel we are abandoning.
            activity_stream = getattr(km, "_activity_stream", None)
            if activity_stream is not None and not activity_stream.socket.closed:
                activity_stream.close()
            km._activity_stream = None

    def set_lease_gate(self, kernel_id: str, gate: t.Any) -> None:
        """Register an ownership predicate for a single kernel.

        ``gate(kernel_id)`` is a sync callable or coroutine function
        returning True while this process is the current lease owner of
        the kernel.  Shutdown consults it right before killing the
        process, closing the window between a session delete decision and
        the actual kill.
        """
        self._kernel_lease_gates[kernel_id] = gate

    def clear_lease_gate(self, kernel_id: str) -> None:
        self._kernel_lease_gates.pop(kernel_id, None)

    async def _lease_allows_destroy(self, kernel_id: str) -> bool:
        gate = self._kernel_lease_gates.get(kernel_id, self._default_lease_gate)
        if gate is None:
            # No gate registered: backwards-compatible behaviour.
            return True
        try:
            result = gate(kernel_id)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as e:
            self.log.warning("Lease gate for kernel %s failed (%s); refusing to kill.", kernel_id, e)
            return False
        return bool(result)

    async def _finish_kernel_start(self, kernel_id):
        """项目内部接口说明。"""
        km = self.get_kernel(kernel_id)
        self.log.debug("Waiting for kernel %s", kernel_id)
        if hasattr(km, "ready"):
            ready = km.ready
            if not isinstance(ready, asyncio.Future):
                ready = asyncio.wrap_future(ready)
            try:
                await ready
            except Exception:
                self.log.exception("Error waiting for kernel manager ready")
                return
        self.log.debug("Kernel %s ready", kernel_id)

        self._kernel_ports[kernel_id] = km.ports
        # Record the kernel's identity from the freshly written connection
        # file; recovery compares it against the persisted lease so a kernel
        # restart that recycled ZMQ ports is never mistaken for the original.
        self._kernel_identities[kernel_id] = self.read_kernel_identity(kernel_id) or ""
        self.start_watching_activity(kernel_id)
        # register callback for failed auto-restart
        self.add_restart_callback(
            kernel_id,
            lambda: self._handle_kernel_died(kernel_id),
            "dead",
        )

    def ports_changed(self, kernel_id):
        """项目内部接口说明。"""
        changed_ports = self._get_changed_ports(kernel_id)
        if changed_ports:
            # If changed, update captured ports and return True, else return False.
            self.log.debug("Port change detected for kernel: %s", kernel_id)
            self._kernel_ports[kernel_id] = changed_ports
            return True
        return False

    def _get_changed_ports(self, kernel_id):
        """项目内部接口说明。"""
        # Get current ports and return comparison with ports captured at startup.
        km = self.get_kernel(kernel_id)
        assert isinstance(km.ports, list)
        assert isinstance(self._kernel_ports[kernel_id], list)
        if km.ports != self._kernel_ports[kernel_id]:
            return km.ports
        return None

    def start_buffering(self, kernel_id, session_key, channels):
        """项目内部接口说明。"""

        if not self.buffer_offline_messages:
            for stream in channels.values():
                stream.close()
            return

        self.log.info("Starting buffering for %s", session_key)
        self._check_kernel_id(kernel_id)
        # clear previous buffering state
        self.stop_buffering(kernel_id)
        buffer_info = self._kernel_buffers[kernel_id]
        # record the session key because only one session can buffer
        buffer_info["session_key"] = session_key
        # TODO: the buffer should likely be a memory bounded queue, we're starting with a list to keep it simple
        buffer_info["buffer"] = []
        buffer_info["channels"] = channels

        # forward any future messages to the internal buffer
        def buffer_msg(channel, msg_parts):
            self.log.debug("Buffering msg on %s:%s", kernel_id, channel)
            buffer_info["buffer"].append((channel, msg_parts))

        for channel, stream in channels.items():
            stream.on_recv(partial(buffer_msg, channel))

    def get_buffer(self, kernel_id, session_key):
        """项目内部接口说明。"""
        self.log.debug("Getting buffer for %s", kernel_id)
        if kernel_id not in self._kernel_buffers:
            return None

        buffer_info = self._kernel_buffers[kernel_id]
        if buffer_info["session_key"] == session_key:
            # remove buffer
            self._kernel_buffers.pop(kernel_id)
            # only return buffer_info if it's a match
            return buffer_info
        else:
            self.stop_buffering(kernel_id)

    def stop_buffering(self, kernel_id):
        """项目内部接口说明。"""
        self.log.debug("Clearing buffer for %s", kernel_id)
        self._check_kernel_id(kernel_id)

        if kernel_id not in self._kernel_buffers:
            return
        buffer_info = self._kernel_buffers.pop(kernel_id)
        # close buffering streams
        for stream in buffer_info["channels"].values():
            if not stream.socket.closed:
                stream.on_recv(None)
                stream.close()

        msg_buffer = buffer_info["buffer"]
        if msg_buffer:
            self.log.info(
                "Discarding %s buffered messages for %s",
                len(msg_buffer),
                buffer_info["session_key"],
            )

    async def _async_shutdown_kernel(self, kernel_id, now=False, restart=False):
        """项目内部接口说明。"""
        self._check_kernel_id(kernel_id)

        # A restart never changes who owns the kernel; only a terminal
        # shutdown needs the fencing check, because that is where we might
        # kill a peer server's kernel.
        if not restart and not await self._lease_allows_destroy(kernel_id):
            self.log.error(
                "Refusing to shut down kernel %s: another generation owns its lease.",
                kernel_id,
            )
            if kernel_id in self._kernel_adopted:
                self.abandon_kernel(kernel_id)
            raise LeaseOwnershipLost(kernel_id)

        # Decrease the metric of number of kernels
        # running for the relevant kernel type by 1
        KERNEL_CURRENTLY_RUNNING_TOTAL.labels(type=self._kernels[kernel_id].kernel_name).dec()

        if kernel_id in self._pending_kernel_tasks:
            task = self._pending_kernel_tasks.pop(kernel_id)
            task.cancel()

        self.stop_watching_activity(kernel_id)
        self.stop_buffering(kernel_id)

        # A kernel adopted during recovery has no provisioner and never
        # launched its process through this manager; the jupyter_client
        # shutdown path would no-op (owns_kernel=False) or assert on the
        # missing provisioner.  Drive the control-channel based shutdown
        # instead, then drop it from the map.
        if kernel_id in self._kernel_adopted and not restart:
            km = self._kernels[kernel_id]
            await self._adopted_shutdown(km, now=now, restart=restart)
            self.remove_kernel(kernel_id)
            self._kernel_adopted.discard(kernel_id)
            self._kernel_connections.pop(kernel_id, None)
            self._kernel_ports.pop(kernel_id, None)
            self._kernel_identities.pop(kernel_id, None)
            self._kernel_lease_gates.pop(kernel_id, None)
            return

        return await self.pinned_superclass._async_shutdown_kernel(
            self, kernel_id, now=now, restart=restart
        )

    shutdown_kernel = _async_shutdown_kernel

    async def _async_restart_kernel(self, kernel_id, now=False):
        """项目内部接口说明。"""
        self._check_kernel_id(kernel_id)

        # A kernel adopted after a crash was never launched from this
        # manager, so jupyter_client's restart path (reuse _launch_args)
        # cannot work.  Ask the adopted kernel to shut down, drop the
        # non-owning attachment, and launch a fresh process under our own
        # ownership with the same kernel id.  The new connection file
        # yields a new identity, which is recorded in the lease.
        if kernel_id in self._kernel_adopted:
            km = self._kernels[kernel_id]
            kernel_name = km.kernel_name
            await self._adopted_shutdown(km, now=now, restart=False)
            self.remove_kernel(kernel_id)
            self._kernel_adopted.discard(kernel_id)
            self._kernel_connections.pop(kernel_id, None)
            self._kernel_ports.pop(kernel_id, None)
            self._kernel_identities.pop(kernel_id, None)
            await self._async_start_kernel(kernel_id=kernel_id, kernel_name=kernel_name)
        else:
            await self.pinned_superclass._async_restart_kernel(self, kernel_id, now=now)
        # A restart can rewrite the connection file with different ports;
        # refresh the identity and tell the lease owner so the persisted
        # record never lags behind the real kernel.
        self._kernel_identities[kernel_id] = self.read_kernel_identity(kernel_id) or ""
        if self._identity_changed_callback is not None:
            try:
                result = self._identity_changed_callback(
                    kernel_id, self._kernel_identities[kernel_id]
                )
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                self.log.exception("Identity-changed callback failed for kernel %s", kernel_id)
        kernel = self.get_kernel(kernel_id)
        # return a Future that will resolve when the kernel has successfully restarted
        channel = kernel.connect_shell()
        future: Future[Any] = Future()

        def finish():
            """项目内部接口说明。"""
            if not channel.closed():  # type:ignore[operator]
                channel.close()
            loop.remove_timeout(timeout)
            kernel.remove_restart_callback(on_restart_failed, "dead")
            kernel._pending_restart_cleanup = None  # type:ignore[attr-defined]

        def on_reply(msg):
            self.log.debug("Kernel info reply received: %s", kernel_id)
            finish()
            if not future.done():
                future.set_result(msg)

        def on_timeout():
            self.log.warning("Timeout waiting for kernel_info_reply: %s", kernel_id)
            finish()
            if not future.done():
                future.set_exception(TimeoutError("Timeout waiting for restart"))

        def on_restart_failed():
            self.log.warning("Restarting kernel failed: %s", kernel_id)
            finish()
            if not future.done():
                future.set_exception(RuntimeError("Restart failed"))

        kernel.add_restart_callback(on_restart_failed, "dead")
        kernel._pending_restart_cleanup = finish  # type:ignore[attr-defined]
        kernel.session.send(channel, "kernel_info_request")
        channel.on_recv(on_reply)  # type:ignore[operator]
        loop = IOLoop.current()
        timeout = loop.add_timeout(loop.time() + self.kernel_info_timeout, on_timeout)
        # Re-establish activity watching if ports have changed...
        if self._get_changed_ports(kernel_id) is not None:
            self.stop_watching_activity(kernel_id)
            self.execution_state = "starting"
            self.start_watching_activity(kernel_id)
        return future

    restart_kernel = _async_restart_kernel

    def notify_connect(self, kernel_id):
        """项目内部接口说明。"""
        if kernel_id in self._kernel_connections:
            self._kernel_connections[kernel_id] += 1

    def notify_disconnect(self, kernel_id):
        """项目内部接口说明。"""
        if kernel_id in self._kernel_connections:
            self._kernel_connections[kernel_id] -= 1

    def kernel_model(self, kernel_id):
        """项目内部接口说明。"""
        self._check_kernel_id(kernel_id)
        kernel = self._kernels[kernel_id]

        model = {
            "id": kernel_id,
            "name": kernel.kernel_name,
            "last_activity": isoformat(kernel.last_activity),
            "execution_state": kernel.execution_state,
            "connections": self._kernel_connections.get(kernel_id, 0),
        }
        if getattr(kernel, "reason", None):
            model["reason"] = kernel.reason
        return model

    def list_kernels(self):
        """项目内部接口说明。"""
        kernels = []
        kernel_ids = self.pinned_superclass.list_kernel_ids(self)
        for kernel_id in kernel_ids:
            try:
                model = self.kernel_model(kernel_id)
                kernels.append(model)
            except (web.HTTPError, KeyError):
                # Probably due to a (now) non-existent kernel, continue building the list
                pass
        return kernels

    # override _check_kernel_id to raise 404 instead of KeyError
    def _check_kernel_id(self, kernel_id):
        """项目内部接口说明。"""
        if kernel_id not in self:
            raise web.HTTPError(404, "Kernel does not exist: %s" % kernel_id)

    # monitoring activity:
    untracked_message_types = List(
        trait=Unicode(),
        config=True,
        default_value=[
            "comm_info_request",
            "comm_info_reply",
            "kernel_info_request",
            "kernel_info_reply",
            "shutdown_request",
            "shutdown_reply",
            "interrupt_request",
            "interrupt_reply",
            "debug_request",
            "debug_reply",
            "stream",
            "display_data",
            "update_display_data",
            "execute_input",
            "execute_result",
            "error",
            "status",
            "clear_output",
            "debug_event",
            "input_request",
            "input_reply",
        ],
        help="""List of kernel message types excluded from user activity tracking.

        This should be a superset of the message types sent on any channel other
        than the shell channel.""",
    )

    def track_message_type(self, message_type):
        return message_type not in self.untracked_message_types

    def start_watching_activity(self, kernel_id):
        """项目内部接口说明。"""
        self.log.debug("Watching kernel activity: %s", kernel_id)
        kernel = self._kernels[kernel_id]
        # add busy/activity markers:
        kernel.reason = ""
        kernel.last_activity = utcnow()
        kernel._activity_stream = kernel.connect_iopub()
        session = Session(
            config=kernel.session.config,
            key=kernel.session.key,
        )

        def record_activity(msg_list):
            """项目内部接口说明。"""
            _idents, fed_msg_list = session.feed_identities(msg_list)
            msg = session.deserialize(fed_msg_list, content=False)

            msg_type = msg["header"]["msg_type"]
            parent_header = msg.get("parent_header")
            parent_msg_type = None if parent_header is None else parent_header.get("msg_type")
            if (
                self.track_message_type(msg_type)
                or self.track_message_type(parent_msg_type)
                or kernel.execution_state == "busy"
            ):
                self.last_kernel_activity = kernel.last_activity = utcnow()
            if msg_type == "status":
                msg = session.deserialize(fed_msg_list)
                execution_state = msg["content"]["execution_state"]
                if self.track_message_type(parent_msg_type):
                    kernel.execution_state = execution_state
                elif kernel.execution_state == "starting" and execution_state != "starting":
                    # We always normalize post-starting execution state to "idle"
                    # unless we know that the status is in response to one of our
                    # tracked message types.
                    kernel.execution_state = "idle"
                self.log.debug(
                    "activity on %s: %s (%s)",
                    kernel_id,
                    msg_type,
                    kernel.execution_state,
                )
            else:
                self.log.debug("activity on %s: %s", kernel_id, msg_type)

        kernel._activity_stream.on_recv(record_activity)

    def stop_watching_activity(self, kernel_id):
        """项目内部接口说明。"""
        kernel = self._kernels[kernel_id]
        if getattr(kernel, "_activity_stream", None):
            if not kernel._activity_stream.socket.closed:
                kernel._activity_stream.close()
            kernel._activity_stream = None
        if getattr(kernel, "_pending_restart_cleanup", None):
            kernel._pending_restart_cleanup()

    def initialize_culler(self):
        """项目内部接口说明。"""
        if (
            not self._initialized_culler
            and self.cull_idle_timeout > 0
            and self._culler_callback is None
        ):
            _ = IOLoop.current()
            if self.cull_interval <= 0:  # handle case where user set invalid value
                self.log.warning(
                    "Invalid value for 'cull_interval' detected (%s) - using default value (%s).",
                    self.cull_interval,
                    self.cull_interval_default,
                )
                self.cull_interval = self.cull_interval_default
            self._culler_callback = PeriodicCallback(self.cull_kernels, 1000 * self.cull_interval)
            self.log.info(
                "Culling kernels with idle durations > %s seconds at %s second intervals ...",
                self.cull_idle_timeout,
                self.cull_interval,
            )
            if self.cull_busy:
                self.log.info("Culling kernels even if busy")
            if self.cull_connected:
                self.log.info("Culling kernels even with connected clients")
            self._culler_callback.start()

        self._initialized_culler = True

    async def cull_kernels(self):
        """项目内部接口说明。"""
        self.log.debug(
            "Polling every %s seconds for kernels idle > %s seconds...",
            self.cull_interval,
            self.cull_idle_timeout,
        )
        """Create a separate list of kernels to avoid conflicting updates while iterating"""
        for kernel_id in list(self._kernels):
            try:
                await self.cull_kernel_if_idle(kernel_id)
            except Exception as e:
                self.log.exception(
                    "The following exception was encountered while checking the idle duration of kernel %s: %s",
                    kernel_id,
                    e,
                )

    async def cull_kernel_if_idle(self, kernel_id):
        """项目内部接口说明。"""
        kernel = self._kernels[kernel_id]

        if getattr(kernel, "execution_state", None) == "dead":
            self.log.warning(
                "Culling '%s' dead kernel '%s' (%s).",
                kernel.execution_state,
                kernel.kernel_name,
                kernel_id,
            )
            await ensure_async(self.shutdown_kernel(kernel_id))
            return

        kernel_spec_metadata = kernel.kernel_spec.metadata
        cull_idle_timeout = kernel_spec_metadata.get("cull_idle_timeout", self.cull_idle_timeout)

        if hasattr(
            kernel, "last_activity"
        ):  # last_activity is monkey-patched, so ensure that has occurred
            self.log.debug(
                "kernel_id=%s, kernel_name=%s, last_activity=%s",
                kernel_id,
                kernel.kernel_name,
                kernel.last_activity,
            )
            dt_now = utcnow()
            dt_idle = dt_now - kernel.last_activity
            # Compute idle properties
            is_idle_time = dt_idle > timedelta(seconds=cull_idle_timeout)
            is_idle_execute = self.cull_busy or (kernel.execution_state != "busy")
            connections = self._kernel_connections.get(kernel_id, 0)
            is_idle_connected = self.cull_connected or not connections
            # Cull the kernel if all three criteria are met
            if is_idle_time and is_idle_execute and is_idle_connected:
                idle_duration = int(dt_idle.total_seconds())
                self.log.warning(
                    "Culling '%s' kernel '%s' (%s) with %d connections due to %s seconds of inactivity.",
                    kernel.execution_state,
                    kernel.kernel_name,
                    kernel_id,
                    connections,
                    idle_duration,
                )
                await ensure_async(self.shutdown_kernel(kernel_id))


# AsyncMappingKernelManager inherits as much as possible from MappingKernelManager,
# overriding only what is different.
class AsyncMappingKernelManager(MappingKernelManager, AsyncMultiKernelManager):
    """项目内部接口说明。"""

    @default("kernel_manager_class")
    def _default_kernel_manager_class(self):
        return "jupyter_server.services.kernels.kernelmanager.ServerKernelManager"

    @validate("kernel_manager_class")
    def _validate_kernel_manager_class(self, proposal):
        """项目内部接口说明。"""
        km_class_value = proposal.value
        km_class = import_item(km_class_value)
        if not issubclass(km_class, ServerKernelManager):
            warnings.warn(
                f"KernelManager class '{km_class}' is not a subclass of 'ServerKernelManager'.  Custom "
                "KernelManager classes should derive from 'ServerKernelManager' beginning with jupyter-server 2.0 "
                "or risk missing functionality.  Continuing...",
                FutureWarning,
                stacklevel=3,
            )
        return km_class_value

    def __init__(self, **kwargs):
        """项目内部接口说明。"""
        self.pinned_superclass = MultiKernelManager
        self._pending_kernel_tasks = {}
        self._kernel_identities = getattr(self, "_kernel_identities", None) or {}
        self._kernel_adopted = getattr(self, "_kernel_adopted", None) or set()
        self._kernel_lease_gates = getattr(self, "_kernel_lease_gates", None) or {}
        self.pinned_superclass.__init__(self, **kwargs)
        self.last_kernel_activity = utcnow()


def emit_kernel_action_event(success_msg: str = "") -> t.Callable[..., t.Any]:
    """项目内部接口说明。"""

    def wrap_method(method):
        @wraps(method)
        async def wrapped_method(self, *args, **kwargs):
            """项目内部接口说明。"""
            # Get the method name from the
            action = method.__name__.replace("_kernel", "")
            # If the method succeeds, emit a success event.
            try:
                out = await method(self, *args, **kwargs)
                data = {
                    "kernel_name": self.kernel_name,
                    "action": action,
                    "status": "success",
                    "msg": success_msg.format(
                        kernel_id=self.kernel_id, kernel_name=self.kernel_name, action=action
                    ),
                }
                if self.kernel_id:
                    data["kernel_id"] = self.kernel_id
                self.emit(
                    schema_id="https://events.jupyter.org/jupyter_server/kernel_actions/v1",
                    data=data,
                )
                return out
            # If the method fails, emit a failed event.
            except Exception as err:
                data = {
                    "kernel_name": self.kernel_name,
                    "action": action,
                    "status": "error",
                    "msg": str(err),
                }
                if self.kernel_id:
                    data["kernel_id"] = self.kernel_id
                # If the exception is an HTTPError (usually via a gateway request)
                # log the status_code and HTTPError log_message.
                if isinstance(err, web.HTTPError):
                    msg = err.log_message or ""
                    data["status_code"] = err.status_code
                    data["msg"] = msg
                self.emit(
                    schema_id="https://events.jupyter.org/jupyter_server/kernel_actions/v1",
                    data=data,
                )
                raise err

        return wrapped_method

    return wrap_method


class ServerKernelManager(AsyncIOLoopKernelManager):
    """项目内部接口说明。"""

    # Define activity-related attributes:
    execution_state = Unicode(
        None, allow_none=True, help="The current execution state of the kernel"
    )
    reason = Unicode("", help="The reason for the last failure against the kernel")

    last_activity = Instance(datetime, help="The last activity on the kernel")

    # A list of pathlib objects, each pointing at an event
    # schema to register with this kernel manager's eventlogger.
    # This trait should not be overridden.
    @property
    def core_event_schema_paths(self) -> list[pathlib.Path]:
        return [DEFAULT_EVENTS_SCHEMA_PATH / "kernel_actions" / "v1.yaml"]

    # This trait is intended for subclasses to override and define
    # custom event schemas.
    extra_event_schema_paths: List[str] = List(
        default_value=[],
        help="""
        A list of pathlib.Path objects pointing at to register with
        the kernel manager's eventlogger.
        """,
    ).tag(config=True)

    event_logger = Instance(EventLogger)

    @default("event_logger")
    def _default_event_logger(self):
        """项目内部接口说明。"""
        if (
            self.parent is not None
            and self.parent.parent is not None
            and hasattr(self.parent.parent, "event_logger")
        ):
            logger = self.parent.parent.event_logger
        else:
            # If parent does not have an event logger, create one.
            logger = EventLogger()
        # Ensure that all the expected schemas are registered. If not, register them.
        schemas = self.core_event_schema_paths + self.extra_event_schema_paths
        for schema_path in schemas:
            # Try registering the event.
            try:
                logger.register_event_schema(schema_path)
            # Pass if it already exists.
            except SchemaRegistryException:
                pass
        return logger

    def emit(self, schema_id, data):
        """项目内部接口说明。"""
        self.event_logger.emit(schema_id=schema_id, data=data)

    @override
    @emit_kernel_action_event(
        success_msg="Kernel {kernel_id} was started.",
    )
    async def start_kernel(self, *args, **kwargs):
        return await super().start_kernel(*args, **kwargs)

    @override
    @emit_kernel_action_event(
        success_msg="Kernel {kernel_id} was shutdown.",
    )
    async def shutdown_kernel(self, *args, **kwargs):
        return await super().shutdown_kernel(*args, **kwargs)

    @override
    @emit_kernel_action_event(
        success_msg="Kernel {kernel_id} was restarted.",
    )
    async def restart_kernel(self, *args, **kwargs):
        return await super().restart_kernel(*args, **kwargs)

    @override
    @emit_kernel_action_event(
        success_msg="Kernel {kernel_id} was interrupted.",
    )
    async def interrupt_kernel(self, *args, **kwargs):
        return await super().interrupt_kernel(*args, **kwargs)
