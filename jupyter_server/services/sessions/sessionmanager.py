"""Session manager with generationed ownership leases.

See :mod:`jupyter_server.services.sessions.lease` for the fencing
protocol.  At a high level, every persisted session row records *who*
owns it (``server_id``), *which generation* of ownership it is in, *when*
the owner's fence expires and *which kernel* (identity hash) it refers
to.  On startup the manager reconciles each row against three sources of
truth:

1. the persisted record,
2. the kernels currently visible to this process, and
3. the kernel's own proof -- a heartbeat reply from the ZMQ identity the
   record names.

Only then does it decide to **take over** (orphaned, alive, identity
matches), **quarantine** (record and kernel disagree) or **clean up**
(kernel gone).  An expired fence is necessary but never sufficient for a
kill: the kernel must also fail to present its proof.
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import asyncio
import os
import pathlib
import time
import uuid
from typing import Any, NewType, cast

KernelName = NewType("KernelName", str)
ModelName = NewType("ModelName", str)

try:
    import sqlite3
except ImportError:
    # fallback on pysqlite2 if Python was build without sqlite
    from pysqlite2 import dbapi2 as sqlite3  # type:ignore[no-redef]

from dataclasses import dataclass, fields

from jupyter_core.utils import ensure_async
from tornado import web
from tornado.ioloop import PeriodicCallback
from traitlets import Bool, Float, Instance, TraitError, Unicode, validate
from traitlets.config.configurable import LoggingConfigurable

from jupyter_server.traittypes import InstanceFromClasses

from .lease import (
    Lease,
    LeaseConflict,
    LeaseState,
    LeaseStore,
)


class KernelSessionRecordConflict(Exception):
    """项目内部接口说明。"""


@dataclass
class KernelSessionRecord:  # noqa: PLW1641 - TODO: implement __hash__
    """项目内部接口说明。"""

    session_id: str | None = None
    kernel_id: str | None = None

    def __eq__(self, other: object) -> bool:
        """项目内部接口说明。"""
        if isinstance(other, KernelSessionRecord):
            condition1 = self.kernel_id and self.kernel_id == other.kernel_id
            condition2 = all(
                [
                    self.session_id == other.session_id,
                    self.kernel_id is None or other.kernel_id is None,
                ]
            )
            if any([condition1, condition2]):
                return True
            # If two records share session_id but have different kernels, this is
            # and ill-posed expression. This should never be true. Raise an exception
            # to inform the user.
            if all(
                [
                    self.session_id,
                    self.session_id == other.session_id,
                    self.kernel_id != other.kernel_id,
                ]
            ):
                msg = (
                    "A single session_id can only have one kernel_id "
                    "associated with. These two KernelSessionRecords share the same "
                    "session_id but have different kernel_ids. This should "
                    "not be possible and is likely an issue with the session "
                    "records."
                )
                raise KernelSessionRecordConflict(msg)
        return False

    def update(self, other: "KernelSessionRecord") -> None:
        """项目内部接口说明。"""
        if not isinstance(other, KernelSessionRecord):
            msg = "'other' must be an instance of KernelSessionRecord."  # type:ignore[unreachable]
            raise TypeError(msg)

        if other.kernel_id and self.kernel_id and other.kernel_id != self.kernel_id:
            msg = "Could not update the record from 'other' because the two records conflict."
            raise KernelSessionRecordConflict(msg)

        for field in fields(self):
            if hasattr(other, field.name) and getattr(other, field.name):
                setattr(self, field.name, getattr(other, field.name))


class KernelSessionRecordList:
    """项目内部接口说明。"""

    _records: list[KernelSessionRecord]

    def __init__(self, *records: KernelSessionRecord):
        """项目内部接口说明。"""
        self._records = []
        for record in records:
            self.update(record)

    def __str__(self):
        """项目内部接口说明。"""
        return str(self._records)

    def __contains__(self, record: KernelSessionRecord | str) -> bool:
        """项目内部接口说明。"""
        if isinstance(record, KernelSessionRecord) and record in self._records:
            return True

        if isinstance(record, str):
            for r in self._records:
                if record in [r.session_id, r.kernel_id]:
                    return True
        return False

    def __len__(self):
        """项目内部接口说明。"""
        return len(self._records)

    def get(self, record: KernelSessionRecord | str) -> KernelSessionRecord:
        """项目内部接口说明。"""
        if isinstance(record, str):
            for r in self._records:
                if record in (r.kernel_id, r.session_id):
                    return r
        elif isinstance(record, KernelSessionRecord):
            for r in self._records:
                if record == r:
                    return record
        msg = f"{record} not found in KernelSessionRecordList."
        raise ValueError(msg)

    def update(self, record: KernelSessionRecord) -> None:
        """项目内部接口说明。"""
        try:
            idx = self._records.index(record)
            self._records[idx].update(record)
        except ValueError:
            self._records.append(record)

    def remove(self, record: KernelSessionRecord) -> None:
        """项目内部接口说明。"""
        if record in self._records:
            self._records.remove(record)


class SessionManager(LoggingConfigurable):
    """项目内部接口说明。"""

    database_filepath = Unicode(
        default_value=":memory:",
        help=(
            "The filesystem path to SQLite Database file "
            "(e.g. /path/to/session_database.db). By default, the session "
            "database is stored in-memory (i.e. `:memory:` setting from sqlite3) "
            "and does not persist when the current Jupyter Server shuts down."
        ),
    ).tag(config=True)

    @validate("database_filepath")
    def _validate_database_filepath(self, proposal):
        """项目内部接口说明。"""
        value = proposal["value"]
        if value == ":memory:":
            return value
        path = pathlib.Path(value)
        if path.exists():
            # Verify that the database path is not a directory.
            if path.is_dir():
                msg = "`database_filepath` expected a file path, but the given path is a directory."
                raise TraitError(msg)
            # Verify that database path is an SQLite 3 Database by checking its header.
            with open(value, "rb") as f:
                header = f.read(100)

            if not header.startswith(b"SQLite format 3") and header != b"":
                msg = "The given file is not an SQLite database file."
                raise TraitError(msg)
        return value

    kernel_manager = Instance("jupyter_server.services.kernels.kernelmanager.MappingKernelManager")
    contents_manager = InstanceFromClasses(
        [
            "jupyter_server.services.contents.manager.ContentsManager",
            "notebook.services.contents.manager.ContentsManager",
        ]
    )

    session_lease_ttl = Float(
        30.0,
        config=True,
        help="""Time in seconds after which an unrenewed session lease is
        considered stale and recoverable by another server process.

        A live server renews all leases it owns at roughly one third of
        this interval.  Expiry only makes a lease *reclaimable*: the
        reclaiming server still has to prove the kernel is gone (or take
        it over if it answers), so clock skew can never kill a kernel
        that still presents a valid identity.""",
    )

    session_lease_enabled = Bool(
        True,
        config=True,
        help="Enable generationed ownership leases and startup recovery.",
    )

    def __init__(self, *args, **kwargs):
        """项目内部接口说明。"""
        super().__init__(*args, **kwargs)
        self._pending_sessions = KernelSessionRecordList()
        self._leases: LeaseStore | None = None
        self._renewal_callback: PeriodicCallback | None = None
        self._reaper_callback: PeriodicCallback | None = None
        # Injectable clock for tests; production uses wall-clock epoch time.
        self._lease_clock = time.time
        self._recovery_done = False
        self._recovery_in_flight: Any = None
        # Wire kernel-side identity/restart notifications once the kernel
        # manager supports them.  The attribute checks keep dummy/manual
        # kernel managers (and the gateway manager) working unchanged.
        km = self.kernel_manager
        if hasattr(km, "_identity_changed_callback"):
            km._identity_changed_callback = self._on_kernel_identity_changed
            if hasattr(km, "set_default_lease_gate"):
                km.set_default_lease_gate(self._owns_kernel_now)

    # Session database initialized below
    _cursor = None
    _connection = None
    _columns = {"session_id", "path", "name", "type", "kernel_id"}

    @property
    def server_id(self) -> str:
        """Stable identity of this server process for lease ownership."""
        return self.lease_store.server_id

    @property
    def lease_store(self) -> LeaseStore:
        """Lazily create the lease store (and migrate a legacy database)."""
        if self._leases is None:
            connection = self.connection
            self._leases = LeaseStore(
                connection,
                ttl=self.session_lease_ttl,
                clock=self._lease_clock,
                log=self.log,
            )
        return self._leases

    @property
    def cursor(self):
        """Raw sqlite cursor, kept for backwards compatibility."""
        if self._cursor is None:
            # Touching the lease store ensures the schema/migration exists.
            self.lease_store  # noqa: B018
            self._cursor = self.connection.cursor()
        return self._cursor

    @property
    def connection(self):
        """项目内部接口说明。"""
        if self._connection is None:
            # check_same_thread=False lets the (rare) synchronous sqlite
            # call paths share the connection; all writes still serialise
            # through BEGIN IMMEDIATE transactions.
            self._connection = sqlite3.connect(
                self.database_filepath, isolation_level=None, check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self):
        """项目内部接口说明。"""
        for callback in (self._renewal_callback, self._reaper_callback):
            if callback is not None:
                callback.stop()
        self._renewal_callback = None
        self._reaper_callback = None
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None

    def __del__(self):
        """项目内部接口说明。"""
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # lease plumbing
    # ------------------------------------------------------------------
    def start_lease_maintenance(self) -> None:
        """Start periodic fence renewal and stale-record reaping."""
        if not self.session_lease_enabled:
            return
        interval = max(1.0, self.session_lease_ttl / 3.0) * 1000
        if self._renewal_callback is None:
            self._renewal_callback = PeriodicCallback(self._renew_leases, interval)
            self._renewal_callback.start()
        if self._reaper_callback is None:
            # Reap tombstones/dead claims at twice the TTL, never on the
            # renewal cadence: recycling must lag well behind fencing.
            self._reaper_callback = PeriodicCallback(
                self._reap_expired_records, max(2.0, self.session_lease_ttl * 2) * 1000
            )
            self._reaper_callback.start()

    def _owned_leases(self) -> list[Lease]:
        return [
            lease
            for lease in self.lease_store.list()
            if lease.owner_id == self.server_id
            and lease.state in (LeaseState.active, LeaseState.claimed)
        ]

    def _lease_for_kernel(self, kernel_id: str) -> Lease | None:
        for lease in self.lease_store.list():
            if lease.kernel_id == kernel_id and lease.state in (
                LeaseState.active,
                LeaseState.claimed,
            ):
                return lease
        return None

    def _renew_leases(self) -> None:
        """Push our ownership fences forward while the process is alive."""
        for lease in self._owned_leases():
            renewed = self.lease_store.renew(lease)
            if renewed is None:
                # Another generation took over: stop treating the kernel
                # (and its in-memory attachment) as ours.
                self.log.warning(
                    "Lost ownership of session %s while renewing its lease; "
                    "detaching its kernel.",
                    lease.session_id,
                )
                self._detach_kernel_if_adopted(lease.kernel_id)
        # Also re-check kernels we attached during recovery: the owner of
        # their record may have changed without a renewal attempt on our
        # part (the row simply no longer matches this server id).
        adopted = set(getattr(self.kernel_manager, "_kernel_adopted", set()))
        for kernel_id in adopted:
            current = self._lease_for_kernel(kernel_id)
            if current is not None and current.owner_id != self.server_id:
                self.log.warning(
                    "Kernel %s is now leased by %s; detaching.",
                    kernel_id,
                    current.owner_id,
                )
                self._detach_kernel_if_adopted(kernel_id)

    def _owns_kernel_now(self, kernel_id: str) -> bool:
        """Ownership predicate handed to the kernel manager's kill path.

        A locally-started kernel without any conflicting record is ours to
        manage.  As soon as a live (non-released) record names a different
        owner, the kernel is off-limits.
        """
        if not kernel_id:
            return True
        try:
            lease = self._lease_for_kernel(kernel_id)
        except Exception:
            return False
        if lease is None:
            return not self._is_adopted(kernel_id)
        return lease.owner_id == self.server_id

    def _is_adopted(self, kernel_id: str | None) -> bool:
        return bool(
            kernel_id
            and hasattr(self.kernel_manager, "_kernel_adopted")
            and kernel_id in self.kernel_manager._kernel_adopted
        )

    def _detach_kernel_if_adopted(self, kernel_id: str | None) -> None:
        if self._is_adopted(kernel_id) and hasattr(self.kernel_manager, "abandon_kernel"):
            self.kernel_manager.abandon_kernel(kernel_id)

    def _on_kernel_identity_changed(self, kernel_id: str, identity: str) -> None:
        """Record a new identity after a kernel restart rewrote its ports."""
        for lease in self._owned_leases():
            if lease.kernel_id == kernel_id:
                updated = self.lease_store.update_fields(lease, identity=identity)
                if updated is None:
                    self.log.warning(
                        "Cannot record identity change for kernel %s: lease lost.",
                        kernel_id,
                    )

    def _kernel_identity(self, kernel_id: str | None) -> str:
        if not kernel_id:
            return ""
        km = self.kernel_manager
        identities = getattr(km, "_kernel_identities", None)
        if isinstance(identities, dict) and identities.get(kernel_id):
            return str(identities[kernel_id])
        if hasattr(km, "read_kernel_identity"):
            return km.read_kernel_identity(kernel_id) or ""
        return ""

    # ------------------------------------------------------------------
    # basic session operations
    # ------------------------------------------------------------------
    async def ensure_recovered(self) -> dict[str, list[str]] | None:
        """Run startup recovery exactly once, even under concurrent calls."""
        if not self.session_lease_enabled or self._recovery_done:
            return None
        if self._recovery_in_flight is None:
            self._recovery_in_flight = self.recover_sessions()
        summary = await self._recovery_in_flight
        self._recovery_in_flight = None
        self._recovery_done = True
        return cast("dict[str, list[str]]", summary)

    async def session_exists(self, path):
        """项目内部接口说明。"""
        await self.ensure_recovered()
        exists = False
        row = self.connection.execute(
            "SELECT * FROM session WHERE path=? AND state IN ('active','claimed')", (path,)
        ).fetchone()
        if row is not None:
            # Note, although we found a row for the session, the associated kernel may have
            # been culled or died unexpectedly.  If that's the case, we should delete the
            # row, thereby terminating the session.  This can be done via a call to
            # row_to_model that tolerates that condition.  If row_to_model returns None,
            # we'll return false, since, at that point, the session doesn't exist anyway.
            model = await self.row_to_model(row, tolerate_culled=True)
            if model is not None:
                exists = True
        return exists

    def new_session_id(self) -> str:
        """项目内部接口说明。"""
        return str(uuid.uuid4())

    async def create_session(
        self,
        path: str | None = None,
        name: ModelName | None = None,
        type: str | None = None,
        kernel_name: KernelName | None = None,
        kernel_id: str | None = None,
    ) -> dict[str, Any]:
        """项目内部接口说明。"""
        await self.ensure_recovered()
        session_id = self.new_session_id()

        # Defend against a second session (and a second kernel) for the
        # same notebook before doing any work: another server may have
        # recovered it, or a concurrent request on this server may have
        # won the race.
        existing_row = self.connection.execute(
            "SELECT * FROM session WHERE path=? AND state IN ('active','claimed')",
            (path,),
        ).fetchone()
        if existing_row is not None:
            existing = Lease.from_row(existing_row)
            if existing.kernel_id and existing.kernel_id in self.kernel_manager:
                result = await self.get_session(session_id=existing.session_id)
                return cast("dict[str, Any]", result)
            if not self.lease_store.is_expired(existing):
                # Another server's fence is still live: creating a kernel
                # here would give the same notebook two owners.
                raise web.HTTPError(
                    409,
                    f"A session for {path!r} is active on another server "
                    f"(owner {existing.owner_id}); refusing to duplicate it.",
                )

        record = KernelSessionRecord(session_id=session_id)
        self._pending_sessions.update(record)
        if kernel_id is not None and kernel_id in self.kernel_manager:
            pass
        else:
            kernel_id = await self.start_kernel_for_session(
                session_id=session_id,
                path=path,
                name=name,
                type=type,
                kernel_name=kernel_name,
                kernel_id=kernel_id,
            )
        record.kernel_id = kernel_id
        self._pending_sessions.update(record)

        try:
            self.lease_store.insert(
                session_id=session_id,
                path=path,
                name=name,
                mtype=type,
                kernel_id=kernel_id,
                identity=self._kernel_identity(kernel_id),
            )
        except sqlite3.IntegrityError as e:
            raise web.HTTPError(409, f"Session already exists: {session_id}") from e
        self.start_lease_maintenance()
        self._pending_sessions.remove(record)
        result = await self.get_session(session_id=session_id)
        return cast("dict[str, Any]", result)

    def get_kernel_env(self, path: str | None, name: ModelName | None = None) -> dict[str, str]:
        """项目内部接口说明。"""
        if name is not None:
            cwd = self.kernel_manager.cwd_for_path(path)
            path = os.path.join(cwd, name)
        assert isinstance(path, str)
        return {**os.environ, "JPY_SESSION_NAME": path}

    async def start_kernel_for_session(
        self,
        session_id: str,
        path: str | None,
        name: ModelName | None,
        type: str | None,
        kernel_name: KernelName | None,
        kernel_id: str | None = None,
    ) -> str:
        """项目内部接口说明。"""
        # allow contents manager to specify kernels cwd
        kernel_path = await ensure_async(self.contents_manager.get_kernel_path(path=path))

        kernel_env = self.get_kernel_env(path, name)
        kernel_id = await self.kernel_manager.start_kernel(
            path=kernel_path,
            kernel_name=kernel_name,
            env=kernel_env,
            kernel_id=kernel_id,
        )
        return cast("str", kernel_id)

    async def save_session(self, session_id, path=None, name=None, type=None, kernel_id=None):
        """Persist a session row under a fresh lease owned by us."""
        self.lease_store.insert(
            session_id=session_id,
            path=path,
            name=name,
            mtype=type,
            kernel_id=kernel_id,
            identity=self._kernel_identity(kernel_id),
        )
        self.start_lease_maintenance()
        return await self.get_session(session_id=session_id)

    def _visible_row(self, row) -> bool:
        lease = Lease.from_row(row)
        if lease.state in (LeaseState.quarantined, LeaseState.released):
            return False
        if lease.owner_id == self.server_id:
            return True
        if lease.state == LeaseState.claimed:
            return False
        # A foreign, active record is visible only when this process is
        # actually serving its kernel (e.g. the in-process manager
        # survived a SessionManager re-creation); otherwise a live peer
        # owns it.
        return bool(lease.kernel_id and lease.kernel_id in self.kernel_manager)

    async def get_session(self, **kwargs):
        """项目内部接口说明。"""
        await self.ensure_recovered()
        if not kwargs:
            msg = "must specify a column to query"
            raise TypeError(msg)

        conditions = []
        for column in kwargs:
            if column not in self._columns:
                msg = f"No such column: {column}"
                raise TypeError(msg)
            conditions.append("%s=?" % column)

        query = (
            "SELECT * FROM session WHERE %s AND state IN ('active','claimed')"
            % (" AND ".join(conditions))
        )  # noqa: S608

        self.cursor.execute(query, list(kwargs.values()))
        try:
            row = self.cursor.fetchone()
        except KeyError:
            # The kernel is missing, so the session just got deleted.
            row = None

        if row is None or not self._visible_row(row):
            q = []
            for key, value in kwargs.items():
                q.append(f"{key}={value!r}")

            raise web.HTTPError(404, "Session not found: %s" % (", ".join(q)))

        try:
            model = await self.row_to_model(row)
        except KeyError as e:
            raise web.HTTPError(404, "Session not found: %s" % str(e)) from e
        return model

    async def update_session(self, session_id, **kwargs):
        """项目内部接口说明。"""
        lease = self.lease_store.get(session_id)
        if lease is None:
            raise web.HTTPError(404, f"Session not found: {session_id!r}")
        if lease.owner_id != self.server_id or lease.state == LeaseState.released:
            raise web.HTTPError(409, f"Session {session_id} is owned by another server")

        if not kwargs:
            # no changes
            return

        for column in kwargs:
            if column not in self._columns:
                raise TypeError("No such column: %r" % column)

        unset = LeaseStore._UNSET
        new_kernel = kwargs.get("kernel_id", unset)
        updated = self.lease_store.update_fields(
            lease,
            kernel_id=new_kernel,
            identity=self._kernel_identity(new_kernel) if new_kernel is not unset else unset,
            path=kwargs.get("path", unset),
            name=kwargs.get("name", unset),
            mtype=kwargs.get("type", unset),
        )
        if updated is None:
            raise web.HTTPError(
                409, f"Session {session_id} changed ownership during update"
            )

        if "path" in kwargs or "name" in kwargs:
            path = updated.path
            name = updated.name
            if hasattr(self.kernel_manager, "update_env"):
                self.kernel_manager.update_env(
                    kernel_id=updated.kernel_id,
                    env=self.get_kernel_env(path, cast("ModelName | None", name)),
                )

    async def kernel_culled(self, kernel_id: str) -> bool:
        """项目内部接口说明。"""
        return kernel_id not in self.kernel_manager

    async def row_to_model(self, row, tolerate_culled=False):
        """项目内部接口说明。"""
        lease = Lease.from_row(row)
        kernel_culled: bool = await ensure_async(self.kernel_culled(row["kernel_id"]))
        if kernel_culled:
            # The kernel was culled or died without deleting the session.
            msg = (
                "Kernel '{kernel_id}' appears to have been culled or died unexpectedly, "
                "invalidating session '{session_id}'.".format(
                    kernel_id=row["kernel_id"], session_id=row["session_id"]
                )
            )
            if lease.owner_id == self.server_id:
                # We own this record: the in-memory kernel is gone, so the
                # session really is dead -- remove it outright.  If we do
                # not own it, another server may still be serving the same
                # kernel; merely hide it from our listings.
                self.lease_store.erase(row["session_id"])
                full = msg + " The session has been removed."
                if tolerate_culled:
                    self.log.warning(f"{full}  Continuing...")
                    return None
                raise KeyError(full)
            self.log.debug("%s  Owned by another server; hiding without deleting.", msg)
            return None

        kernel_model = await ensure_async(self.kernel_manager.kernel_model(row["kernel_id"]))
        model = {
            "id": row["session_id"],
            "path": row["path"],
            "name": row["name"],
            "type": row["type"],
            "kernel": kernel_model,
        }
        if row["type"] == "notebook":
            # Provide the deprecated API.
            model["notebook"] = {"path": row["path"], "name": row["name"]}
        return model

    async def list_sessions(self):
        """项目内部接口说明。"""
        await self.ensure_recovered()
        c = self.cursor.execute(
            "SELECT * FROM session WHERE state IN ('active','claimed')"
        )
        result = []
        # We need to use fetchall() here, because row_to_model can delete rows,
        # which messes up the cursor if we're iterating over rows.
        for row in c.fetchall():
            if not self._visible_row(row):
                continue
            try:
                model = await self.row_to_model(row)
                if model is not None:
                    result.append(model)
            except KeyError:
                pass
        return result

    def list_quarantined_sessions(self) -> list[Lease]:
        """Records parked due to record/kernel disagreement (inspection)."""
        return self.lease_store.list(LeaseState.quarantined)

    async def delete_session(self, session_id):
        """项目内部接口说明。"""
        record = KernelSessionRecord(session_id=session_id)
        self._pending_sessions.update(record)
        try:
            lease = self.lease_store.get(session_id)
            if lease is None or lease.state == LeaseState.released:
                raise web.HTTPError(404, f"Session not found: {session_id!r}")
            if lease.owner_id != self.server_id:
                raise web.HTTPError(
                    409,
                    f"Session {session_id} is owned by another server generation; "
                    "not shutting down its kernel.",
                )

            # Push our fence forward so a peer cannot win a claim in the
            # window between the ownership check and the actual kill.
            renewed = self.lease_store.renew(lease)
            if renewed is None:
                raise web.HTTPError(
                    409,
                    f"Session {session_id} changed ownership before deletion; "
                    "not shutting down its kernel.",
                )
            lease = renewed

            kernel_id = lease.kernel_id
            kernel_present = kernel_id is not None and kernel_id in self.kernel_manager
            if kernel_present:
                try:
                    await ensure_async(self.kernel_manager.shutdown_kernel(kernel_id))
                except Exception as err:
                    # LeaseOwnershipLost (or a gate failure from a subclass)
                    # means another generation won between our decision and
                    # the kill: leave everything alone.
                    name = type(err).__name__
                    if name == "LeaseOwnershipLost":
                        self.log.error(
                            "Aborting delete of session %s: %s", session_id, err
                        )
                        raise web.HTTPError(409, str(err)) from err
                    raise

            # The kernel is gone (or was never started for this row).  The
            # conditional release is the second fence: even if the shutdown
            # above raced a take-over, this write fails rather than wiping
            # the new owner's record.
            try:
                self.lease_store.release(lease)
            except LeaseConflict as err:
                self.log.error("Aborting record deletion for %s: %s", session_id, err)
                raise web.HTTPError(409, str(err)) from err
            self.lease_store.erase(session_id)
        finally:
            self._pending_sessions.remove(record)

    # ------------------------------------------------------------------
    # connection authorization (websocket reconnect path)
    # ------------------------------------------------------------------
    async def authorize_kernel_connection(
        self, kernel_id: str, client_session_id: str | None = None
    ) -> bool:
        """Decide whether a websocket may attach to ``kernel_id``.

        Returns False when the kernel is covered by a lease owned by a
        different server generation -- that kernel is either still served
        by a live peer or has not been taken over yet, and attaching our
        own channels could interleave with the peer.  A client in that
        situation should simply retry; once the fence expires and recovery
        completes the kernel will be owned by this server.
        """
        if kernel_id not in self.kernel_manager:
            return False
        lease = self._lease_for_kernel(kernel_id)
        if lease is None:
            # Kernel we started ourselves but no longer referenced by a
            # session: backwards-compatible attach (e.g. debug sessions).
            return True
        return lease.owner_id == self.server_id

    # ------------------------------------------------------------------
    # startup recovery
    # ------------------------------------------------------------------
    async def recover_sessions(self) -> dict[str, list[str]]:
        """Reconcile the persisted database with live kernels.

        Runs once at server startup, before traffic is served.  For each
        stale record it compares the three sources of truth (record,
        visible kernel, kernel proof) and chooses one of:

        *take over* -- fence expired, kernel answers, identity matches;
        *quarantine* -- record and answering kernel disagree;
        *clean up*  -- the kernel cannot present any proof.

        Returns a summary dict with the session ids in each bucket.
        """
        summary: dict[str, list[str]] = {
            "adopted": [],
            "quarantined": [],
            "cleaned": [],
            "skipped": [],
        }
        if not self.session_lease_enabled:
            return summary

        store = self.lease_store
        for lease in list(store.list()):
            try:
                action = await self._recover_one(lease)
            except LeaseConflict as err:
                # Another server won the compare-and-swap concurrently.
                self.log.info("Skipping %s during recovery: %s", lease.session_id, err)
                summary["skipped"].append(lease.session_id)
                continue
            except Exception:
                self.log.exception(
                    "Error recovering session %s; leaving its record untouched.",
                    lease.session_id,
                )
                summary["skipped"].append(lease.session_id)
                continue
            summary[action].append(lease.session_id)

        self.start_lease_maintenance()
        return summary

    async def _recover_one(self, lease: Lease) -> str:
        """Recover a single lease; returns adopted|quarantined|cleaned|skipped."""
        store = self.lease_store
        kernel_id = lease.kernel_id

        # Released tombstones and half-created rows without a kernel are
        # always safe garbage.
        if lease.state == LeaseState.released or kernel_id is None:
            store.erase(lease.session_id)
            return "cleaned"

        # The fence of a live owner has not elapsed.  This includes a
        # record freshly renewed by a *concurrently starting* peer: we do
        # not touch the kernel no matter what our local view says.
        if not store.is_expired(lease) and lease.owner_id != self.server_id:
            return "skipped"

        km = self.kernel_manager
        read_identity = getattr(km, "read_kernel_identity", None)
        probe = getattr(km, "probe_kernel", None)

        identity_on_disk = read_identity(kernel_id) if read_identity else None
        if identity_on_disk is None:
            if read_identity is not None:
                # A real kernel manager reports no connection file: the
                # kernel cannot be the recorded one and no proof is
                # possible.  Claim then erase -- but never kill.
                store.claim(lease)
                store.erase(lease.session_id)
                self.log.info(
                    "Cleaned session %s: kernel %s left no connection file.",
                    lease.session_id,
                    kernel_id,
                )
                return "cleaned"
            # Kernel manager without identity support (tests, minimal
            # custom managers): the in-memory map is the only proof.
            if kernel_id not in km:
                store.claim(lease)
                store.erase(lease.session_id)
                return "cleaned"

        # If this process already holds the kernel (shared/persistent
        # kernel manager), that attachment is itself the proof; no probe
        # or second manager is required.
        locally_held = kernel_id in km

        if locally_held:
            live_identity = self._kernel_identity(kernel_id) or identity_on_disk or lease.identity
            alive = True
        else:
            live_identity = identity_on_disk
            if probe is not None:
                # Two probes with a short gap: a single lost ping must not
                # be enough to condemn a kernel.
                alive = bool(await probe(kernel_id))
                if not alive:
                    await asyncio.sleep(0.25)
                    alive = bool(await probe(kernel_id))
            else:
                alive = False

        identity_matches = (
            identity_on_disk is None  # cannot verify on minimal managers
            or not lease.identity
            or lease.identity == live_identity
        )

        if alive and identity_matches:
            # Orphaned but reachable and recognisable: take it over.
            claimed = store.claim(lease)
            try:
                adopted = True
                if not locally_held and hasattr(km, "adopt_kernel"):
                    adopted = await km.adopt_kernel(kernel_id)
                elif not locally_held:
                    adopted = False
                if not adopted:
                    # Mid-takeover failure: park the record instead of
                    # leaving a dangling claim and never touch the kernel.
                    store.quarantine(claimed, identity=live_identity)
                    self._detach_kernel_if_adopted(kernel_id)
                    self.log.warning(
                        "Quarantined session %s: kernel %s could not be attached.",
                        lease.session_id,
                        kernel_id,
                    )
                    return "quarantined"
                store.activate(claimed, identity=live_identity)
            except Exception:
                # Roll the claim into quarantine so a crash here cannot
                # produce two acting owners.
                try:
                    store.quarantine(claimed, identity=live_identity)
                except LeaseConflict:
                    pass
                self._detach_kernel_if_adopted(kernel_id)
                raise
            self.log.info(
                "Took over session %s with live kernel %s.",
                lease.session_id,
                kernel_id,
            )
            return "adopted"

        if alive and not identity_matches:
            # Something is answering on these ports, but it is not the
            # kernel the record describes (recycled ports, different key).
            # Quarantine: we neither adopt the stranger nor kill it.
            claimed = store.claim(lease)
            store.quarantine(claimed, identity=identity_on_disk)
            self.log.warning(
                "Quarantined session %s: live kernel %s has a different "
                "identity than the persisted record.",
                lease.session_id,
                kernel_id,
            )
            return "quarantined"

        # Connection file present, but no heartbeat: the kernel is dead.
        # Claim and remove only the bookkeeping (and our stale connection
        # file).  No kill is issued against anything alive.
        claimed = store.claim(lease)
        try:
            if kernel_id in km:
                # Defensive: the in-memory manager may still hold it.
                await ensure_async(km.shutdown_kernel(kernel_id))
        except Exception:
            pass
        connection_file = os.path.join(
            getattr(km, "connection_dir", ""), f"kernel-{kernel_id}.json"
        )
        try:
            if connection_file and os.path.exists(connection_file):
                os.remove(connection_file)
        except OSError:
            self.log.debug("Could not remove stale connection file %s", connection_file)
        store.erase(claimed.session_id)
        self.log.info(
            "Cleaned session %s: kernel %s did not present a valid proof.",
            lease.session_id,
            kernel_id,
        )
        return "cleaned"

    async def _reap_expired_records(self) -> None:
        """Periodic garbage collection that never kills on wall time alone.

        * released rows are erased;
        * our own stale ``claimed`` rows with no proof behind them are
          erased;
        * quarantined rows are erased only once the kernel proof has
          disappeared.  A quarantined kernel that still answers is left
          isolated forever -- expiry alone never condemns it.
        """
        store = self.lease_store
        for lease in list(store.list()):
            kernel_id = lease.kernel_id
            if lease.state == LeaseState.released:
                store.erase(lease.session_id)
                continue
            if lease.state == LeaseState.claimed and lease.owner_id == self.server_id:
                # A claim we still hold past the fence means a recovery
                # step got interrupted; if there is no proof, collect it,
                # otherwise let a fresh recovery/peer sort it out.
                proof = await self._kernel_presents_proof(kernel_id)
                if not proof:
                    store.erase(lease.session_id)
                continue
            if lease.state == LeaseState.quarantined and store.is_expired(lease):
                proof = await self._kernel_presents_proof(kernel_id)
                if not proof:
                    store.erase(lease.session_id)

    async def _kernel_presents_proof(self, kernel_id: str | None) -> bool:
        """True iff the kernel can still answer from its named identity."""
        if not kernel_id:
            return False
        km = self.kernel_manager
        if hasattr(km, "read_kernel_identity") and km.read_kernel_identity(kernel_id) is None:
            return False
        if hasattr(km, "probe_kernel"):
            return bool(await km.probe_kernel(kernel_id))
        return kernel_id in km
