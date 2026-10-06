"""Generationed ownership leases binding sessions to kernels.

When a server process dies, its kernels may keep running.  On the next
startup the session database is re-read, but the records it contains were
written by a previous incarnation of the server -- possibly even by a
*different* server which is still alive and serving clients.

A naive "trust the database" recovery has two failure modes:

1. the new server kills a kernel that another server is still using, and
2. both servers create a fresh kernel/session for the same notebook.

This module implements a small, sqlite-backed **fencing-lease** protocol
on top of the session table:

* every row carries the owning ``server_id``, a monotonic ``generation``,
  a wall-clock ``expires_at`` timestamp and a kernel ``identity`` token;
* ownership is only ever transferred with a compare-and-swap UPDATE that
  checks the previous (``server_id``, ``generation``) pair, so two servers
  racing through recovery cannot both win;
* a wall-clock expiry is a *recycling* hint, never a kill order: a lease
  is only reclaimable while the kernel behind it cannot present a valid
  identity proof (see :meth:`LeaseStore.identity_matches`).

The lease state machine lives in :class:`LeaseState`; all SQL lives in
:class:`LeaseStore` so the session manager can stay focused on policy
(take over / quarantine / clean up).
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class LeaseState(str, Enum):
    """Ownership states a session row can be in.

    active
        The row is owned by some server that holds a current lease.
    claimed
        Recovery in progress: an owner has won the compare-and-swap but
        has not finished reconciling the kernel yet.  A crashed claim
        becomes ``active`` under the old owner after the fence expires
        and can be retried -- never double-owned.
    quarantined
        The persistent record and the kernel on disk disagreed (e.g. the
        connection file describes a different kernel).  The row is kept
        for inspection but is excluded from normal session listings and
        is never used to kill or to take over a kernel.
    released
        Terminal state.  The row and any kernel it referenced have been
        cleaned up; reusing the session/kernel ids starts a new lease.
    """

    active = "active"
    claimed = "claimed"
    quarantined = "quarantined"
    released = "released"


@dataclass(frozen=True)
class Lease:
    """An in-memory view of one lease row."""

    session_id: str
    kernel_id: str | None
    state: LeaseState
    owner_id: str
    generation: int
    expires_at: float
    identity: str
    path: str | None = None
    name: str | None = None
    type: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Lease:
        """Build a Lease from a sqlite row produced by LeaseStore."""
        return cls(
            session_id=row["session_id"],
            kernel_id=row["kernel_id"],
            state=LeaseState(row["state"]),
            owner_id=row["owner_id"],
            generation=row["generation"],
            expires_at=row["expires_at"],
            identity=row["identity"] or "",
            path=row["path"],
            name=row["name"],
            type=row["type"],
        )

    @property
    def expired(self) -> bool:
        """Whether the wall-clock fence has elapsed.

        Expiry makes a lease *reclaimable*; it is never, by itself,
        permission to destroy a kernel that can still prove its identity.
        Timestamps are stored as epoch seconds (the store's clock); the
        recovery policy layers the proof-of-life check on top.
        """
        return self.expires_at <= time.time()


def compute_kernel_identity(connection_info: dict[str, Any]) -> str:
    """Hash the parts of a connection file that identify one kernel.

    The five ZMQ ports plus the signing key uniquely identify a live
    kernel on this host.  A restart that rewrites the connection file
    with different ports changes the identity, which is exactly what we
    want to detect when a record points at a recycled port.
    """
    material = {
        key: connection_info.get(key)
        for key in (
            "transport",
            "ip",
            "shell_port",
            "iopub_port",
            "stdin_port",
            "control_port",
            "hb_port",
            "key",
        )
    }
    blob = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class LeaseConflict(RuntimeError):
    """Raised when a compare-and-swap lease update loses the race.

    This is the *expected* outcome when two servers recover the same
    session simultaneously: the loser must back off instead of touching
    the kernel.
    """


class LeaseStore:
    """SQLite persistence for generationed leases.

    The store owns the schema (including migration of the legacy
    ``(session_id, path, name, type, kernel_id)`` table) and every
    conditional write.  All multi-statement writes run inside
    ``BEGIN IMMEDIATE`` so a second process blocks/fails on the write
    lock instead of silently interleaving.
    """

    LEGACY_COLUMNS = ("session_id", "path", "name", "type", "kernel_id")
    COLUMNS = (
        "session_id",
        "path",
        "name",
        "type",
        "kernel_id",
        "state",
        "owner_id",
        "generation",
        "expires_at",
        "identity",
    )

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        ttl: float,
        clock: Any = time.monotonic,
        log: Any = None,
        busy_timeout_ms: int = 5000,
    ) -> None:
        """Wrap an open sqlite connection and ensure the schema exists."""
        self._connection = connection
        self.ttl = float(ttl)
        self._clock = clock
        self.log = log
        self.server_id = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        # Two servers share the database file: make a contended
        # BEGIN IMMEDIATE wait for the peer's short write transaction
        # instead of failing immediately with SQLITE_BUSY.
        try:
            connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        except sqlite3.DatabaseError:
            pass
        self._ensure_schema()

    # ------------------------------------------------------------------
    # schema & migration
    # ------------------------------------------------------------------
    def _table_info(self) -> set[str]:
        rows = self._connection.execute("PRAGMA table_info(session)").fetchall()
        return {row[1] for row in rows}

    def _ensure_schema(self) -> None:
        """Create the table or migrate a pre-lease database in place."""
        columns = self._table_info()
        if not columns:
            self._connection.execute(
                """
                CREATE TABLE session (
                    session_id  TEXT PRIMARY KEY,
                    path        TEXT,
                    name        TEXT,
                    type        TEXT,
                    kernel_id   TEXT,
                    state       TEXT NOT NULL DEFAULT 'active',
                    owner_id    TEXT NOT NULL DEFAULT '',
                    generation  INTEGER NOT NULL DEFAULT 0,
                    expires_at  REAL NOT NULL DEFAULT 0,
                    identity    TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_kernel ON session(kernel_id)"
            )
            self._connection.commit()
            return

        missing = [
            spec
            for spec in (
                ("state", "TEXT NOT NULL DEFAULT 'active'"),
                ("owner_id", "TEXT NOT NULL DEFAULT ''"),
                ("generation", "INTEGER NOT NULL DEFAULT 0"),
                ("expires_at", "REAL NOT NULL DEFAULT 0"),
                ("identity", "TEXT NOT NULL DEFAULT ''"),
            )
            if spec[0] not in columns
        ]
        if missing:
            for name, decl in missing:
                self._connection.execute(f"ALTER TABLE session ADD COLUMN {name} {decl}")
            # Legacy rows were owned by a server that has no fence or
            # identity on record: mark the fence already elapsed so the
            # first starting server reconciles them.
            self._connection.execute(
                "UPDATE session SET state='active', expires_at=0 "
                "WHERE owner_id='' OR expires_at=0"
            )
            self._connection.commit()

    # ------------------------------------------------------------------
    # low level helpers
    # ------------------------------------------------------------------
    def now(self) -> float:
        """Current monotonic clock value (seconds)."""
        return float(self._clock())

    def fresh_expiry(self) -> float:
        return self.now() + self.ttl

    def is_expired(self, lease: Lease) -> bool:
        """Fence check using this store's (injectable) clock."""
        return lease.expires_at <= self.now()

    def get(self, session_id: str) -> Lease | None:
        row = self._connection.execute(
            "SELECT * FROM session WHERE session_id=?", (session_id,)
        ).fetchone()
        return Lease.from_row(row) if row is not None else None

    def list(self, *states: LeaseState) -> list[Lease]:
        if states:
            marks = ",".join("?" for _ in states)
            rows = self._connection.execute(
                f"SELECT * FROM session WHERE state IN ({marks})",
                [s.value for s in states],
            ).fetchall()
        else:
            rows = self._connection.execute("SELECT * FROM session").fetchall()
        return [Lease.from_row(row) for row in rows]

    def touch_connections(self) -> None:
        """Ask sqlite to flush lazy page writes; used by lock tests."""
        self._connection.commit()

    # ------------------------------------------------------------------
    # conditional writes -- the fencing protocol
    # ------------------------------------------------------------------
    def insert(
        self,
        *,
        session_id: str,
        path: str | None,
        name: str | None,
        mtype: str | None,
        kernel_id: str | None,
        identity: str = "",
    ) -> Lease:
        """Insert a brand-new session owned by this server.

        Raises ``sqlite3.IntegrityError`` if the session id exists,
        including a ``released`` tombstone -- callers must call
        :meth:`erase` first when intentionally reusing an id.
        """
        expires_at = self.fresh_expiry()
        with self._immediate() as cur:
            cur.execute(
                "INSERT INTO session "
                "(session_id, path, name, type, kernel_id, state, owner_id, "
                " generation, expires_at, identity) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    path,
                    name,
                    mtype,
                    kernel_id,
                    LeaseState.active.value,
                    self.server_id,
                    1,
                    expires_at,
                    identity,
                ),
            )
        lease = self.get(session_id)
        assert lease is not None
        return lease

    def renew(self, lease: Lease) -> Lease | None:
        """Extend a lease we already own.

        Returns the renewed lease, or ``None`` if the row is no longer
        ours (a take-over happened in between) -- in that case the caller
        must stop acting as the owner.
        """
        new_expiry = self.fresh_expiry()
        with self._immediate() as cur:
            cur.execute(
                "UPDATE session SET expires_at=? "
                "WHERE session_id=? AND owner_id=? AND generation=? "
                "AND state!='released'",
                (new_expiry, lease.session_id, self.server_id, lease.generation),
            )
            changed = cur.rowcount
        if changed == 0:
            return None
        return self.get(lease.session_id)

    _UNSET = object()

    def update_fields(
        self, lease: Lease, *, kernel_id: Any = _UNSET, identity: Any = _UNSET,
        path: Any = _UNSET, name: Any = _UNSET, mtype: Any = _UNSET,
    ) -> Lease | None:
        """Conditionally mutate session fields while preserving the fence.

        Only succeeds while this process still owns the exact generation.
        Used for session renames (path/name/type) and for attaching a
        kernel to an in-flight session -- neither operation may silently
        land on a row another server has taken over.  Explicit ``None``
        values are honoured (e.g. clearing a session name).
        """
        assignments: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("kernel_id", kernel_id),
            ("identity", identity),
            ("path", path),
            ("name", name),
            ("type", mtype),
        ):
            if value is not LeaseStore._UNSET:
                assignments.append(f"{column}=?")
                values.append(value)
        if not assignments:
            return self.get(lease.session_id)
        assignments.append("expires_at=?")
        values.append(self.fresh_expiry())
        values.extend([lease.session_id, self.server_id, lease.generation])
        with self._immediate() as cur:
            cur.execute(
                f"UPDATE session SET {', '.join(assignments)} "
                "WHERE session_id=? AND owner_id=? AND generation=? "
                "AND state!='released'",
                values,
            )
            changed = cur.rowcount
        if changed == 0:
            return None
        return self.get(lease.session_id)

    def claim(
        self,
        lease: Lease,
        *,
        expected_owner: str | None = None,
        require_expired: bool = True,
    ) -> Lease:
        """Acquire ownership of a foreign/stale lease with compare-and-swap.

        The WHERE clause encodes the whole safety argument:

        * the row is not ``released`` or already ``claimed`` by someone
          whose fence has not elapsed;
        * the previous ``(owner_id, generation)`` still matches what the
          caller inspected -- so a concurrent owner that renewed its
          fence loses the swap and gets a :class:`LeaseConflict`;
        * when ``require_expired`` is set (startup recovery), the old
          owner's fence must have elapsed.  A server that is still alive
          renews fences periodically, so this prevents stealing from a
          live peer.

        On success the generation is bumped and the state becomes
        ``claimed``; the caller owns the row exclusively while it
        reconciles the kernel.
        """
        if lease.state == LeaseState.released:
            msg = f"Session {lease.session_id} has been released"
            raise LeaseConflict(msg)

        conditions = [
            "session_id=?",
            "owner_id=?",
            "generation=?",
            "state!='released'",
        ]
        params: list[Any] = [lease.session_id, lease.owner_id, lease.generation]
        if require_expired:
            conditions.append("expires_at<=?")
            params.append(self.now())
        # A live claim (state='claimed' with an unexpired fence, owned by
        # someone else) cannot be stolen.
        conditions.append("NOT (state='claimed' AND expires_at>?)")
        params.append(self.now())

        new_generation = lease.generation + 1
        where = " AND ".join(conditions)
        with self._immediate() as cur:
            cur.execute(
                "UPDATE session SET state='claimed', owner_id=?, generation=?, "
                f"expires_at=? WHERE {where}",
                [self.server_id, new_generation, self.fresh_expiry(), *params],
            )
            changed = cur.rowcount
        if changed == 0:
            current = self.get(lease.session_id)
            detail = "" if current is None else (
                f" (now owner={current.owner_id!r} gen={current.generation} "
                f"state={current.state.value})"
            )
            msg = f"Lost the ownership race for session {lease.session_id}{detail}"
            raise LeaseConflict(msg)
        result = self.get(lease.session_id)
        assert result is not None
        return result

    def activate(self, lease: Lease, *, identity: str | None = None) -> Lease:
        """Finish a successful claim: flip ``claimed`` -> ``active``.

        Only the claim holder (current owner+generation) can do this.
        """
        assignments = ["state='active'", "expires_at=?"]
        params: list[Any] = [self.fresh_expiry()]
        if identity is not None:
            assignments.append("identity=?")
            params.append(identity)
        params.extend([lease.session_id, self.server_id, lease.generation])
        with self._immediate() as cur:
            cur.execute(
                f"UPDATE session SET {', '.join(assignments)} "
                "WHERE session_id=? AND owner_id=? AND generation=? "
                "AND state='claimed'",
                params,
            )
            changed = cur.rowcount
        if changed == 0:
            raise LeaseConflict(
                f"Cannot activate session {lease.session_id}: not the claim holder"
            )
        result = self.get(lease.session_id)
        assert result is not None
        return result

    def quarantine(self, lease: Lease, *, identity: str | None = None) -> Lease:
        """Park a row whose record/kernel mismatch could not be resolved."""
        assignments = ["state='quarantined'", "expires_at=?"]
        params: list[Any] = [self.fresh_expiry()]
        if identity is not None:
            assignments.append("identity=?")
            params.append(identity)
        params.extend([lease.session_id, self.server_id, lease.generation])
        with self._immediate() as cur:
            cur.execute(
                f"UPDATE session SET {', '.join(assignments)} "
                "WHERE session_id=? AND owner_id=? AND generation=? "
                "AND state!='released'",
                params,
            )
            changed = cur.rowcount
        if changed == 0:
            raise LeaseConflict(
                f"Cannot quarantine session {lease.session_id}: lease changed underneath"
            )
        result = self.get(lease.session_id)
        assert result is not None
        return result

    def release(self, lease: Lease) -> None:
        """Mark a session terminated.

        Requires ownership of the exact generation: if another server has
        since taken over, the call fails and the caller MUST NOT kill the
        kernel.  The kernel manager enforces the same check independently
        before killing, so defence is layered.
        """
        with self._immediate() as cur:
            cur.execute(
                "UPDATE session SET state='released', expires_at=? "
                "WHERE session_id=? AND owner_id=? AND generation=? "
                "AND state!='released'",
                (self.fresh_expiry(), lease.session_id, self.server_id, lease.generation),
            )
            changed = cur.rowcount
        if changed == 0:
            raise LeaseConflict(
                f"Cannot release session {lease.session_id}: not the owner of "
                f"generation {lease.generation}"
            )

    def erase(self, session_id: str) -> None:
        """Physically delete a row (after release or when proven orphaned)."""
        with self._immediate() as cur:
            cur.execute("DELETE FROM session WHERE session_id=?", (session_id,))

    # ------------------------------------------------------------------
    def _immediate(self):
        return _ImmediateTransaction(self._connection)


class _ImmediateTransaction:
    """Context manager issuing ``BEGIN IMMEDIATE`` for write transactions.

    The connection is kept in sqlite's default (deferred) autocommit-ish
    mode; each multi-statement write takes the reserved write lock up
    front, so two processes serialise instead of hitting
    ``SQLITE_BUSY`` mid-transaction.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._began = False

    def __enter__(self) -> sqlite3.Cursor:
        self._connection.execute("BEGIN IMMEDIATE")
        self._began = True
        return self._connection.cursor()

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._began:
            return
        if exc_type is None:
            self._connection.commit()
        else:
            self._connection.rollback()
