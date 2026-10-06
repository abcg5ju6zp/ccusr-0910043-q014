"""Regression tests for generationed session/kernel ownership leases."""

import asyncio
import json
import os
import pathlib
import sqlite3

import pytest
from tornado import web

from jupyter_server.services.contents.manager import ContentsManager
from jupyter_server.services.kernels.kernelmanager import MappingKernelManager
from jupyter_server.services.sessions.lease import (
    LeaseConflict,
    LeaseState,
    LeaseStore,
    compute_kernel_identity,
)
from jupyter_server.services.sessions.sessionmanager import SessionManager


# ---------------------------------------------------------------------------
# Fake kernel world
# ---------------------------------------------------------------------------
class FakeKernel:
    def __init__(self, kernel_name="python", info=None):
        self.kernel_name = kernel_name
        self.execution_state = "idle"
        self.info = info or {}
        self.shutdown_requested = False


class FakeKernelManager(MappingKernelManager):
    """In-memory stand-in for MappingKernelManager with lease hooks."""

    def __init__(self, connection_dir, live_kernels=None, fail_adopt=()):
        super().__init__()
        self.connection_dir = str(connection_dir)
        # Shared, externally-mutable set of kernels answering heartbeats.
        self.live_kernels = live_kernels if live_kernels is not None else set()
        self.fail_adopt = set(fail_adopt)
        self._kernels: dict[str, FakeKernel] = {}
        self._kernel_connections: dict[str, int] = {}
        self._kernel_identities: dict[str, str] = {}
        self._kernel_adopted: set[str] = set()
        self._kernel_lease_gates: dict = {}
        self._default_lease_gate = None
        self._identity_changed_callback = None
        self.shutdown_calls: list[str] = []

    def __contains__(self, kernel_id):
        return kernel_id in self._kernels

    # MappingKernelManager-like surface --------------------------------
    def set_default_lease_gate(self, gate):
        self._default_lease_gate = gate

    def set_lease_gate(self, kernel_id, gate):
        self._kernel_lease_gates[kernel_id] = gate

    def _connection_file_path(self, kernel_id):
        return os.path.join(self.connection_dir, f"kernel-{kernel_id}.json")

    def _write_connection_file(self, kernel_id, *, ports=None, key="abc123", kernel_name="python"):
        info = {
            "transport": "tcp",
            "ip": "127.0.0.1",
            "shell_port": (ports or (5001, 5002, 5003, 5004, 5005))[0],
            "iopub_port": (ports or (5001, 5002, 5003, 5004, 5005))[1],
            "stdin_port": (ports or (5001, 5002, 5003, 5004, 5005))[2],
            "control_port": (ports or (5001, 5002, 5003, 5004, 5005))[3],
            "hb_port": (ports or (5001, 5002, 5003, 5004, 5005))[4],
            "key": key,
            "signature_scheme": "hmac-sha256",
            "kernel_name": kernel_name,
        }
        os.makedirs(self.connection_dir, exist_ok=True)
        with open(self._connection_file_path(kernel_id), "w") as f:
            json.dump(info, f)
        return info

    def read_kernel_identity(self, kernel_id):
        path = self._connection_file_path(kernel_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return compute_kernel_identity(json.load(f))

    async def probe_kernel(self, kernel_id, timeout=None):
        return kernel_id in self.live_kernels

    async def start_kernel(self, *, kernel_id=None, path=None, kernel_name="python", **kwargs):
        if kernel_id is None:
            kernel_id = f"k-{len(self._kernels)}"
        info = self._write_connection_file(kernel_id, kernel_name=kernel_name)
        self._kernels[kernel_id] = FakeKernel(kernel_name, info)
        self._kernel_connections[kernel_id] = 0
        self._kernel_identities[kernel_id] = compute_kernel_identity(info)
        self.live_kernels.add(kernel_id)
        return kernel_id

    async def adopt_kernel(self, kernel_id, *, kernel_name=None):
        if kernel_id in self.fail_adopt:
            return False
        path = self._connection_file_path(kernel_id)
        if not os.path.exists(path):
            return False
        with open(path) as f:
            info = json.load(f)
        self._kernels[kernel_id] = FakeKernel(info.get("kernel_name", "python"), info)
        self._kernel_connections[kernel_id] = 0
        self._kernel_identities[kernel_id] = compute_kernel_identity(info)
        self._kernel_adopted.add(kernel_id)
        return True

    def abandon_kernel(self, kernel_id):
        self._kernels.pop(kernel_id, None)
        self._kernel_adopted.discard(kernel_id)
        self._kernel_connections.pop(kernel_id, None)
        self._kernel_identities.pop(kernel_id, None)
        self._kernel_lease_gates.pop(kernel_id, None)

    async def shutdown_kernel(self, kernel_id, now=False, restart=False):
        gate = self._kernel_lease_gates.get(kernel_id, self._default_lease_gate)
        if gate is not None and not gate(kernel_id):
            from jupyter_server.services.kernels.kernelmanager import LeaseOwnershipLost

            raise LeaseOwnershipLost(kernel_id)
        self.shutdown_calls.append(kernel_id)
        self._kernels.pop(kernel_id, None)
        self.live_kernels.discard(kernel_id)
        self._kernel_adopted.discard(kernel_id)
        path = self._connection_file_path(kernel_id)
        if os.path.exists(path):
            os.remove(path)

    async def kernel_model(self, kernel_id):
        k = self._kernels[kernel_id]
        return {
            "id": kernel_id,
            "name": k.kernel_name,
            "last_activity": "2026-01-01T00:00:00Z",
            "execution_state": k.execution_state,
            "connections": self._kernel_connections.get(kernel_id, 0),
        }

    def update_env(self, kernel_id, env):
        pass

    def cwd_for_path(self, path, **kwargs):
        return "."


class FakeClock:
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_manager(tmp_path, *, km=None, ttl=30.0, db=None, clock=None):
    connection_dir = tmp_path / "runtime"
    connection_dir.mkdir(exist_ok=True)
    km = km if km is not None else FakeKernelManager(connection_dir)
    sm = SessionManager(
        kernel_manager=km,
        contents_manager=ContentsManager(),
        database_filepath=db or str(tmp_path / "sessions.db"),
        session_lease_ttl=ttl,
    )
    if clock is not None:
        sm._lease_clock = clock
        sm._leases = LeaseStore(sm.connection, ttl=ttl, clock=clock, log=sm.log)
    return sm, km


def seed_lease(sm, **overrides):
    """Insert a raw lease row, possibly owned by a foreign, dead server."""
    defaults = dict(
        session_id="sess-foreign",
        path="/nb.ipynb",
        name=None,
        type="notebook",
        kernel_id="kid-foreign",
        state=LeaseState.active.value,
        owner_id="dead-server",
        generation=3,
        expires_at=0,
        identity="",
    )
    defaults.update(overrides)
    cols = ",".join(defaults)
    marks = ",".join("?" for _ in defaults)
    sm.connection.execute(
        f"INSERT INTO session ({cols}) VALUES ({marks})", list(defaults.values())
    )
    return defaults


# ---------------------------------------------------------------------------
# LeaseStore unit tests
# ---------------------------------------------------------------------------
def _store(db, clock, ttl=30.0):
    conn = sqlite3.connect(db, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return LeaseStore(conn, ttl=ttl, clock=clock, log=None), conn


def test_lease_claim_requires_expired_fence(tmp_path):
    clock = FakeClock()
    store, _ = _store(":memory:", clock)
    lease = store.insert(
        session_id="s1", path="/a", name=None, mtype="notebook", kernel_id="k1"
    )
    # fresh fence -> another server cannot steal it
    with pytest.raises(LeaseConflict):
        store.claim(lease)


def test_lease_generation_bumps_and_blocks_stale_writer(tmp_path):
    clock = FakeClock()
    store, _ = _store(":memory:", clock)
    lease = store.insert(
        session_id="s1", path="/a", name=None, mtype="notebook", kernel_id="k1"
    )
    clock.advance(60)
    claimed = store.claim(lease)
    assert claimed.generation == lease.generation + 1
    assert claimed.state == LeaseState.claimed
    active = store.activate(claimed, identity="ident")
    assert active.state == LeaseState.active
    assert active.owner_id == store.server_id
    # the old owner's stale view can no longer mutate or release it
    with pytest.raises(LeaseConflict):
        store.release(lease)
    assert store.renew(lease) is None


def test_two_servers_only_one_wins_claim(tmp_path):
    db = str(tmp_path / "shared.db")
    clock = FakeClock()
    a, conn_a = _store(db, clock)
    lease_a = a.insert(
        session_id="s1", path="/a", name=None, mtype="notebook", kernel_id="k1"
    )
    # simulate server A dying; a fresh process (B) and C open the same db
    clock.advance(60)
    b, conn_b = _store(db, clock)
    c, conn_c = _store(db, clock)
    seen = b.get("s1")
    won = b.claim(seen)
    assert won.owner_id == b.server_id
    # C inspected the same row before B's write and races it
    with pytest.raises(LeaseConflict):
        c.claim(seen)
    # C's fresh read shows B's generation
    assert c.get("s1").owner_id == b.server_id
    conn_a.close()
    conn_b.close()
    conn_c.close()


def test_live_claim_cannot_be_stolen(tmp_path):
    clock = FakeClock()
    store, _ = _store(":memory:", clock)
    lease = store.insert(
        session_id="s1", path="/a", name=None, mtype="notebook", kernel_id="k1"
    )
    clock.advance(60)
    claimed = store.claim(lease)  # state=claimed, fresh fence
    # even though original lease was expired, the claim itself is a live fence
    with pytest.raises(LeaseConflict):
        store.claim(claimed)


def test_legacy_schema_migrates(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE session (session_id, path, name, type, kernel_id)")
    conn.execute(
        "INSERT INTO session VALUES (?,?,?,?,?)",
        ("s1", "/a.ipynb", None, "notebook", "k1"),
    )
    conn.commit()
    conn.close()
    clock = FakeClock()
    store, _ = _store(str(db), clock)
    lease = store.get("s1")
    assert lease is not None
    assert lease.state == LeaseState.active
    # legacy rows get an elapsed fence so the first server reconciles them
    assert store.is_expired(lease)


# ---------------------------------------------------------------------------
# SessionManager recovery scenarios
# ---------------------------------------------------------------------------
async def test_recovery_takes_over_live_orphan(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    info = km._write_connection_file("kid-foreign")
    km.live_kernels.add("kid-foreign")
    seed_lease(sm, identity=compute_kernel_identity(info), expires_at=0)

    summary = await sm.recover_sessions()
    assert summary == {
        "adopted": ["sess-foreign"],
        "quarantined": [],
        "cleaned": [],
        "skipped": [],
    }
    lease = sm.lease_store.get("sess-foreign")
    assert lease.state == LeaseState.active
    assert lease.owner_id == sm.server_id
    assert lease.generation == 4
    assert "kid-foreign" in km._kernel_adopted
    # kernel still answers and was never shut down
    assert km.shutdown_calls == []

    sessions = await sm.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["kernel"]["id"] == "kid-foreign"


async def test_recovery_cleans_dead_kernel_without_killing(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    info = km._write_connection_file("kid-foreign")  # not in live_kernels
    seed_lease(
        sm,
        kernel_id="kid-foreign",
        identity=compute_kernel_identity(info),
        expires_at=0,
    )

    summary = await sm.recover_sessions()
    assert summary["cleaned"] == ["sess-foreign"]
    assert sm.lease_store.get("sess-foreign") is None
    # no kill against anything alive; stale connection file removed
    assert km.shutdown_calls == []
    assert not os.path.exists(km._connection_file_path("kid-foreign"))


async def test_recovery_cleans_missing_connection_file(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    seed_lease(sm, identity="whatever", expires_at=0)
    summary = await sm.recover_sessions()
    assert summary["cleaned"] == ["sess-foreign"]


async def test_recovery_skips_live_foreign_fence(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    info = km._write_connection_file("kid-foreign")
    km.live_kernels.add("kid-foreign")
    seed_lease(
        sm,
        identity=compute_kernel_identity(info),
        owner_id="peer-alive",
        expires_at=clock() + 100,
    )
    summary = await sm.recover_sessions()
    assert summary["skipped"] == ["sess-foreign"]
    assert "kid-foreign" not in km._kernel_adopted
    # creating a second session for the same notebook is refused
    with pytest.raises(web.HTTPError) as exc:
        await sm.create_session(
            path="/nb.ipynb", kernel_name="python", type="notebook"
        )
    assert exc.value.status_code == 409
    # nothing new got started
    assert km._kernels == {}


async def test_recovery_quarantines_identity_mismatch(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    # kernel answering with a *different* identity than the record
    km._write_connection_file("kid-foreign", ports=(6001, 6002, 6003, 6004, 6005))
    km.live_kernels.add("kid-foreign")
    seed_lease(
        sm,
        identity=compute_kernel_identity(
            {
                "transport": "tcp",
                "ip": "127.0.0.1",
                "shell_port": 1,
                "iopub_port": 2,
                "stdin_port": 3,
                "control_port": 4,
                "hb_port": 5,
                "key": "old",
            }
        ),
        expires_at=0,
    )
    summary = await sm.recover_sessions()
    assert summary["quarantined"] == ["sess-foreign"]
    lease = sm.lease_store.get("sess-foreign")
    assert lease.state == LeaseState.quarantined
    # stranger kernel is neither adopted nor killed
    assert "kid-foreign" not in km._kernel_adopted
    assert km.shutdown_calls == []
    assert await sm.list_sessions() == []
    assert [l.session_id for l in sm.list_quarantined_sessions()] == ["sess-foreign"]


async def test_expired_fence_with_live_proof_is_not_killed(tmp_path):
    """Wall-clock expiry alone must never kill a kernel with valid proof."""
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    info = km._write_connection_file("kid-foreign")
    km.live_kernels.add("kid-foreign")
    seed_lease(sm, identity=compute_kernel_identity(info), expires_at=0)
    # even a quarantined, expired row with an answering kernel survives GC
    lease = sm.lease_store.get("sess-foreign")
    claimed = sm.lease_store.claim(lease)
    sm.lease_store.quarantine(claimed)
    clock.advance(1000)
    await sm._reap_expired_records()
    assert sm.lease_store.get("sess-foreign").state == LeaseState.quarantined
    # once the proof disappears, GC may collect it
    km.live_kernels.discard("kid-foreign")
    await sm._reap_expired_records()
    assert sm.lease_store.get("sess-foreign") is None


async def test_two_servers_simultaneous_recovery(tmp_path):
    clock = FakeClock()
    db = str(tmp_path / "shared.db")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    live = {"kid-foreign"}
    km_a = FakeKernelManager(runtime, live_kernels=live)
    km_b = FakeKernelManager(runtime, live_kernels=live)
    info = km_a._write_connection_file("kid-foreign")
    sm_a, _ = make_manager(tmp_path, km=km_a, db=db, clock=clock)
    sm_b, _ = make_manager(
        tmp_path, km=km_b, db=db, clock=clock
    )
    seed_lease(sm_a, identity=compute_kernel_identity(info), expires_at=0)
    # force both to evaluate recovery against the same expired snapshot
    results = await asyncio.gather(
        sm_a.recover_sessions(), sm_b.recover_sessions()
    )
    adopted = [r["adopted"] for r in results]
    assert sum(len(x) for x in adopted) == 1
    skipped = [r["skipped"] for r in results]
    assert any("sess-foreign" in s for s in skipped)
    lease = sm_a.lease_store.get("sess-foreign")
    assert lease.state == LeaseState.active
    owners = {sm_a.server_id, sm_b.server_id}
    assert lease.owner_id in owners
    # exactly one process attached
    assert ("kid-foreign" in km_a._kernel_adopted) ^ (
        "kid-foreign" in km_b._kernel_adopted
    )


async def test_mid_takeover_failure_quarantines(tmp_path):
    clock = FakeClock()
    km = FakeKernelManager(tmp_path / "runtime", fail_adopt={"kid-foreign"})
    (tmp_path / "runtime").mkdir(exist_ok=True)
    sm = SessionManager(
        kernel_manager=km,
        contents_manager=ContentsManager(),
        database_filepath=str(tmp_path / "s.db"),
    )
    sm._lease_clock = clock
    sm._leases = LeaseStore(sm.connection, ttl=30, clock=clock, log=sm.log)
    info = km._write_connection_file("kid-foreign")
    km.live_kernels.add("kid-foreign")
    seed_lease(sm, identity=compute_kernel_identity(info), expires_at=0)
    summary = await sm.recover_sessions()
    assert summary["quarantined"] == ["sess-foreign"]
    assert sm.lease_store.get("sess-foreign").state == LeaseState.quarantined
    assert "kid-foreign" not in km._kernel_adopted


async def test_rename_conditional_on_ownership(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    model = await sm.create_session(
        path="/a.ipynb", kernel_name="python", type="notebook"
    )
    sid = model["id"]
    await sm.update_session(sid, path="/b.ipynb")
    assert (await sm.get_session(session_id=sid))["path"] == "/b.ipynb"

    # a peer steals the lease after our fence expires
    lease = sm.lease_store.get(sid)
    clock.advance(60)
    peer = LeaseStore(
        sqlite3.connect(":memory:", isolation_level=None), ttl=30, clock=clock
    )
    # emulate the peer by writing directly through a second store on same db
    peer_conn = sm.connection
    peer = LeaseStore(peer_conn, ttl=30, clock=clock)
    stolen = peer.claim(lease)
    peer.activate(stolen)
    with pytest.raises(web.HTTPError) as exc:
        await sm.update_session(sid, path="/c.ipynb")
    assert exc.value.status_code == 409


async def test_delete_after_foreign_takeover_does_not_kill(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    model = await sm.create_session(
        path="/a.ipynb", kernel_name="python", type="notebook"
    )
    sid = model["id"]
    kid = model["kernel"]["id"]
    lease = sm.lease_store.get(sid)
    clock.advance(60)
    peer = LeaseStore(sm.connection, ttl=30, clock=clock)
    peer.activate(peer.claim(lease))

    with pytest.raises(web.HTTPError) as exc:
        await sm.delete_session(sid)
    assert exc.value.status_code == 409
    assert kid not in km.shutdown_calls
    assert sm.lease_store.get(sid).owner_id == peer.server_id


async def test_kernel_restart_records_new_identity(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    model = await sm.create_session(
        path="/a.ipynb", kernel_name="python", type="notebook"
    )
    sid = model["id"]
    kid = model["kernel"]["id"]
    old_identity = sm.lease_store.get(sid).identity
    # restart rewrote the connection file with new ports
    km._write_connection_file(kid, ports=(7001, 7002, 7003, 7004, 7005))
    new_identity = km.read_kernel_identity(kid)
    km._identity_changed_callback(kid, new_identity)
    assert sm.lease_store.get(sid).identity == new_identity != old_identity


async def test_reconnect_authorization(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    model = await sm.create_session(
        path="/a.ipynb", kernel_name="python", type="notebook"
    )
    kid = model["kernel"]["id"]
    assert await sm.authorize_kernel_connection(kid, "client-1")
    # foreign-owned lease for another kernel that happens to be attached here
    km._write_connection_file("kid-stranger", ports=(8001, 8002, 8003, 8004, 8005))
    await km.adopt_kernel("kid-stranger")
    sm.connection.execute(
        "INSERT INTO session (session_id,path,name,type,kernel_id,state,owner_id,"
        "generation,expires_at,identity) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "s-stranger",
            "/x.ipynb",
            None,
            "notebook",
            "kid-stranger",
            "active",
            "peer",
            1,
            clock() + 100,
            km.read_kernel_identity("kid-stranger"),
        ),
    )
    assert not await sm.authorize_kernel_connection("kid-stranger", "client-2")
    # unknown kernel is rejected outright
    assert not await sm.authorize_kernel_connection("nope", "client-3")


async def test_renewal_loses_ownership_detaches_adopted(tmp_path):
    clock = FakeClock()
    sm, km = make_manager(tmp_path, clock=clock)
    info = km._write_connection_file("kid-foreign")
    km.live_kernels.add("kid-foreign")
    seed_lease(sm, identity=compute_kernel_identity(info), expires_at=0)
    await sm.recover_sessions()
    assert "kid-foreign" in km._kernel_adopted
    # peer steals it while we hold a stale generation
    lease = sm.lease_store.get("sess-foreign")
    peer = LeaseStore(sm.connection, ttl=30, clock=clock)
    clock.advance(60)
    peer.activate(peer.claim(lease))
    sm._renew_leases()
    assert "kid-foreign" not in km._kernel_adopted
    assert "kid-foreign" not in km._kernels
