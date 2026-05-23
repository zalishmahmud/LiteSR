"""
llmsr_tracker.py
================
Drop this file next to your existing llmsr/ code.

Usage — add TWO lines to whatever script launches your run:

    import llmsr_tracker
    llmsr_tracker.patch()          # monkey-patches ExperienceBuffer/Island/Cluster

    # ... rest of your normal code unchanged ...

    # Optional: block until dashboard disconnects
    # llmsr_tracker.wait()

The tracker starts a WebSocket server on ws://localhost:8765.
Open the React dashboard in your browser, it will connect automatically.
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
import functools
from typing import Any

# ── optional dep check ────────────────────────────────────────────────────────
try:
    import websockets
    import websockets.server
except ImportError:
    raise ImportError(
        "\n\n  llmsr_tracker requires 'websockets'.\n"
        "  Install it with:  pip install websockets\n"
    )

# ── internal state ────────────────────────────────────────────────────────────
_start_time: float = time.time()
_event_id: int = 0
_clients: set = set()
_loop: asyncio.AbstractEventLoop | None = None
_event_backlog: list[dict] = []          # sent to late-joining clients
_BACKLOG_MAX = 2000
_server_thread: threading.Thread | None = None
WS_PORT = 8765


# ── event emission ────────────────────────────────────────────────────────────

def _make_event(event_type: str, source: str, data: dict) -> dict:
    global _event_id
    evt = {
        "id":         _event_id,
        "type":       event_type,
        "source":     source,
        "relativeMs": int((time.time() - _start_time) * 1000),
        "data":       _sanitize(data),
    }
    _event_id += 1
    return evt


def _sanitize(obj: Any) -> Any:
    """Recursively make obj JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj == float("inf"):  return  "Infinity"
        if obj == float("-inf"): return "-Infinity"
        return obj
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    return str(obj)


def emit(event_type: str, source: str, data: dict) -> None:
    """Emit an event to the dashboard. Thread-safe."""
    evt = _make_event(event_type, source, data)
    _event_backlog.append(evt)
    if len(_event_backlog) > _BACKLOG_MAX:
        _event_backlog.pop(0)

    if _loop and _clients:
        msg = json.dumps(evt)
        asyncio.run_coroutine_threadsafe(_broadcast(msg), _loop)


async def _broadcast(msg: str) -> None:
    dead = set()
    for ws in _clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


# ── WebSocket server ──────────────────────────────────────────────────────────

async def _handler(websocket):
    _clients.add(websocket)
    print(f"[tracker] Dashboard connected ({len(_clients)} client(s))")

    # Send full backlog so late-joining dashboard catches up
    if _event_backlog:
        await websocket.send(json.dumps({"type": "_backlog", "events": _event_backlog}))

    try:
        async for _ in websocket:
            pass          # we don't need client→server messages
    except Exception:
        pass
    finally:
        _clients.discard(websocket)
        print(f"[tracker] Dashboard disconnected ({len(_clients)} client(s))")


def _find_free_port(start: int) -> int:
    """Try ports starting at `start` until one is free."""
    import socket
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start}–{start + 20}")


def _run_server():
    global _loop, WS_PORT
    import socket as _socket

    _loop = asyncio.new_event_loop()
    # Do NOT call asyncio.set_event_loop() here — that would overwrite the main
    # thread's event loop and break any asyncio used by the host program.
    # This loop runs purely inside this daemon thread.

    async def _serve():
        global WS_PORT

        # Build a socket with SO_REUSEADDR + SO_REUSEPORT so:
        #   (a) TIME_WAIT sockets left from a previous run don't block us
        #   (b) if we need to rebind mid-run we can do so without collision
        def _make_socket(port: int):
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
            except AttributeError:
                pass   # SO_REUSEPORT not available on all platforms (Windows)
            sock.bind(("localhost", port))
            return sock

        # Find a free port (usually the first one works thanks to REUSEADDR)
        sock = None
        for port in range(WS_PORT, WS_PORT + 20):
            try:
                sock = _make_socket(port)
                WS_PORT = port
                break
            except OSError:
                print(f"[tracker] Port {port} busy, trying {port + 1}…")

        if sock is None:
            print("[tracker] ERROR: could not bind to any port. Tracker disabled.")
            return

        print(f"[tracker] WebSocket server listening on ws://localhost:{WS_PORT}")

        # Pass our pre-bound socket directly — websockets won't try to bind again
        async with websockets.server.serve(_handler, sock=sock):
            await asyncio.Future()   # run forever

    _loop.run_until_complete(_serve())


def _start_server():
    global _server_thread
    if _server_thread and _server_thread.is_alive():
        return
    _server_thread = threading.Thread(target=_run_server, daemon=True, name="llmsr-tracker-ws")
    _server_thread.daemon = True   # ensure it never blocks program exit
    _server_thread.start()

    # Wait just until the socket is bound (port is confirmed) before returning
    import time
    deadline = time.time() + 3.0
    while WS_PORT == 8765 and not _server_thread.is_alive():
        if time.time() > deadline:
            break
        time.sleep(0.05)
    time.sleep(0.1)   # tiny extra margin for the bind to complete


def wait():
    """Block the main thread (useful at end of script to keep server alive)."""
    if _server_thread:
        _server_thread.join()


# ── monkey-patch helpers ──────────────────────────────────────────────────────

def _wrap(original_fn, before=None, after=None):
    """Wrap an instance method, calling before/after hooks with (self, *args, **kwargs)."""
    @functools.wraps(original_fn)
    def wrapper(self, *args, **kwargs):
        if before:
            before(self, *args, **kwargs)
        result = original_fn(self, *args, **kwargs)
        if after:
            after(self, result, *args, **kwargs)
        return result
    return wrapper


# ── patch functions ───────────────────────────────────────────────────────────

def _patch_experience_buffer():
    from litesr.solver_agent import buffer as eb

    # ── Island id counter — incremented in Island.__init__ BEFORE hook
    #    reset by ExperienceBuffer.__init__ and reset_islands BEFORE hooks
    _island_counter = {"next": 0}

    # ── ExperienceBuffer.__init__
    _orig_eb_init = eb.ExperienceBuffer.__init__
    def _eb_init_before(self, config, template, function_to_evolve):
        _island_counter["next"] = 0   # reset counter before islands are created
    def _eb_init_after(self, result, config, template, function_to_evolve):
        emit("buffer:created", "ExperienceBuffer", {
            "numIslands":         config.num_islands,
            "functionsPerPrompt": config.functions_per_prompt,
            "resetPeriodSec":     config.reset_period,
        })
    eb.ExperienceBuffer.__init__ = _wrap(_orig_eb_init, before=_eb_init_before, after=_eb_init_after)

    # ── ExperienceBuffer._register_program_in_island
    _orig_reg_island = eb.ExperienceBuffer._register_program_in_island
    def _reg_island_before(self, program, island_id, scores_per_test, **kwargs):
        import numpy as np
        score = eb._reduce_score(scores_per_test)
        prev  = self._best_score_per_island[island_id]
        if score > prev:
            emit("buffer:best_score_updated", f"ExperienceBuffer.island[{island_id}]", {
                "islandId":  island_id,
                "prevScore": None if prev == float("-inf") else prev,
                "newScore":  score,
                "program":   str(program),
            })
    eb.ExperienceBuffer._register_program_in_island = _wrap(
        _orig_reg_island, before=_reg_island_before)

    # ── ExperienceBuffer.reset_islands
    _orig_reset = eb.ExperienceBuffer.reset_islands
    def _reset_before(self):
        scores = [s if s != float("-inf") else None for s in self._best_score_per_island]
        emit("buffer:islands_reset", "ExperienceBuffer", {
            "scoresBeforeReset": scores,
            "numIslands":        len(self._islands),
        })
        # Counter will be set per-island via the sorted reset indices;
        # we track which slots are being reset so new Island objects get the right id.
        # Store the reset indices so Island.__init__ can pick them up in order.
        import numpy as np
        indices_sorted = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        num_to_reset = len(self._islands) // 2
        _island_counter["reset_queue"] = list(indices_sorted[:num_to_reset])
        _island_counter["reset_pos"] = 0
    def _reset_after(self, result):
        # Clean up reset queue
        _island_counter.pop("reset_queue", None)
        _island_counter.pop("reset_pos", None)
    eb.ExperienceBuffer.reset_islands = _wrap(_orig_reset, before=_reset_before, after=_reset_after)

    # ── Island.__init__
    _orig_island_init = eb.Island.__init__
    def _island_init_before(self, template, function_to_evolve,
                            functions_per_prompt, temp_init, temp_period):
        # Assign id from reset_queue if available, else use sequential counter
        if "reset_queue" in _island_counter:
            pos = _island_counter["reset_pos"]
            queue = _island_counter["reset_queue"]
            if pos < len(queue):
                self._tracker_id = queue[pos]
                _island_counter["reset_pos"] = pos + 1
            else:
                self._tracker_id = _island_counter["next"]
                _island_counter["next"] += 1
        else:
            self._tracker_id = _island_counter["next"]
            _island_counter["next"] += 1

    def _island_init_after(self, result, template, function_to_evolve,
                           functions_per_prompt, temp_init, temp_period):
        island_id = self._tracker_id   # guaranteed set by before hook
        emit("island:created", f"island[{island_id}]", {
            "islandId":           island_id,
            "functionsPerPrompt": functions_per_prompt,
            "tempInit":           temp_init,
            "tempPeriod":         temp_period,
        })
    eb.Island.__init__ = _wrap(_orig_island_init, before=_island_init_before, after=_island_init_after)

    # ── Island.register_program
    _orig_island_reg = eb.Island.register_program
    def _island_reg_after(self, result, program, scores_per_test):
        sig       = eb._get_signature(scores_per_test)
        score     = eb._reduce_score(scores_per_test)
        island_id = getattr(self, "_tracker_id", "?")
        # Stamp the cluster so sample_program events can include islandId + signature
        if sig in self._clusters:
            self._clusters[sig]._tracker_island_id  = island_id
            self._clusters[sig]._tracker_signature  = str(sig)
        emit("island:program_registered", f"island[{island_id}]", {
            "islandId":      island_id,
            "signature":     str(sig),
            "score":         score,
            "totalPrograms": self._num_programs,
            "clusterCount":  len(self._clusters),
            "program":       str(program),
        })
    eb.Island.register_program = _wrap(_orig_island_reg, after=_island_reg_after)

    # ── Cluster.__init__
    _orig_cluster_init = eb.Cluster.__init__
    def _cluster_init_after(self, result, score, implementation):
        emit("cluster:created", f"cluster[{id(self)}]", {
            "score":        score,
            "firstProgram": str(implementation),
            "programCount": 1,
        })
    eb.Cluster.__init__ = _wrap(_orig_cluster_init, after=_cluster_init_after)

    # ── Cluster.register_program
    _orig_cluster_reg = eb.Cluster.register_program
    def _cluster_reg_after(self, result, program):
        emit("cluster:program_added", f"cluster[{id(self)}]", {
            "programCount": len(self._programs),
            "newProgram":   str(program),
            "score":        self._score,
        })
    eb.Cluster.register_program = _wrap(_orig_cluster_reg, after=_cluster_reg_after)

    # ── Cluster.sample_program
    _orig_sample = eb.Cluster.sample_program
    def _sample_after(self, result):
        emit("cluster:sampled", f"cluster[{id(self)}]", {
            "sampledProgram": str(result),
            "poolSize":       len(self._programs),
            "score":          self._score,
            "islandId":       getattr(self, "_tracker_island_id", None),
            "signature":      str(getattr(self, "_tracker_signature", "")),
        })
    eb.Cluster.sample_program = _wrap(_orig_sample, after=_sample_after)


# ── patch Sampler ─────────────────────────────────────────────────────────────

def _patch_sampler():
    from litesr.solver_agent import sampler as sm

    _orig_hook = sm.Sampler._on_epoch_start

    def _on_epoch_start_patched(self, epoch_num: int) -> None:
        emit("sampler:epoch", "Sampler", {
            "epoch":         epoch_num,
            "globalSamples": sm.Sampler._global_samples_nums,
        })
        _orig_hook(self, epoch_num)

    sm.Sampler._on_epoch_start = _on_epoch_start_patched


# ── public API ────────────────────────────────────────────────────────────────

def patch(port: int = WS_PORT):
    """
    Call once at the top of your script before anything else runs.
    Starts the WebSocket server and monkey-patches ExperienceBuffer/Island/Cluster.
    """
    global WS_PORT
    WS_PORT = port
    _start_server()
    _patch_experience_buffer()
    _patch_sampler()
    print(f"[tracker] Patched ExperienceBuffer, Island, Cluster, Sampler.")
    print(f"[tracker] Dashboard WebSocket: ws://localhost:{WS_PORT}")