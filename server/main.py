"""
FastAPI server for OGC simulation.

Endpoints:
    GET  /              — serves the Three.js viewer (index.html)
    POST /load          — load cloth + obstacle meshes, returns initial mesh data
    POST /config        — update simulation parameters
    POST /run           — start simulation (streams frames over WebSocket)
    POST /pause         — pause a running simulation
    POST /reset         — reset to initial state
    GET  /export        — export simulation as .glb download
    WS   /ws            — WebSocket for live frame streaming

Run:
    cd server && uvicorn main:app --reload --port 8000
    or:
    python server/main.py
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import os
from pathlib import Path

import numpy as np

# Add project root to path so ogc_sim is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from concurrent.futures import ThreadPoolExecutor

from ogc_sim.sim.config import SceneConfig
from ogc_sim.sim.runner import OGCSimulator
from ogc_sim.sim.frame import FrameData
from ogc_sim.io.exporter import export_gltf


app = FastAPI(title="OGC Simulation Server")

# Serve static files (index.html, viewer.js, etc.)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global simulator instance
sim = OGCSimulator()

# Connected WebSocket clients
ws_clients: list[WebSocket] = []

# Store frames for export
all_frames: list[FrameData] = []

# Thread pool for long-running simulation steps
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sim_")


# ======================================================================
# Helper: pack frame as binary for WebSocket
# ======================================================================

def pack_frame_binary(frame: FrameData) -> bytes:
    """
    Pack a FrameData into a compact binary message.

    Format:
        4 bytes: step (uint32)
        4 bytes: num_contacts (uint32)
        4 bytes: N_cloth vertices (uint32)
        N_cloth * 3 * 4 bytes: cloth vertex positions (float32)
    """
    V = frame.V_cloth.astype(np.float32)
    n_verts = len(V)
    header = struct.pack("<III", frame.step, frame.num_contacts, n_verts)
    return header + V.tobytes()


def pack_initial_mesh(frame: FrameData, F_cloth: np.ndarray, F_obstacle: np.ndarray) -> dict:
    """
    Pack full mesh data as JSON for initial load.

    Includes vertices, faces, and topology for both cloth and obstacle.
    """
    return {
        "type": "init",
        "cloth": {
            "vertices": frame.V_cloth.tolist(),
            "faces": F_cloth.tolist(),
        },
        "obstacle": {
            "vertices": frame.V_obstacle.tolist(),
            "faces": F_obstacle.tolist(),
        },
        "step": frame.step,
    }


# ======================================================================
# REST endpoints
# ======================================================================

@app.get("/")
async def index():
    """Serve the main viewer page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/load")
async def load_scene(
    cloth_path: str = "",
    obstacle_path: str = "",
    dt: float = 0.02,
    mass: float = 1.0,
    k_s: float = 200.0,
    k_c: float = 500.0,
    r: float = 0.05,
    r_q: float = 0.1,
    gamma_p: float = 0.45,
    gamma_e: float = 0.0,
    n_iter: int = 5,
    n_steps: int = 100,
    gravity_x: float = 0.0,
    gravity_y: float = 0.0,
    gravity_z: float = 0.0,
    vel_x: float = 0.0,
    vel_y: float = 0.0,
    vel_z: float = 0.0,
):
    """Load meshes and initialise the simulation."""
    global all_frames

    config = SceneConfig(
        cloth_path=cloth_path,
        obstacle_path=obstacle_path,
        dt=dt,
        mass=mass,
        k_s=k_s,
        k_c=k_c,
        r=r,
        r_q=r_q,
        gamma_p=gamma_p,
        gamma_e=gamma_e,
        n_iter=n_iter,
        n_steps=n_steps,
        gravity=(gravity_x, gravity_y, gravity_z),
        cloth_initial_velocity=(vel_x, vel_y, vel_z),
    )

    try:
        initial_frame = sim.load(config)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    all_frames = [initial_frame]

    return pack_initial_mesh(initial_frame, sim.F_cloth, sim.F_obstacle)


@app.post("/load_upload")
async def load_upload(
    cloth_file: UploadFile = File(...),
    obstacle_file: UploadFile = File(...),
):
    """Load meshes from uploaded files."""
    global all_frames

    # Save uploads to temp files
    upload_dir = PROJECT_ROOT / "server" / "uploads"
    upload_dir.mkdir(exist_ok=True)

    cloth_path = upload_dir / cloth_file.filename
    obstacle_path = upload_dir / obstacle_file.filename

    with open(cloth_path, "wb") as f:
        f.write(await cloth_file.read())
    with open(obstacle_path, "wb") as f:
        f.write(await obstacle_file.read())

    config = SceneConfig(
        cloth_path=str(cloth_path),
        obstacle_path=str(obstacle_path),
    )

    try:
        initial_frame = sim.load(config)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    all_frames = [initial_frame]
    return pack_initial_mesh(initial_frame, sim.F_cloth, sim.F_obstacle)


@app.post("/config")
async def update_config(params: dict):
    """Update simulation parameters without reloading meshes."""
    try:
        sim.update_config(**params)
        return {"status": "ok", "updated": list(params.keys())}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/reset")
async def reset():
    """Reset simulation to initial state."""
    global all_frames
    frame = sim.reset()
    if frame is None:
        return JSONResponse({"error": "No scene loaded"}, status_code=400)
    all_frames = [frame]
    return {"status": "ok", "step": 0}


@app.post("/step")
async def step_once():
    """Run a single simulation step, returns frame data as JSON."""
    if not sim.is_loaded:
        return JSONResponse({"error": "No scene loaded"}, status_code=400)
    try:
        # Run in thread pool to avoid blocking
        frame = await asyncio.get_event_loop().run_in_executor(
            executor, sim.step
        )
        all_frames.append(frame)
        return {
            "step": frame.step,
            "num_contacts": frame.num_contacts,
            "vertices": frame.V_cloth.tolist(),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/export")
async def export():
    """Export all recorded frames as a .glb file."""
    if not all_frames or not sim.is_loaded:
        return JSONResponse({"error": "No frames to export"}, status_code=400)

    output_path = PROJECT_ROOT / "server" / "output" / "simulation.glb"
    try:
        export_gltf(all_frames, sim.F_cloth, sim.F_obstacle, output_path)
        return FileResponse(
            str(output_path),
            media_type="model/gltf-binary",
            filename="simulation.glb",
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ======================================================================
# WebSocket — live frame streaming
# ======================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket for live simulation streaming.

    Protocol:
        Client sends JSON commands:
            {"cmd": "run", "n_steps": 100}
            {"cmd": "step"}
            {"cmd": "pause"}
            {"cmd": "reset"}

        Server responds:
            Binary frames (pack_frame_binary) during simulation
            JSON messages for status/errors
    """
    await ws.accept()
    ws_clients.append(ws)

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            cmd = msg.get("cmd", "")

            if cmd == "run":
                n = msg.get("n_steps", sim.config.n_steps if sim.config else 50)
                if not sim.is_loaded:
                    await ws.send_json({"error": "No scene loaded"})
                    continue

                print(f"[WS] Starting run: {n} steps, current step={sim.step_num}")
                await ws.send_json({"status": "running", "n_steps": n})

                for i in range(n):
                    if sim._paused:
                        await ws.send_json({"status": "paused", "step": sim.step_num})
                        break

                    try:
                        print(f"[WS] Running step {i+1}/{n}...", flush=True)
                        # Run the step in a thread pool to avoid blocking the WebSocket
                        frame = await asyncio.get_event_loop().run_in_executor(
                            executor, sim.step
                        )
                        all_frames.append(frame)
                        print(f"[WS] Step {frame.step} done, sending frame...", flush=True)

                        # Send binary frame to this client
                        if ws.client_state == WebSocketState.CONNECTED:
                            pkt = pack_frame_binary(frame)
                            print(f"[WS] Sending {len(pkt)} bytes to client", flush=True)
                            await ws.send_bytes(pkt)

                        # Broadcast to other clients
                        for client in ws_clients:
                            if client is not ws and client.client_state == WebSocketState.CONNECTED:
                                try:
                                    await client.send_bytes(pack_frame_binary(frame))
                                except Exception:
                                    pass

                        # Yield to event loop so the client can render
                        await asyncio.sleep(0.01)

                    except Exception as e:
                        print(f"[WS] ERROR in step {i}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        await ws.send_json({"error": str(e), "step": sim.step_num})
                        break

                print(f"[WS] Run complete at step {sim.step_num}", flush=True)
                await ws.send_json({"status": "done", "step": sim.step_num})

            elif cmd == "step":
                if not sim.is_loaded:
                    await ws.send_json({"error": "No scene loaded"})
                    continue
                try:
                    frame = await asyncio.get_event_loop().run_in_executor(
                        executor, sim.step
                    )
                    all_frames.append(frame)
                    await ws.send_bytes(pack_frame_binary(frame))
                except Exception as e:
                    await ws.send_json({"error": str(e)})

            elif cmd == "pause":
                sim.pause()
                await ws.send_json({"status": "paused", "step": sim.step_num})

            elif cmd == "reset":
                frame = sim.reset()
                if frame:
                    all_frames.clear()
                    all_frames.append(frame)
                    await ws.send_json({"status": "reset", "step": 0})
                else:
                    await ws.send_json({"error": "No scene loaded"})

            else:
                await ws.send_json({"error": f"Unknown command: {cmd}"})

    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ======================================================================
# Shutdown handler
# ======================================================================

import signal
import threading

def shutdown_handler(signum, frame):
    print("\n[SERVER] Shutting down...", flush=True)
    executor.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    import uvicorn

    try:
        print("[SERVER] Starting OGC Simulation Server on http://0.0.0.0:8000")
        print("[SERVER] Press CTRL+C to shut down")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n[SERVER] Interrupted by user", flush=True)
    finally:
        print("[SERVER] Cleaning up...", flush=True)
        executor.shutdown(wait=False)
        sys.exit(0)
