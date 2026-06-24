# app.py — order board server
"""
Source of truth for the order board. Consumes the daemon's discrete events
(live / pin / timed / absent) at POST /api/orders, holds each order's state, runs the
end-of-life lifecycle (oven -> rack -> onroute -> finished), and pushes every change to
browsers over SSE (/events). Routes are computed off the request thread by a background
worker so a slow A* never blocks (or times out) the daemon's POST.

Daemon contract (batched):
    POST /api/orders  {"frame": N, "updates": [ <event>, ... ]}
    live   {"event","number","type","timed","created_at"} -> create (status live) / re-announce (preparing)
    pin    {"event","number","address","lat","lon","kind"} -> set location, kick off routing
    timed  {"event","number","timed"}                      -> update timed flag
    absent {"event","number"}                              -> begin end-of-life (status oven)

Order record (stored + broadcast):
    {number, status, type, timed, address, lat, lon, route, distance, countdown, createdAt}
    status   : live | preparing | oven | rack | onroute | finished
    countdown: epoch when the current phase ends (None outside the lifecycle)
    distance : route length in metres (deliveries only)
"""
import json
import time
import threading
from threading import Lock
from queue import Queue, Empty
import logging

from flask import Flask, jsonify, render_template, request, Response

import router

app = Flask(__name__)

orders: dict[str, dict] = {}          # number -> order record
lock = Lock()
event_subscribers: list[Queue] = []
subscribers_lock = Lock()
route_jobs: "Queue[str]" = Queue()    # numbers awaiting a route computation

# --- lifecycle timings (seconds) ---
OVEN_SECS = 360                       # absent -> in the oven
RACK_SECS = 180                       # oven   -> on the rack
SPEED_MPS = 6.0                       # onroute time = route_metres / SPEED_MPS  (~21.6 km/h)

# --- in-memory last frame ------------------------------------------------------------------
_last_frame: bytes | None = None
_last_frame_lock = Lock()

def now() -> float:
    return time.time()


# --- SSE broadcast ------------------------------------------------------------------------
def _broadcast(payload: dict):
    data = json.dumps(payload)
    with subscribers_lock:
        dead = []
        for q in event_subscribers:
            try:
                q.put_nowait(data)
            except Exception:
                dead.append(q)
        for q in dead:
            event_subscribers.remove(q)


def broadcast_order(o: dict):
    _broadcast(o)                                  # full record -> frontend upserts by number


def broadcast_delete(number: str):
    _broadcast({"event": "deleted", "number": number})


# --- event handling -----------------------------------------------------------------------
def _new_order(number, type_, timed, created_at=None) -> dict:
    return {"number": number, "status": "live", "type": type_, "timed": bool(timed),
            "address": None, "lat": None, "lon": None, "gid": None,
            "route": None, "distance": None, "countdown": None,
            "createdAt": created_at or now()}      # daemon stamps first-seen; now() only if absent


def apply_event(ev: dict):
    """Apply one daemon event to the order map. Caller holds `lock`."""
    et = ev.get("event")
    number = ev.get("number")
    if number is None:
        return

    if et == "live":
        if ev.get("type") in ("Carryout", "Dinein"):   # neither belongs on the delivery board
            return
        o = orders.get(number)
        if o is None:
            orders[number] = _new_order(number, ev.get("type"), ev.get("timed"),
                                        ev.get("created_at"))
        else:                                      # re-announce: back to prep, cancel end-of-life
            o["status"] = "preparing"
            o["countdown"] = None
            if ev.get("type") is not None:
                o["type"] = ev["type"]
            o["timed"] = bool(ev.get("timed", o["timed"]))
        broadcast_order(orders[number])

    elif et == "pin":
        o = orders.get(number)
        if o is None:                              # defensive: pin before live (resync ordering)
            o = orders[number] = _new_order(number, None, False)
        o["lat"] = ev.get("lat")
        o["lon"] = ev.get("lon")
        o["address"] = ev.get("address")
        o["gid"] = ev.get("gid")
        broadcast_order(o)                         # show the marker now; route follows async
        if o["gid"] is not None or o["lat"] is not None:
            route_jobs.put(number)

    elif et == "timed":
        o = orders.get(number)
        if o is not None:
            o["timed"] = bool(ev.get("timed"))
            broadcast_order(o)

    elif et == "absent":
        o = orders.get(number)
        if o is not None:                          # ignore absent for an unknown number
            o["status"] = "oven"
            o["countdown"] = now() + OVEN_SECS
            broadcast_order(o)

# --- Suppress /latest.jpg logs ---
class FilterLatestJpg(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        silenced_routes = ["GET /latest.jpg", "POST /api/frame"]
        
        # Drop the log if any of the silenced routes are in the message
        return not any(route in msg for route in silenced_routes)

# Apply the filter to the default Werkzeug logger
logging.getLogger("werkzeug").addFilter(FilterLatestJpg())

@app.post("/api/orders")
def upsert():
    data = request.get_json(force=True, silent=True) or {}
    updates = data.get("updates")
    if updates is None:                            # tolerate a single bare event too
        updates = [data]
    with lock:
        for ev in updates:
            print(f"[recv] {ev}", flush=True)      # the daemon posts diffs only -> log them all
            apply_event(ev)
    return "", 200


@app.delete("/api/orders/<number>")
def delete(number):
    with lock:
        existed = orders.pop(number, None) is not None
    if existed:
        broadcast_delete(number)
    return "", 202


@app.get("/api/orders")
def list_orders():
    with lock:
        return jsonify(list(orders.values()))


@app.get("/")
def index():
    return render_template("index.html")

# --- Frame stream -------------------------------------------------------------------    
@app.post("/api/frame")
def push_frame():
    global _last_frame
    data = request.get_data()          # raw JPEG bytes
    if not data:
        return "", 400
    with _last_frame_lock:
        _last_frame = data
        
    # Announce to all connected browsers that a new frame is ready
    _broadcast({"event": "new_frame"}) 
    
    return "", 204

@app.get("/latest.jpg")
def latest_jpg():
    with _last_frame_lock:
        frame = _last_frame
    if frame is None:
        return "", 404
    return Response(frame, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})

# --- background workers -------------------------------------------------------------------
def _router_worker():
    """Compute routes off the request thread; broadcast route + distance when ready.
    Prefers gid routing (precomputed address->segment match + route cache); falls back to
    raw lat/lon nearest-node routing when no gid resolved."""
    while True:
        number = route_jobs.get()
        with lock:
            o = orders.get(number)
            gid = o.get("gid") if o else None
            ll = (o["lat"], o["lon"]) if o else None
        try:
            if gid is not None:
                res = router.route_to(gid)
            elif ll and ll[0] is not None:
                res = router.route_to_xy(ll[0], ll[1])
            else:
                continue
        except Exception as e:                     # RouteNotFound, DB down, etc. -> no route
            print(f"[route] {number} failed: {e}", flush=True)
            continue
        with lock:
            o = orders.get(number)
            if o is None:                          # deleted while routing
                continue
            o["route"] = res.segments
            o["distance"] = res.meters
            broadcast_order(o)


def _timerloop():
    """Advance the lifecycle once per second: at most one transition per order per expiry."""
    while True:
        time.sleep(1)
        t = now()
        with lock:
            for o in list(orders.values()):
                cd = o.get("countdown")
                if cd is None or cd > t:
                    continue
                st = o["status"]
                if st == "oven":
                    o["status"] = "rack"; o["countdown"] = t + RACK_SECS
                elif st == "rack":
                    if o.get("distance"):                       # delivery -> drive it
                        o["status"] = "onroute"; o["countdown"] = t + o["distance"] / SPEED_MPS
                    else:                                       # carryout / no route -> done
                        o["status"] = "finished"; o["countdown"] = None
                elif st == "onroute":
                    o["status"] = "finished"; o["countdown"] = None
                else:
                    continue
                broadcast_order(o)


# --- SSE stream ---------------------------------------------------------------------------
@app.get("/events")
def events():
    q: Queue = Queue(maxsize=100)
    with subscribers_lock:
        event_subscribers.append(q)
    with lock:
        snapshot = list(orders.values())

    def gen():
        try:
            yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
            while True:
                try:
                    data = q.get(timeout=15)
                    yield f"data: {data}\n\n"
                except Empty:
                    yield ": keepalive\n\n"
        finally:
            with subscribers_lock:
                if q in event_subscribers:
                    event_subscribers.remove(q)

    return Response(gen(), mimetype="text/event-stream")


threading.Thread(target=_router_worker, daemon=True).start()
threading.Thread(target=_timerloop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
