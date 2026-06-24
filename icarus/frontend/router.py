# router.py — owned by the daemon process
"""
Compute (or fetch cached) routes from store to a destination gid.
Pure data layer: connects to Postgres, returns RouteResult.

Public API:
  - route_to(gid) -> RouteResult
"""
import os
from dataclasses import dataclass

import psycopg2


DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "database": os.getenv("DB_NAME",     "routing_groningen"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "kokosnoot"),
    "port":     os.getenv("DB_PORT",     "5432"),
}

STORE = {
    "lat": float(os.getenv("STORE_LAT", "53.24657003069633")),
    "lon": float(os.getenv("STORE_LON", "6.587648954036653")),
}


@dataclass
class RouteResult:
    segments: list[str]   # geojson line strings
    meters: float


class RouteNotFound(Exception):
    pass


def _get_cached(cur, gid):
    cur.execute(
        "SELECT route_geom, route_meters FROM routing_export.route_cache "
        "WHERE nummeraanduiding_gid = %s",
        (gid,),
    )
    return cur.fetchone()


def _save_cache(cur, gid, segments, meters):
    cur.execute(
        """
        INSERT INTO routing_export.route_cache
            (nummeraanduiding_gid, route_geom, route_meters)
        SELECT %s, ST_AsGeoJSON(ST_Transform(
            ST_LineMerge(ST_Collect(ST_GeomFromGeoJSON(feat))), 4326)), %s
        FROM unnest(%s::text[]) AS feat
        ON CONFLICT (nummeraanduiding_gid) DO UPDATE
          SET route_geom = EXCLUDED.route_geom,
              route_meters = EXCLUDED.route_meters
        """,
        (gid, meters, segments),
    )


def _nearest_node_xy(cur, lat, lon):
    cur.execute(
        """
        SELECT source FROM routing_export.wegvakken
        ORDER BY geometrie <-> ST_Transform(
            ST_SetSRID(ST_Point(%s, %s), 4326), 28992)
        LIMIT 1
        """,
        (lon, lat),
    )
    r = cur.fetchone()
    return r[0] if r else None


_store_node_cache = None


def _store_node(cur):
    """Store graph node never changes; resolve it once per process."""
    global _store_node_cache
    if _store_node_cache is None:
        _store_node_cache = _nearest_node_xy(cur, STORE["lat"], STORE["lon"])
    return _store_node_cache


def _nearest_node_gid(cur, gid):
    cur.execute(
        """
        SELECT w.source
        FROM routing_export.adres_wegvak ak
        JOIN routing_export.wegvakken w ON w.wvk_id = ak.wegvak_id
        WHERE ak.nummeraanduiding_gid = %s
        LIMIT 1
        """,
        (gid,),
    )
    r = cur.fetchone()
    return r[0] if r else None


def _astar(cur, start, end):
    cur.execute(
        """
        SELECT ST_AsGeoJSON(ST_Transform(w.geometrie, 4326)), r.cost
        FROM pgr_aStar(
            'SELECT id, source, target, cost, reverse_cost,
                    ST_X(ST_StartPoint(geometrie)) AS x1,
                    ST_Y(ST_StartPoint(geometrie)) AS y1,
                    ST_X(ST_EndPoint(geometrie))   AS x2,
                    ST_Y(ST_EndPoint(geometrie))   AS y2
             FROM routing_export.wegvakken',
            %s, %s, directed := false
        ) AS r
        JOIN routing_export.wegvakken AS w ON r.edge = w.id
        ORDER BY r.seq
        """,
        (start, end),
    )
    rows = cur.fetchall()
    return [r[0] for r in rows], sum(r[1] for r in rows if r[1] is not None)


def route_to(gid: int) -> RouteResult:
    db = psycopg2.connect(**DB_CONFIG)
    try:
        with db.cursor() as cur:
            cached = _get_cached(cur, gid)
            if cached is not None:
                geom, meters = cached
                return RouteResult([geom], meters or 0.0)

            store_node = _store_node(cur)
            dest_node = _nearest_node_gid(cur, gid)
            if store_node is None or dest_node is None:
                raise RouteNotFound(f"no graph node for gid={gid}")

            segments, meters = _astar(cur, store_node, dest_node)
            if not segments:
                raise RouteNotFound(f"no path to gid={gid}")

            _save_cache(cur, gid, segments, meters)
            db.commit()
            return RouteResult(segments, meters)
    finally:
        db.close()


def route_to_xy(lat: float, lon: float) -> RouteResult:
    """Route from the store to an arbitrary WGS84 lat/lon (the daemon sends coordinates,
    not a gid). No DB cache — the cache table is keyed by nummeraanduiding_gid — so callers
    should route once per destination (the server does: only on the 'pin' event)."""
    db = psycopg2.connect(**DB_CONFIG)
    try:
        with db.cursor() as cur:
            store_node = _store_node(cur)
            dest_node = _nearest_node_xy(cur, lat, lon)
            if store_node is None or dest_node is None:
                raise RouteNotFound(f"no graph node for ({lat},{lon})")

            segments, meters = _astar(cur, store_node, dest_node)
            if not segments:
                raise RouteNotFound(f"no path to ({lat},{lon})")
            return RouteResult(segments, meters)
    finally:
        db.close()