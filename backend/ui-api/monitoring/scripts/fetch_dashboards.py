#!/usr/bin/env python3
import json
import urllib.request
from pathlib import Path

DASHBOARDS = [
    {"id": 1860, "revision": 42, "file": "node-exporter-full.json"},
    {"id": 3662, "revision": 2, "file": "prometheus-overview.json"},
    {"id": 7587, "revision": 3, "file": "blackbox-exporter.json"},
    {"id": 13639, "revision": 2, "file": "loki-logs.json"},
]

DATASOURCE_BY_PLUGIN = {
    "prometheus": "Prometheus",
    "loki": "Loki",
}

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "grafana" / "dashboards"


def download_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_input_mapping(dashboard: dict) -> dict:
    mapping = {}
    inputs = dashboard.get("__inputs", [])
    for entry in inputs:
        if entry.get("type") != "datasource":
            continue
        name = entry.get("name")
        plugin = (entry.get("pluginName") or str(entry.get("pluginId") or "")).lower()
        for key, datasource in DATASOURCE_BY_PLUGIN.items():
            if key in plugin:
                mapping[name] = datasource
                break
    return mapping


def replace_inputs(value, mapping):
    if isinstance(value, str):
        for key, datasource in mapping.items():
            if value == f"${{{key}}}":
                return datasource
        return value
    if isinstance(value, list):
        return [replace_inputs(item, mapping) for item in value]
    if isinstance(value, dict):
        return {k: replace_inputs(v, mapping) for k, v in value.items()}
    return value


def normalize_dashboard(dashboard: dict) -> dict:
    mapping = build_input_mapping(dashboard)
    if mapping:
        dashboard = replace_inputs(dashboard, mapping)
    dashboard.pop("__inputs", None)
    return dashboard


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in DASHBOARDS:
        url = f"https://grafana.com/api/dashboards/{item['id']}/revisions/{item['revision']}/download"
        data = download_json(url)
        data = normalize_dashboard(data)
        out_path = OUT_DIR / item["file"]
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
