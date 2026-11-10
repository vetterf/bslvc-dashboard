"""Describe group-coloured traces for the browser-side palette editor."""

import re


VTYPE_DEFAULTS = {"ENL": "#1f77b4", "ESL": "#ff7f0e", "EFL": "#2ca02c", "Other": "#cccccc"}


def tag_group_colors(fig, grouping, use_legendgroup=False):
    if grouping in ("vtype", "vtype_balanced"):
        grouping = "vtype"
    if grouping not in ("variety", "gender", "vtype"):
        return fig

    is_dict = isinstance(fig, dict)
    if is_dict:
        layout = fig.setdefault("layout", {})
        layout["meta"] = {**(layout.get("meta") or {}), "group_color_kind": grouping}
        traces = fig.get("data", [])
    else:
        fig.update_layout(meta={**(fig.layout.meta or {}), "group_color_kind": grouping})
        traces = fig.data
    for trace in traces:
        if is_dict:
            _tag_dict_trace(trace, use_legendgroup)
            continue
        group = trace.legendgroup if use_legendgroup else trace.name
        if not group:
            continue
        if not use_legendgroup:
            group = re.sub(r" \((?:spoken|written)\)$", "", str(group), flags=re.IGNORECASE)
        if grouping == "vtype":
            _apply_vtype_default(trace, str(group))
        original = trace.to_plotly_json()
        meta = dict(trace.meta) if isinstance(trace.meta, dict) else {}
        meta["group_color"] = {
            "group": str(group),
            "marker": original.get("marker", {}).get("color"),
            "line": original.get("line", {}).get("color"),
            "fill": original.get("fillcolor"),
        }
        trace.meta = meta
    return fig


def _tag_dict_trace(trace, use_legendgroup):
    group = trace.get("legendgroup") if use_legendgroup else trace.get("name")
    if not group:
        return
    if not use_legendgroup:
        group = re.sub(r" \((?:spoken|written)\)$", "", str(group), flags=re.IGNORECASE)
    meta = dict(trace.get("meta") or {})
    meta["group_color"] = {
        "group": str(group),
        "marker": (trace.get("marker") or {}).get("color"),
        "line": (trace.get("line") or {}).get("color"),
        "fill": trace.get("fillcolor"),
    }
    trace["meta"] = meta


def _apply_vtype_default(trace, group):
    """Give uncolored variety-type traces the same ENL/ESL/EFL colors the UMAP uses."""
    default = VTYPE_DEFAULTS.get(re.sub(r"^AI-", "", group))
    if not default:
        return
    if trace.marker is None or trace.marker.color is None:
        trace.update(marker={"color": default})
    if trace.type == "box" and (trace.line is None or trace.line.color is None):
        trace.update(line={"color": default})
