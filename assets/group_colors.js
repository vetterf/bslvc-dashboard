(function () {
    "use strict";

    const defaults = ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
        "#19d3f3", "#ff6692", "#b6e880", "#ff97ff", "#fecb52"];
    const genderDefaults = {Female: "#e377c2", Male: "#1f77b4", "Non-binary": "#2ca02c"};
    const vtypeDefaults = {ENL: "#1f77b4", ESL: "#ff7f0e", EFL: "#2ca02c", Other: "#cccccc"};
    const noUpdate = () => window.dash_clientside.no_update;
    const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const validColor = color => typeof color === "string" && /^#[0-9a-f]{6}$/i.test(color);
    const normalizeGroup = (kind, group) => {
        if (kind !== "gender" || typeof group !== "string") return group;
        const lower = group.toLowerCase();
        return {female: "Female", male: "Male", "non-binary": "Non-binary"}[lower] || group;
    };
    const paletteColor = (palette, kind, group) =>
        (palette && palette[kind] && palette[kind][normalizeGroup(kind, group)]) || null;

    function rgba(color, alpha) {
        const n = parseInt(color.slice(1), 16);
        return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${alpha})`;
    }

    function replaceColor(target, key, color) {
        if (color == null) delete target[key];
        else target[key] = color;
    }

    function recolorGrouped(fig, palette) {
        const kind = fig && fig.layout && fig.layout.meta && fig.layout.meta.group_color_kind;
        if (!kind || !fig.data) return noUpdate();
        let changed = false;
        const data = fig.data.map(trace => {
            const original = trace.meta && trace.meta.group_color;
            if (!original) return trace;
            const color = paletteColor(palette, kind, original.group);
            const copy = {...trace};
            if (trace.marker || original.marker != null || (color && !trace.fill)) {
                copy.marker = {...trace.marker};
                replaceColor(copy.marker, "color", color || original.marker);
                if (!Object.keys(copy.marker).length) delete copy.marker;
            }
            if (trace.line || original.line != null) {
                copy.line = {...trace.line};
                replaceColor(copy.line, "color", color && original.line != null ? color : original.line);
                if (!Object.keys(copy.line).length) delete copy.line;
            }
            if (trace.error_x && trace.error_x.color) {
                copy.error_x = {...trace.error_x, color: color || original.marker};
            }
            if (original.fill != null) {
                const match = String(original.fill).match(/rgba\([^,]+,[^,]+,[^,]+,\s*([\d.]+)\)/i);
                copy.fillcolor = color && match ? rgba(color, match[1]) : original.fill;
            }
            if (!equal(copy, trace)) changed = true;
            return copy;
        });
        return changed ? {...fig, data} : noUpdate();
    }

    function lexicalDisplay(fig, showCi, fadeNx) {
        if (!fig || !fig.data) return noUpdate();
        let changed = false;
        const data = fig.data.map(trace => {
            const meta = trace.meta || {};
            let copy = trace;
            if (meta.ci_band) {
                const visible = showCi !== false;
                if ((trace.visible !== false) !== visible) copy = {...copy, visible};
            }
            if (Array.isArray(meta.nx_opacity)) {
                const opacity = fadeNx === false ? 0.9 : meta.nx_opacity;
                if (!equal(trace.marker && trace.marker.opacity, opacity)) {
                    copy = {...copy, marker: {...copy.marker, opacity}};
                }
            }
            if (copy !== trace) changed = true;
            return copy;
        });
        return changed ? {...fig, data} : noUpdate();
    }

    function varietyType(info, variety) {
        const type = info && typeof info.Variety_Type === "string"
            ? info.Variety_Type.trim().replace(/^AI-/, "") : "";
        if (["ENL", "ESL", "EFL"].includes(type)) return type;
        variety = String(variety || "").replace(/^AI-GPT-/, "");
        if (["US", "England", "Scotland", "Ireland"].includes(variety) ||
            variety.startsWith("England_") || variety.startsWith("England (") ||
            variety.startsWith("Ireland (")) return "ENL";
        if (["Gibraltar", "Malta", "India", "Puerto Rico"].includes(variety)) return "ESL";
        if (["Slovenia", "Germany", "Sweden", "Spain (Balearic Islands)"].includes(variety)) return "EFL";
        return "Other";
    }

    function rowGender(trace, index, info) {
        const row = trace.customdata && trace.customdata[index];
        return normalizeGroup("gender", Array.isArray(row) && row[4] != null ? row[4] : info && info.Gender);
    }

    function recolorUmap(mode, fig, palette, informants) {
        if (!fig || !fig.data || !informants) return noUpdate();
        const byId = Object.fromEntries(informants.map(row => [String(row.InformantID), row]));
        let changed = false;
        const data = fig.data.map(trace => {
            if (trace.meta && trace.meta.kde_contour) {
                const meta = {...trace.meta};
                const original = meta.group_color_contour || {
                    line: trace.line && trace.line.color,
                    colorscale: trace.colorscale,
                };
                meta.group_color_contour = original;
                const copy = {...trace, meta};
                const color = mode === "Variety" && paletteColor(palette, "variety", trace.legendgroup);
                if (original.colorscale && original.line) {
                    copy.line = {...trace.line, color: color || original.line};
                    copy.colorscale = color
                        ? original.colorscale.map(([step, old]) => {
                            const match = String(old).match(/rgba\([^,]+,[^,]+,[^,]+,\s*([\d.]+)\)/i);
                            return [step, match ? rgba(color, match[1]) : old];
                        })
                        : original.colorscale;
                }
                if (!equal(copy, trace)) changed = true;
                return copy;
            }
            if (!Array.isArray(trace.ids) || !trace.marker) return trace;
            const meta = {...(trace.meta || {})};
            const baseline = meta.group_color_umap || trace.marker.color;
            meta.group_color_umap = baseline;
            const colors = trace.ids.map((id, i) => {
                const info = byId[String(id)];
                const variety = trace.name;
                const original = Array.isArray(baseline) ? baseline[i] : baseline;
                if (mode === "Variety") return paletteColor(palette, "variety", variety) || original;
                if (mode === "Gender") {
                    const group = rowGender(trace, i, info);
                    return paletteColor(palette, "gender", group) || genderDefaults[group] || "#cccccc";
                }
                const type = varietyType(info, variety);
                return paletteColor(palette, "vtype", type) || vtypeDefaults[type] || "#cccccc";
            });
            const copy = {...trace, marker: {...trace.marker, color: colors}, meta};
            if (!equal(copy, trace)) changed = true;
            return copy;
        });
        return changed ? {...fig, data} : noUpdate();
    }

    function controls(page, kind, groups, palette) {
        if (!kind) return "Group colors are unavailable for this plot or color setting.";
        const entries = Object.entries(groups).sort(([a], [b]) => a.localeCompare(b));
        if (!entries.length) return "Render a plot to choose colors for its groups.";
        return entries.map(([group, defaultColor]) => ({
            namespace: "dash_mantine_components",
            type: "ColorInput",
            props: {
                id: {type: "group-color-input", page, kind, group, default: defaultColor},
                label: group,
                value: paletteColor(palette, kind, group) || defaultColor,
                format: "hex",
                size: "xs",
                disallowInput: true,
                swatches: [],
            },
        }));
    }

    function groupedControls(page, fig, palette) {
        const kind = fig && fig.layout && fig.layout.meta && fig.layout.meta.group_color_kind;
        const groups = {};
        if (kind && fig.data) fig.data.forEach((trace, index) => {
            const original = trace.meta && trace.meta.group_color;
            if (!original) return;
            const group = normalizeGroup(kind, original.group);
            const color = original.marker || original.line;
            if (!groups[group] && validColor(color)) groups[group] = color;
            else if (!groups[group] && !original.fill) groups[group] = defaults[index % defaults.length];
        });
        return controls(page, kind, groups, palette);
    }

    function grammarControls(plotType, view, mode, umap, item, palette, informants) {
        if (plotType === "item") return groupedControls("grammar", item, palette);
        if (view !== "umap-plot") {
            return "Group colors can only be changed while the UMAP plot is shown. Switch to the UMAP view to edit them; the Comparison plot reuses these colors.";
        }
        if (!["Variety", "Gender", "Variety type"].includes(mode)) {
            return controls("grammar", null, {}, palette);
        }
        const kind = mode === "Variety type" ? "vtype" : mode.toLowerCase();
        const groups = {};
        const byId = Object.fromEntries((informants || []).map(row => [String(row.InformantID), row]));
        ((umap && umap.data) || []).forEach(trace => {
            if (!Array.isArray(trace.ids) || !trace.marker) return;
            const baseline = trace.meta && trace.meta.group_color_umap || trace.marker.color;
            trace.ids.forEach((id, i) => {
                const info = byId[String(id)];
                const group = kind === "variety" ? trace.name
                    : kind === "vtype" ? varietyType(info, trace.name)
                    : rowGender(trace, i, info);
                if (group == null || group === "") return;
                const color = kind === "variety"
                    ? (Array.isArray(baseline) ? baseline[i] : baseline)
                    : kind === "vtype" ? vtypeDefaults[group] || "#cccccc"
                    : genderDefaults[group] || "#cccccc";
                if (validColor(color)) groups[group] = color;
            });
        });
        return controls("grammar", kind, groups, palette);
    }

    function savePalette(values, reset, ids, palette) {
        const context = window.dash_clientside.callback_context;
        if (typeof context.triggered_id === "string" && context.triggered_id.endsWith("-group-color-reset")) {
            return {};
        }
        const triggered = context.triggered_id;
        if (!triggered || typeof triggered !== "object" || typeof triggered.group !== "string") {
            return noUpdate();
        }
        const updated = {...(palette || {})};
        let changed = false;
        (ids || []).forEach((id, i) => {
            if (id.page !== triggered.page || id.kind !== triggered.kind ||
                id.group !== triggered.group || id.default !== triggered.default) return;
            const color = values[i];
            if (!validColor(color)) return;
            const group = normalizeGroup(id.kind, id.group);
            const existing = updated[id.kind] || {};
            if (!existing[group] && color.toLowerCase() === id.default.toLowerCase()) return;
            if (existing[group] === color) return;
            updated[id.kind] = {...existing};
            updated[id.kind][group] = color;
            changed = true;
        });
        return changed ? updated : noUpdate();
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        groupColors: {
            savePalette,
            recolorGrouped,
            recolorUmap,
            lexicalDisplay,
            grammarControls,
            lexicalControls: (fig, palette) => groupedControls("lexical", fig, palette),
        },
    });
}());
