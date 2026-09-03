"""Helper functions for the Lexical Sets analysis page.

Mirrors the structure of grammarFunctions.py but scoped to the LexicalItems
dataset. Kept as a separate module (no Dash Mantine component instances at
module level) so it stays safe to import from background callbacks if needed
in the future - see /memories/repo/dash-background-callbacks.md.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pages.data.retrieve_data as retrieve_data

# Separator used to encode (variety, gender, year) tuples into single tree
# node values. Chosen because it cannot plausibly appear inside a variety,
# gender or year label (unlike "_", which does appear in variety names such
# as "England_North").
TREE_SEP = "|||"

AGE_GROUP_ORDER = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", ">69"]
AGE_GROUP_MIDPOINTS = {
    "<20": 15, "20-29": 24.5, "30-39": 34.5, "40-49": 44.5,
    "50-59": 54.5, "60-69": 64.5, ">69": 75,
}
_AGE_BINS = [-np.inf, 19, 29, 39, 49, 59, 69, np.inf]

GENDER_COLORS = {"Female": "#e6550d", "Male": "#3182bd", "All": "#555555"}

# Sociodemographic columns offered for raw-data export, mirroring the grammar module's export.
LEXICAL_SOCIODEM_COLUMNS = [
    'InformantID', 'Age', 'Gender', 'MainVariety', 'MainVariety_Original', 'AdditionalVarieties',
    'YearsLivedInMainVariety', 'RatioMainVariety', 'CountryCollection', 'Year',
    'Nationality', 'EthnicSelfID', 'CountryID', 'YearsLivedOutside',
    'YearsLivedInside', 'YearsLivedOtherEnglish', 'LanguageHome',
    'LanguageFather', 'LanguageMother', 'Qualifications',
    'QualiMother', 'QualiFather', 'QualiPartner',
    'PrimarySchool', 'SecondarySchool', 'Occupation', 'OccupMother', 'OccupFather',
    'OccupPartner',
]

# Combined qualitative palettes (74 distinct colors) used as a fallback when
# coloring by MainVariety, since the app-wide VarietyColorMap's auto-generated
# fallback only has 10 colors and collides once more than ~10 unmapped
# varieties appear together (the dataset has ~25 varieties in total).
_VARIETY_FALLBACK_PALETTE = (
    list(px.colors.qualitative.Dark24)
    + list(px.colors.qualitative.Light24)
    + list(px.colors.qualitative.Alphabet)
)


def get_variety_color_map(varieties):
    """
    Build a variety -> color map for the given varieties, reusing the
    app-wide fixed colors (retrieve_data.get_color_for_variety) where
    available, and assigning collision-free colors from a large combined
    palette to any remaining (unmapped) varieties.
    """
    base_map = retrieve_data.get_color_for_variety(type="grammar")
    color_map = {}
    used_colors = set()
    for v in varieties:
        if v in base_map.fixed_map:
            color_map[v] = base_map.fixed_map[v]
            used_colors.add(color_map[v])

    fallback_idx = 0
    for v in varieties:
        if v in color_map:
            continue
        while fallback_idx < len(_VARIETY_FALLBACK_PALETTE) and _VARIETY_FALLBACK_PALETTE[fallback_idx] in used_colors:
            fallback_idx += 1
        if fallback_idx < len(_VARIETY_FALLBACK_PALETTE):
            color_map[v] = _VARIETY_FALLBACK_PALETTE[fallback_idx]
            used_colors.add(color_map[v])
            fallback_idx += 1
        else:
            color_map[v] = base_map.get(v)  # extremely unlikely: more varieties than total colors
    return color_map


##############
## Tree builders
##############

def drawParticipantsTreeLexical(informants):
    """
    Build a 3-level participant tree: MainVariety > Gender > Year.
    Unlike the grammar tree, this intentionally stops at Year (no individual
    participant leaves) since there are too many participants (>6000) for a
    fully nested tree to be practical in the DOM.
    """
    data = informants.loc[:, ['InformantID', 'MainVariety', 'Gender', 'Year']].copy()
    data['MainVariety'] = data['MainVariety'].fillna('Unknown')
    data['Gender'] = data['Gender'].fillna('Unknown')
    data['Year'] = data['Year'].fillna('Unknown').astype(str)

    treeData = []
    for variety in sorted(data['MainVariety'].unique()):
        vdata = data[data['MainVariety'] == variety]
        gender_children = []
        for gender in sorted(vdata['Gender'].unique()):
            gdata = vdata[vdata['Gender'] == gender]
            year_children = []
            for year in sorted(gdata['Year'].unique()):
                ydata = gdata[gdata['Year'] == year]
                year_children.append({
                    'value': f"{variety}{TREE_SEP}{gender}{TREE_SEP}{year}",
                    'label': f"{year}  ({len(ydata)})",
                })
            gender_children.append({
                'value': f"{variety}{TREE_SEP}{gender}",
                'label': f"{gender}  ({len(gdata)})",
                'children': year_children,
            })
        treeData.append({
            'value': variety,
            'label': f"{variety}  ({len(vdata)})",
            'children': gender_children,
        })
    return [{
        'value': 'participantslexical',
        'label': f'All Participants  ({len(data)})',
        'children': treeData,
    }]


def get_participants_from_tree_selection(checked_values, informants):
    """Translate checked MainVariety/Gender/Year tree values into an InformantID list."""
    if not checked_values:
        return []

    selection_data = informants.loc[:, ['InformantID', 'MainVariety', 'Gender', 'Year']].copy()
    selection_data['MainVariety'] = selection_data['MainVariety'].fillna('Unknown')
    selection_data['Gender'] = selection_data['Gender'].fillna('Unknown')
    selection_data['Year'] = selection_data['Year'].fillna('Unknown').astype(str)

    mask = pd.Series(False, index=selection_data.index)
    for val in checked_values:
        if val == 'participantslexical':
            continue
        parts = val.split(TREE_SEP)
        if len(parts) == 1:
            mask |= (selection_data['MainVariety'] == parts[0])
        elif len(parts) == 2:
            mask |= (selection_data['MainVariety'] == parts[0]) & (selection_data['Gender'] == parts[1])
        elif len(parts) == 3:
            mask |= (
                (selection_data['MainVariety'] == parts[0])
                & (selection_data['Gender'] == parts[1])
                & (selection_data['Year'] == parts[2])
            )
    return selection_data.loc[mask, 'InformantID'].tolist()


def drawLexicalItemsTree(lexicalMeta):
    """Flat tree of lexical items under one wrapper node, sorted alphabetically by label."""
    meta = lexicalMeta.sort_values('axis_label', key=lambda s: s.str.lower())
    children = [
        {'value': row['column'], 'label': row['axis_label']}
        for _, row in meta.iterrows()
    ]
    return [{
        'value': 'lexicalitems',
        'label': f'Lexical Items  ({len(children)})',
        'children': children,
    }]


def normalize_lexical_tree_selection(checked_participants, checked_items, informants, all_items):
    """Expand top-level/'select all' tree selections into concrete IDs."""
    if checked_participants in (['participantslexical'], None, []):
        participants = informants['InformantID'].tolist()
    else:
        participants = get_participants_from_tree_selection(checked_participants, informants)

    if not checked_items or checked_items == ['lexicalitems']:
        items = list(all_items)
    else:
        items = [c for c in checked_items if c != 'lexicalitems']

    return participants, items


##############
## Data preparation
##############

# "Apparent time" (age) x-axis, displayed oldest (left) to youngest (right).
AGE_GROUP_ORDER = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", ">69"]
AGE_GROUP_DISPLAY_ORDER = list(reversed(AGE_GROUP_ORDER))
AGE_GROUP_MIDPOINTS = {
    "<20": 15, "20-29": 24.5, "30-39": 34.5, "40-49": 44.5,
    "50-59": 54.5, "60-69": 64.5, ">69": 75,
}
_AGE_BINS = [-np.inf, 19, 29, 39, 49, 59, 69, np.inf]

FACET_MIN_ROW_HEIGHT = 220  # px, keeps each facet readable even with many items
FACET_MAX_WIDTH = 350  # px, caps how wide a single facet column can grow


def get_age_group(age_series):
    return pd.cut(age_series, bins=_AGE_BINS, labels=AGE_GROUP_ORDER)


def _birth_decade_label(decade_start):
    return f"{int(decade_start)}-{int(decade_start) + 9}"


def get_x_axis_config(mode, df=None):
    """
    Returns the x-axis configuration for the requested time-axis mode.

    - 'apparent_time': fixed age-group bins, displayed oldest (left) to
      youngest (right).
    - 'birth_year': 10-year birth-year cohorts (Year of data collection minus
      Age), derived dynamically from the data actually present, displayed in
      chronological order (earliest/oldest cohort left).

    `group_midpoints` values are only used internally to order the smoother
    used for facet ordering; for 'birth_year' they are negated so that,
    consistently with 'apparent_time', a HIGHER value always means "older".
    """
    if mode == 'birth_year':
        decade_starts = sorted(df['XGroupSortKey'].dropna().unique()) if df is not None else []
        group_order = [_birth_decade_label(d) for d in decade_starts]
        group_midpoints = {_birth_decade_label(d): -d for d in decade_starts}
        return {
            'group_order': group_order,
            'group_midpoints': group_midpoints,
            'axis_title': "Birth cohort",
            'hover_label': "Birth cohort",
        }
    return {
        'group_order': AGE_GROUP_DISPLAY_ORDER,
        'group_midpoints': AGE_GROUP_MIDPOINTS,
        'axis_title': "Age group",
        'hover_label': "Age group",
    }


def _classify_lexical_value(value):
    if pd.isna(value):
        return 'missing'
    s = str(value).strip()
    if not s:
        return 'missing'
    if s.upper() in ('NX', 'NXC'):
        return 'nx'
    try:
        float(s)
        return 'numeric'
    except ValueError:
        return 'missing'


def prepare_lexical_long_data(lexical_raw, informants, items, participant_ids, mode='apparent_time',
                               exclude_small_cohorts=False, min_cohort_size=5):
    """
    Build a long-format dataframe (one row per participant x item) filtered to
    the selected participants and items, plus the resulting x-axis config.

    Participants with 'ND' ("never had the chance to answer") on ANY of the
    selected items are dropped entirely, so all facets share the same
    participant base. NX/NXC ("uses neither") and literal missing/NULL values
    are kept but excluded from the numeric mean.

    `exclude_small_cohorts` only applies in 'birth_year' mode: birth-year
    cohorts with fewer than `min_cohort_size` participants (in the current
    selection) are dropped from the x-axis entirely, for all items/series.

    Returns (long_df, x_axis_config). long_df is empty if there is nothing to plot.
    """
    empty_config = get_x_axis_config(mode)
    if not participant_ids or not items:
        return pd.DataFrame(), empty_config

    items = [i for i in items if i in lexical_raw.columns]
    if not items:
        return pd.DataFrame(), empty_config

    lex = lexical_raw.loc[lexical_raw['InformantID'].isin(participant_ids), ['InformantID'] + items].copy()
    if lex.empty:
        return pd.DataFrame(), empty_config

    is_nd_any = (lex[items] == 'ND').any(axis=1)
    lex = lex.loc[~is_nd_any]
    if lex.empty:
        return pd.DataFrame(), empty_config

    informant_cols = informants.loc[:, ['InformantID', 'Age', 'Gender', 'MainVariety', 'Year']].copy()
    informant_cols['Age'] = pd.to_numeric(informant_cols['Age'], errors='coerce')
    informant_cols['Year'] = pd.to_numeric(informant_cols['Year'], errors='coerce')

    merged = lex.merge(informant_cols, on='InformantID', how='left')

    if mode == 'birth_year':
        birth_year = merged['Year'] - merged['Age']
        decade_start = np.floor(birth_year / 10) * 10
        merged['XGroupSortKey'] = decade_start
        merged['XGroup'] = decade_start.apply(lambda d: _birth_decade_label(d) if pd.notna(d) else None)
    else:
        merged['XGroup'] = get_age_group(merged['Age'])

    # Rows without a usable age/birth-year cannot be placed in any facet x-position.
    merged = merged.dropna(subset=['XGroup'])
    if merged.empty:
        return pd.DataFrame(), empty_config

    if mode == 'birth_year' and exclude_small_cohorts:
        cohort_sizes = merged.groupby('XGroup')['InformantID'].nunique()
        valid_cohorts = cohort_sizes[cohort_sizes >= min_cohort_size].index
        merged = merged[merged['XGroup'].isin(valid_cohorts)]
        if merged.empty:
            return pd.DataFrame(), empty_config

    x_config = get_x_axis_config(mode, merged)

    long_df = merged.melt(
        id_vars=['InformantID', 'Age', 'Gender', 'MainVariety', 'XGroup'],
        value_vars=items, var_name='Item', value_name='RawValue',
    )
    long_df['ValueType'] = long_df['RawValue'].apply(_classify_lexical_value)
    long_df['NumericValue'] = pd.to_numeric(long_df['RawValue'], errors='coerce')
    long_df.loc[long_df['ValueType'] != 'numeric', 'NumericValue'] = np.nan
    return long_df, x_config


def aggregate_lexical_facets(long_df, series_by='gender'):
    """
    Aggregate mean rating, 95% CI, N, NX share/count and NA (missing) count
    per Item x XGroup x Series. `series_by` controls what 'Series' is:
    - 'none': a single 'All' series (no separation)
    - 'gender': Female / Male
    - 'variety': MainVariety
    """
    out_cols = ['Item', 'XGroup', 'Series', 'Mean', 'CILower', 'CIUpper', 'N', 'NXCount', 'NACount', 'NXShare']
    if long_df.empty:
        return pd.DataFrame(columns=out_cols)

    df = long_df.copy()
    if series_by == 'gender':
        df = df[df['Gender'].isin(['Female', 'Male'])]
        df['Series'] = df['Gender']
    elif series_by == 'variety':
        df['Series'] = df['MainVariety']
    else:
        df['Series'] = 'All'
    group_cols = ['Item', 'XGroup', 'Series']

    if df.empty:
        return pd.DataFrame(columns=out_cols)

    rows = []
    for keys, g in df.groupby(group_cols, observed=True):
        item, x_group, series = keys
        numeric_vals = g.loc[g['ValueType'] == 'numeric', 'NumericValue']
        n = int(numeric_vals.count())
        mean = numeric_vals.mean() if n > 0 else np.nan
        if n > 1:
            se = numeric_vals.std(ddof=1) / np.sqrt(n)
            ci_lower, ci_upper = mean - 1.96 * se, mean + 1.96 * se
        else:
            ci_lower, ci_upper = np.nan, np.nan
        nx_count = int((g['ValueType'] == 'nx').sum())
        na_count = int((g['ValueType'] == 'missing').sum())
        nx_share = nx_count / (n + nx_count) if (n + nx_count) > 0 else np.nan
        rows.append({
            'Item': item, 'XGroup': x_group, 'Series': series,
            'Mean': mean, 'CILower': ci_lower, 'CIUpper': ci_upper,
            'N': n, 'NXCount': nx_count, 'NACount': na_count, 'NXShare': nx_share,
        })
    return pd.DataFrame(rows)


##############
## Trend ordering (LOESS smoother)
##############

def _lowess_smooth(x, y, frac=0.6):
    """Minimal locally-weighted regression smoother (tricube weights), no external deps."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return y.copy()

    r = max(2, int(np.ceil(frac * n)))
    y_smooth = np.zeros(n)
    for i in range(n):
        dist = np.abs(x - x[i])
        idx = np.argsort(dist)[:r]
        d = dist[idx]
        max_d = d.max() if d.max() > 0 else 1.0
        w = np.clip(1 - (d / max_d) ** 3, 0, None) ** 3
        X = np.vstack([np.ones(r), x[idx]]).T
        Wd = np.diag(w)
        try:
            beta, *_ = np.linalg.lstsq(Wd @ X, Wd @ y[idx], rcond=None)
            y_smooth[i] = beta[0] + beta[1] * x[i]
        except np.linalg.LinAlgError:
            y_smooth[i] = np.average(y[idx], weights=w)
    return y_smooth


def compute_item_trend_slope(agg_df, item, group_midpoints):
    """
    Slope of a LOESS-smoothed curve of mean rating vs. an age score
    (group_midpoints, where a higher value always means "older"), combining
    both genders (weighted by N per group).
    """
    sub = agg_df[agg_df['Item'] == item]
    if sub.empty:
        return 0.0

    combined = (
        sub.groupby('XGroup', observed=True)
        .apply(lambda g: np.average(g['Mean'], weights=g['N']) if g['N'].sum() > 0 else np.nan)
    )
    sorted_groups = sorted(group_midpoints.keys(), key=lambda g: group_midpoints[g])
    combined = combined.reindex(sorted_groups).dropna()
    if len(combined) < 2:
        return 0.0

    x = np.array([group_midpoints[g] for g in combined.index])
    y = combined.values
    order = np.argsort(x)
    x, y = x[order], y[order]

    y_smooth = _lowess_smooth(x, y)
    return (y_smooth[-1] - y_smooth[0]) / (x[-1] - x[0])


def order_items_by_trend(agg_df, items, group_midpoints):
    """
    Sort items so the top-left facet is the one with the steepest "oldest
    lowest, youngest highest" trend, i.e. rating decreasing as "oldness"
    (group_midpoints) increases - ascending sort of that slope.
    """
    slopes = {item: compute_item_trend_slope(agg_df, item, group_midpoints) for item in items}
    return sorted(items, key=lambda it: slopes[it])


def order_items_alphabetically(items, item_meta_map):
    return sorted(items, key=lambda it: item_meta_map.get(it, {}).get('axis_label', it).lower())


def compute_item_overall_mean(long_df, item):
    """
    Overall mean rating for an item, pooling every selected participant's
    numeric rating regardless of age/birth-year group, gender or MainVariety.
    Since each participant contributes one value, larger subgroups naturally
    carry proportionally more weight - i.e. this is already the weighted
    average across varieties and genders the sort is meant to reflect.
    """
    vals = long_df.loc[(long_df['Item'] == item) & (long_df['ValueType'] == 'numeric'), 'NumericValue']
    return vals.mean() if len(vals) else np.nan


def order_items_by_average(long_df, items, descending=True):
    """Sort items by their overall weighted-average rating (items with no data sort last)."""
    averages = {item: compute_item_overall_mean(long_df, item) for item in items}
    return sorted(
        items,
        key=lambda it: (np.isnan(averages[it]), -averages[it] if descending else averages[it]),
    )


def order_items(items, sort_by='trend', agg_df=None, group_midpoints=None, item_meta_map=None, long_df=None):
    """Dispatch to the requested facet-ordering strategy: 'trend', 'alphabetical' or 'average'."""
    if sort_by == 'alphabetical':
        return order_items_alphabetically(items, item_meta_map or {})
    if sort_by == 'average':
        return order_items_by_average(long_df if long_df is not None else pd.DataFrame(columns=['Item', 'ValueType', 'NumericValue']), items)
    return order_items_by_trend(agg_df, items, group_midpoints or {})


##############
## Plot building
##############

# Y-axis padding beyond the -2..2 rating scale so markers at the extremes aren't clipped.
Y_AXIS_PADDING = 0.2


def _scale_marker_sizes(counts, scale="sqrt", min_size=1, max_size=15):
    """
    Scale marker sizes using logarithmic, square-root, or linear scaling.

    Parameters
    ----------
    counts : array-like
        Marker counts. Non-positive and NaN values produce hidden markers.
    scale : {"log", "sqrt", "linear"}, default="log"
        Scaling method.
    min_size : float, default=1
        Size assigned to the smallest positive count.
    max_size : float, default=15
        Size assigned to the largest count.

    Returns
    -------
    numpy.ndarray
        Scaled marker sizes. Markers with non-positive counts are set to 0.
    """
    if scale not in {"log", "sqrt", "linear"}:
        raise ValueError("scale must be one of: 'log', 'sqrt', or 'linear'")

    counts = np.asarray(counts, dtype=float)
    counts = np.nan_to_num(counts, nan=0.0)
    counts = np.maximum(counts, 0.0)

    max_count = (
        counts.max()
        if counts.size and counts.max() > 0
        else 1.0
    )

    if scale == "log":
        relative = np.log1p(counts) / np.log1p(max_count)
    elif scale == "sqrt":
        relative = np.sqrt(counts / max_count)
    else:  # linear
        relative = counts / max_count

    scaled = min_size + (max_size - min_size) * relative
    scaled[counts <= 0] = 0  # hide markers where there is no data

    return scaled

def _scale_nx_opacity(nx_share, min_opacity=0.2):
    """Higher NX share (more 'uses neither' relative to numeric ratings) -> more transparent marker."""
    nx_share = np.nan_to_num(np.asarray(nx_share, dtype=float), nan=0.0)
    opacity = 1.0 - 2.0 * np.clip(nx_share, 0, 1)
    return np.clip(opacity, min_opacity, 1.0)


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def build_lexical_facet_plot(agg_df, items_ordered, item_meta_map, x_axis_config,
                              series_by='gender', facet_cols=4,
                              show_ci=False, encode_nx_opacity=False):
    group_order = x_axis_config['group_order']
    hover_label = x_axis_config['hover_label']

    n_items = len(items_ordered)
    if n_items == 0 or agg_df.empty:
        fig = go.Figure()
        fig.update_layout(template="simple_white", annotations=[{
            "text": "No data available for the current selection.",
            "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16},
        }])
        return fig

    ncols = min(facet_cols, n_items)
    nrows = int(np.ceil(n_items / ncols))
    subplot_titles = [item_meta_map.get(it, {}).get('axis_label', it) for it in items_ordered]

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles,
                         shared_yaxes=True, horizontal_spacing=0.04,
                         vertical_spacing=0.6 / max(nrows, 1))

    if series_by == 'gender':
        series_list = ['Female', 'Male']
        series_color_map = GENDER_COLORS
    elif series_by == 'variety':
        series_list = sorted(agg_df['Series'].dropna().unique())
        series_color_map = get_variety_color_map(series_list)
    else:
        series_list = ['All']
        series_color_map = {'All': '#1f77b4'}
    legend_shown = set()

    for i, item in enumerate(items_ordered):
        row, col = i // ncols + 1, i % ncols + 1
        item_df = agg_df[agg_df['Item'] == item]
        meta = item_meta_map.get(item, {})
        american = meta.get('american', item)
        british = meta.get('british', item)
        axis_label = meta.get('axis_label', item)

        for series in series_list:
            g_df = (
                item_df[item_df['Series'] == series]
                .set_index('XGroup')
                .reindex(group_order)
            )
            y = g_df['Mean'].values
            n = g_df['N'].fillna(0).values
            nx = g_df['NXCount'].fillna(0).values
            na = g_df['NACount'].fillna(0).values
            nx_share = g_df['NXShare'].values
            color = series_color_map.get(series, '#555555')

            if show_ci:
                valid = g_df['Mean'].notna() & g_df['CILower'].notna() & g_df['CIUpper'].notna()
                if valid.any():
                    x_valid = [grp for grp, ok in zip(group_order, valid) if ok]
                    lower_valid = g_df.loc[valid, 'CILower'].values
                    upper_valid = g_df.loc[valid, 'CIUpper'].values
                    fig.add_trace(go.Scatter(
                        x=x_valid + x_valid[::-1],
                        y=np.concatenate([upper_valid, lower_valid[::-1]]),
                        mode='lines',  # no markers - keep the band, not dots at its edges
                        fill='toself',
                        fillcolor=_hex_to_rgba(color, 0.08),
                        line=dict(width=0),
                        hoverinfo='skip',
                        showlegend=False,
                        legendgroup=series,
                    ), row=row, col=col)

            marker_opacity = _scale_nx_opacity(nx_share) if encode_nx_opacity else 0.9
            customdata = np.stack([n, nx, na, np.nan_to_num(nx_share, nan=0.0) * 100], axis=-1)

            fig.add_trace(go.Scatter(
                x=group_order, y=y, mode='lines+markers',
                name=series,
                legendgroup=series,
                showlegend=series not in legend_shown,
                line=dict(color=color),
                marker=dict(size=_scale_marker_sizes(n), color=color, opacity=marker_opacity),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{axis_label}</b><br>"
                    f"American: {american}<br>British: {british}<br>"
                    f"{hover_label}: " + "%{x}<br>"
                    "Mean rating: %{y:.2f}<br>"
                    "N (mean based on): %{customdata[0]:.0f}<br>"
                    "Uses neither (NX): %{customdata[1]:.0f} (%{customdata[3]:.0f}%)<br>"
                    "Missing (NA): %{customdata[2]:.0f}"
                    "<extra></extra>"
                ),
            ), row=row, col=col)
            legend_shown.add(series)

    fig.update_yaxes(
        range=[-2 - Y_AXIS_PADDING, 2 + Y_AXIS_PADDING],
        tickvals=[-2, -1, 0, 1, 2],
        zeroline=True, zerolinewidth=1, zerolinecolor='#cccccc',
    )
    fig.update_xaxes(tickangle=45, categoryorder='array', categoryarray=group_order)
    fig.update_layout(
        template="simple_white",
        height=max(FACET_MIN_ROW_HEIGHT, FACET_MIN_ROW_HEIGHT * nrows),
        #width=ncols * FACET_MAX_WIDTH,
        autosize=True,
        margin=dict(t=60, b=20, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12)
    return fig


def compute_lexical_data(lexical_raw, informants, participant_ids, items, mode='apparent_time',
                          series_by='gender', sort_by='trend', item_meta_map=None,
                          exclude_small_cohorts=False, min_cohort_size=5):
    """Filter/aggregate the data and order facets ('trend', 'alphabetical' or 'average'), without building a figure."""
    long_df, x_config = prepare_lexical_long_data(
        lexical_raw, informants, items, participant_ids, mode=mode,
        exclude_small_cohorts=exclude_small_cohorts, min_cohort_size=min_cohort_size,
    )
    if long_df.empty:
        return long_df, pd.DataFrame(), [], x_config

    agg_df = aggregate_lexical_facets(long_df, series_by=series_by)
    present_items = [it for it in items if it in agg_df['Item'].unique()]
    ordered_items = order_items(present_items, sort_by=sort_by, agg_df=agg_df,
                                 group_midpoints=x_config['group_midpoints'], item_meta_map=item_meta_map,
                                 long_df=long_df)
    return long_df, agg_df, ordered_items, x_config


def compute_lexical_plot(lexical_raw, informants, participant_ids, items, item_meta_map,
                          series_by='gender', facet_cols=4, mode='apparent_time',
                          show_ci=False, encode_nx_opacity=False, sort_by='trend',
                          exclude_small_cohorts=False, min_cohort_size=5):
    """End-to-end: filter/aggregate data, order facets, and build the plot."""
    _, agg_df, ordered_items, x_config = compute_lexical_data(
        lexical_raw, informants, participant_ids, items, mode=mode,
        series_by=series_by, sort_by=sort_by, item_meta_map=item_meta_map,
        exclude_small_cohorts=exclude_small_cohorts, min_cohort_size=min_cohort_size,
    )
    return build_lexical_facet_plot(agg_df, ordered_items, item_meta_map, x_config,
                                     series_by=series_by, facet_cols=facet_cols,
                                     show_ci=show_ci, encode_nx_opacity=encode_nx_opacity)


##############
## Averaged across items plot
##############

def compute_average_lexical_data(long_df, series_by='gender'):
    """
    Aggregate mean rating (item-weighted average: equal weight per item regardless of N)
    per XGroup x Series, averaging across all lexical items.
    
    For each item x XGroup x Series combination, compute the mean. Then average these
    means across all items, giving equal weight to each item regardless of data point count.
    """
    out_cols = ['XGroup', 'Series', 'Mean', 'CILower', 'CIUpper', 'N', 'NXCount', 'NACount', 'NXShare']
    if long_df.empty:
        return pd.DataFrame(columns=out_cols)

    df = long_df.copy()
    if series_by == 'gender':
        df = df[df['Gender'].isin(['Female', 'Male'])]
        df['Series'] = df['Gender']
    elif series_by == 'variety':
        df['Series'] = df['MainVariety']
    else:
        df['Series'] = 'All'
    
    # First aggregate by Item x XGroup x Series
    item_agg_cols = ['Item', 'XGroup', 'Series']
    item_rows = []
    for keys, g in df.groupby(item_agg_cols, observed=True):
        item, x_group, series = keys
        numeric_vals = g.loc[g['ValueType'] == 'numeric', 'NumericValue']
        n = int(numeric_vals.count())
        mean = numeric_vals.mean() if n > 0 else np.nan
        nx_count = int((g['ValueType'] == 'nx').sum())
        na_count = int((g['ValueType'] == 'missing').sum())
        item_rows.append({
            'Item': item, 'XGroup': x_group, 'Series': series,
            'Mean': mean, 'N': n, 'NXCount': nx_count, 'NACount': na_count,
        })
    
    item_agg_df = pd.DataFrame(item_rows)
    if item_agg_df.empty:
        return pd.DataFrame(columns=out_cols)
    
    # Now average across items for each XGroup x Series (item-level weighting)
    rows = []
    for keys, g in item_agg_df.groupby(['XGroup', 'Series'], observed=True):
        x_group, series = keys
        # Equal weight per item: only use items with valid means
        valid_means = g[g['Mean'].notna()]['Mean'].values
        if len(valid_means) == 0:
            continue
        
        mean = float(np.mean(valid_means))
        # For CI: use standard error across item means
        if len(valid_means) > 1:
            se = np.std(valid_means, ddof=1) / np.sqrt(len(valid_means))
            ci_lower, ci_upper = mean - 1.96 * se, mean + 1.96 * se
        else:
            ci_lower, ci_upper = np.nan, np.nan
        
        # Aggregate counts from the items
        n = int(g['N'].sum())
        nx_count = int(g['NXCount'].sum())
        na_count = int(g['NACount'].sum())
        nx_share = nx_count / (n + nx_count) if (n + nx_count) > 0 else np.nan
        
        rows.append({
            'XGroup': x_group, 'Series': series,
            'Mean': mean, 'CILower': ci_lower, 'CIUpper': ci_upper,
            'N': n, 'NXCount': nx_count, 'NACount': na_count, 'NXShare': nx_share,
        })
    return pd.DataFrame(rows)


def build_average_lexical_plot(agg_df, x_axis_config, series_by='gender', show_ci=False, encode_nx_opacity=False):
    """Build a single line plot of averaged lexical ratings across all items."""
    group_order = x_axis_config['group_order']
    hover_label = x_axis_config['hover_label']
    
    if agg_df.empty:
        fig = go.Figure()
        fig.update_layout(template="simple_white", annotations=[{
            "text": "No data available for the current selection.",
            "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16},
        }])
        return fig
    
    if series_by == 'gender':
        series_list = ['Female', 'Male']
        series_color_map = GENDER_COLORS
    elif series_by == 'variety':
        series_list = sorted(agg_df['Series'].dropna().unique())
        series_color_map = get_variety_color_map(series_list)
    else:
        series_list = ['All']
        series_color_map = {'All': '#1f77b4'}
    
    fig = go.Figure()
    legend_shown = set()
    
    for series in series_list:
        s_df = agg_df[agg_df['Series'] == series].set_index('XGroup').reindex(group_order)
        y = s_df['Mean'].values
        n = s_df['N'].fillna(0).values
        nx = s_df['NXCount'].fillna(0).values
        nx_share = s_df['NXShare'].values
        color = series_color_map.get(series, '#555555')
        
        if show_ci:
            valid = s_df['Mean'].notna() & s_df['CILower'].notna() & s_df['CIUpper'].notna()
            if valid.any():
                x_valid = [grp for grp, ok in zip(group_order, valid) if ok]
                lower_valid = s_df.loc[valid, 'CILower'].values
                upper_valid = s_df.loc[valid, 'CIUpper'].values
                fig.add_trace(go.Scatter(
                    x=x_valid + x_valid[::-1],
                    y=np.concatenate([upper_valid, lower_valid[::-1]]),
                    mode='lines', fill='toself',
                    fillcolor=_hex_to_rgba(color, 0.08),
                    line=dict(width=0),
                    hoverinfo='skip', showlegend=False, legendgroup=series,
                ))
        
        marker_opacity = _scale_nx_opacity(nx_share) if encode_nx_opacity else 0.9
        customdata = np.stack([n, nx, np.nan_to_num(nx_share, nan=0.0) * 100], axis=-1)
        
        fig.add_trace(go.Scatter(
            x=group_order, y=y, mode='lines+markers',
            name=series, legendgroup=series,
            showlegend=series not in legend_shown,
            line=dict(color=color),
            marker=dict(size=_scale_marker_sizes(n), color=color, opacity=marker_opacity),
            customdata=customdata,
            hovertemplate=(
                f"<b>Average across all items</b><br>"
                f"{hover_label}: %{{x}}<br>"
                "Mean rating: %{y:.2f}<br>"
                "N (mean based on): %{customdata[0]:.0f}<br>"
                "Uses neither (NX): %{customdata[1]:.0f} (%{customdata[2]:.0f}%)"
                "<extra></extra>"
            ),
        ))
        legend_shown.add(series)
    
    fig.update_yaxes(
        range=[-2 - Y_AXIS_PADDING, 2 + Y_AXIS_PADDING],
        tickvals=[-2, -1, 0, 1, 2],
        zeroline=True, zerolinewidth=1, zerolinecolor='#cccccc',
    )
    fig.update_xaxes(tickangle=45, categoryorder='array', categoryarray=group_order)
    fig.update_layout(
        template="simple_white", height=400, autosize=True,
        margin=dict(t=40, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=x_axis_config.get('axis_title', 'Age group'),
        yaxis_title='Mean rating (averaged across items)',
    )
    return fig


##############
## Heatmap by item and variety
##############

def _linear_regression_slope(x, y):
    """
    Compute linear regression slope (trend). x and y should be arrays/lists.
    Returns slope, or NaN if insufficient valid data.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    
    if valid.sum() < 2:
        return np.nan
    
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]
    
    # Fit y = a + b*x
    coeffs = np.polyfit(x_valid, y_valid, 1)
    return float(coeffs[0])  # slope


def _normalize_lexical_rating_for_dist(value):
    """Normalize lexical responses into rating buckets for distribution counting."""
    if pd.isna(value):
        return "Null"
    
    value_str = str(value).strip()
    if not value_str:
        return "Null"
    
    if value_str.upper() == "NX":
        return "NX"
    
    try:
        numeric_value = float(value_str)
        if numeric_value in {-2.0, -1.0, 0.0, 1.0, 2.0}:
            return str(int(numeric_value))
    except (TypeError, ValueError):
        pass
    
    return "Null"


def _get_rating_distribution_for_cell(series):
    """Count occurrences of each rating value in a series."""
    distribution = {"-2": 0, "-1": 0, "0": 0, "1": 0, "2": 0, "NX": 0, "Null": 0}
    for value in series:
        bucket = _normalize_lexical_rating_for_dist(value)
        distribution[bucket] += 1
    return distribution


def compute_lexical_heatmap_by_item(long_df, x_axis_config, color_by_trend=False, item_meta_map=None, sort_items='average'):
    """
    Build a heatmap with items on Y-axis and varieties on X-axis.
    
    If color_by_trend=False:
      - Aggregates across ALL age groups (pools them)
      - Colors cells by mean rating value
      - X-axis: varieties (sorted by global average)
      - Y-axis: items (sorted by average across varieties)
    
    If color_by_trend=True:
      - Computes trend (slope) for each item within each variety across age groups
      - Colors cells by trend value (red=downward, blue=upward)
      - X-axis: varieties
      - Requires the x_axis_config to specify age groups for trend calculation
    
    sort_items: 'alphabetically', 'average' (default), or 'slope' (for trend mode only)
    
    Returns a figure.
    """
    if long_df.empty:
        fig = go.Figure()
        fig.update_layout(template="simple_white", annotations=[{
            "text": "No data available for the current selection.",
            "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16},
        }])
        return fig
    
    if item_meta_map is None:
        item_meta_map = {}
    
    df = long_df.copy()
    group_order = x_axis_config.get('group_order', [])
    
    if color_by_trend:
        # Mode 1: Trend across age groups, by variety x item
        # Aggregate by item x variety x age_group
        item_variety_group_rows = []
        for keys, g in df.groupby(['Item', 'MainVariety', 'XGroup'], observed=True):
            item, variety, x_group = keys
            numeric_vals = g.loc[g['ValueType'] == 'numeric', 'NumericValue']
            n = int(numeric_vals.count())
            mean = numeric_vals.mean() if n > 0 else np.nan
            item_variety_group_rows.append({
                'Item': item, 'Variety': variety, 'XGroup': x_group, 'Mean': mean, 'N': n,
            })
        
        item_variety_group_df = pd.DataFrame(item_variety_group_rows)
        if item_variety_group_df.empty:
            fig = go.Figure()
            fig.update_layout(template="simple_white", annotations=[{
                "text": "No data available for the current selection.",
                "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16},
            }])
            return fig
        
        # Compute trend for each item x variety
        # First get all unique items and varieties
        unique_items = sorted(item_variety_group_df['Item'].unique())
        unique_varieties = sorted(item_variety_group_df['Variety'].unique())
        
        z_trends = np.full((len(unique_items), len(unique_varieties)), np.nan)
        z_means = np.full((len(unique_items), len(unique_varieties)), np.nan)
        
        for i, item in enumerate(unique_items):
            for j, variety in enumerate(unique_varieties):
                item_variety_data = item_variety_group_df[
                    (item_variety_group_df['Item'] == item) & 
                    (item_variety_group_df['Variety'] == variety)
                ]
                
                if not item_variety_data.empty:
                    # Map group labels to positions
                    group_positions = {g: idx for idx, g in enumerate(group_order)}
                    available_groups = [g for g in group_order if g in item_variety_data['XGroup'].values]
                    
                    if len(available_groups) >= 2:
                        x_vals = [group_positions[g] for g in available_groups]
                        means = []
                        for g in available_groups:
                            m = item_variety_data[item_variety_data['XGroup'] == g]['Mean'].values
                            if len(m) > 0:
                                means.append(float(m[0]))
                        
                        if len(means) >= 2:
                            slope = _linear_regression_slope(x_vals, means)
                            z_trends[i, j] = slope
                            # Also store mean for display
                            z_means[i, j] = np.mean(means)
        
        # Sort items based on sort_items parameter
        if sort_items == 'alphabetically':
            sorted_items = sorted(unique_items)
        elif sort_items == 'slope':
            # Sort by signed trend (descending), so upward and downward trends differ
            item_avg_trends = np.nanmean(z_trends, axis=1)
            item_indices = np.argsort(-item_avg_trends)
            sorted_items = [unique_items[idx] for idx in item_indices if not np.isnan(item_avg_trends[idx])]
            sorted_items.extend([unique_items[idx] for idx in item_indices if np.isnan(item_avg_trends[idx])])
        else:  # default: 'average'
            # Sort by average trend/mean (descending)
            item_avg_trends = np.nanmean(z_trends, axis=1)
            item_indices = np.argsort(-np.abs(item_avg_trends))  # Sort by abs magnitude, descending
            sorted_items = [unique_items[idx] for idx in item_indices if not np.isnan(item_avg_trends[idx])]
            sorted_items.extend([unique_items[idx] for idx in item_indices if np.isnan(item_avg_trends[idx])])
        
        # Sort varieties by average trend across items
        variety_avg_trends = np.nanmean(z_trends, axis=0)
        variety_indices = np.argsort(-np.abs(variety_avg_trends))
        sorted_varieties = [unique_varieties[idx] for idx in variety_indices if not np.isnan(variety_avg_trends[idx])]
        sorted_varieties.extend([unique_varieties[idx] for idx in variety_indices if np.isnan(variety_avg_trends[idx])])
        
        # Reorder arrays using fancy indexing
        item_idx_map = {item: i for i, item in enumerate(unique_items)}
        variety_idx_map = {var: j for j, var in enumerate(unique_varieties)}
        item_indices = np.array([item_idx_map[it] for it in sorted_items])
        variety_indices = np.array([variety_idx_map[v] for v in sorted_varieties])
        z_trends_sorted = z_trends[np.ix_(item_indices, variety_indices)]
        z_means_sorted = z_means[np.ix_(item_indices, variety_indices)]
        
        # Create heatmap colored by trend
        fig = go.Figure(data=go.Heatmap(
            z=z_trends_sorted,
            x=sorted_varieties,
            y=sorted_items,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="Trend<br>(slope across<br>age groups)"),
            hovertemplate='<b>Item: %{y}</b><br><b>Variety: %{x}</b><br>Trend (slope): %{z:.3f}<extra></extra>',
        ))
        
    else:
        # Mode 2: Aggregate across all age groups, color by mean value
        # Aggregate by item x variety (pool all age groups)
        item_variety_rows = []
        raw_data_cache = {}  # Cache raw values for hover
        
        for keys, g in df.groupby(['Item', 'MainVariety'], observed=True):
            item, variety = keys
            numeric_vals = g.loc[g['ValueType'] == 'numeric', 'NumericValue']
            n = int(numeric_vals.count())
            mean = numeric_vals.mean() if n > 0 else np.nan
            
            # Get raw values for distribution
            raw_vals = g['RawValue'].values
            nx_count = int((g['ValueType'] == 'nx').sum())
            na_count = int((g['ValueType'] == 'missing').sum())
            dist = _get_rating_distribution_for_cell(raw_vals)
            
            item_variety_rows.append({
                'Item': item, 'Variety': variety, 'Mean': mean, 'N': n,
                'NXCount': nx_count, 'NACount': na_count,
            })
            raw_data_cache[(item, variety)] = {
                'distribution': dist,
                'n_total': len(raw_vals),
                'raw_vals': raw_vals,
            }
        
        item_variety_df = pd.DataFrame(item_variety_rows)
        if item_variety_df.empty:
            fig = go.Figure()
            fig.update_layout(template="simple_white", annotations=[{
                "text": "No data available for the current selection.",
                "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16},
            }])
            return fig
        
        # Get unique items and varieties
        unique_items = sorted(item_variety_df['Item'].unique())
        unique_varieties = sorted(item_variety_df['Variety'].unique())
        
        # Compute global means for sorting
        item_global_means = item_variety_df.groupby('Item')['Mean'].mean()
        variety_global_means = item_variety_df.groupby('Variety')['Mean'].mean()
        
        # Sort items based on sort_items parameter
        if sort_items == 'alphabetically':
            sorted_items = sorted(unique_items)
        else:  # default: 'average' (ascending - lowest average first, reversed from before)
            sorted_items = item_global_means.sort_values(ascending=True).index.tolist()
        
        # Sort varieties by average across items (ascending, like data_overview)
        sorted_varieties = variety_global_means.sort_values(ascending=True).index.tolist()
        
        # Pivot to item x variety table
        df_pivot = item_variety_df.pivot_table(
            index='Item', columns='Variety', values='Mean', aggfunc='first'
        )
        # Reorder
        df_pivot = df_pivot.loc[sorted_items, sorted_varieties]
        
        # Build customdata for hover with distribution info
        customdata = np.empty((len(sorted_items), len(sorted_varieties), 12), dtype=object)
        for i, item in enumerate(sorted_items):
            for j, variety in enumerate(sorted_varieties):
                cell_mean = df_pivot.iloc[i, j]
                metadata = raw_data_cache.get((item, variety), {
                    'distribution': {"-2": 0, "-1": 0, "0": 0, "1": 0, "2": 0, "NX": 0, "Null": 0},
                    'n_total': 0,
                    'raw_vals': [],
                })
                dist = metadata['distribution']
                
                item_meta = item_meta_map.get(item, {})
                american = item_meta.get('american', item)
                british = item_meta.get('british', item)
                
                customdata[i, j, 0] = item
                customdata[i, j, 1] = variety
                customdata[i, j, 2] = f'{cell_mean:.2f}' if pd.notna(cell_mean) else 'N/A'
                customdata[i, j, 3] = american
                customdata[i, j, 4] = british
                customdata[i, j, 5] = dist.get('-2', 0)
                customdata[i, j, 6] = dist.get('-1', 0)
                customdata[i, j, 7] = dist.get('0', 0)
                customdata[i, j, 8] = dist.get('1', 0)
                customdata[i, j, 9] = dist.get('2', 0)
                customdata[i, j, 10] = dist.get('NX', 0)
                customdata[i, j, 11] = dist.get('Null', 0)
        
        fig = go.Figure(data=go.Heatmap(
            z=df_pivot.values,
            x=sorted_varieties,
            y=sorted_items,
            colorscale='ylgnbu',
            zmid=0,
            colorbar=dict(title="Mean<br>rating"),
            customdata=customdata,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                '<b>Variety: %{customdata[1]}</b><br>'
                '<b>Mean: %{customdata[2]}</b><br>'
                'Distribution -2/-1/0/1/2: %{customdata[5]} / %{customdata[6]} / %{customdata[7]} / %{customdata[8]} / %{customdata[9]}<br>'
                'NX: %{customdata[10]} | Null: %{customdata[11]}<br>'
                'American: %{customdata[3]}<br>'
                'British: %{customdata[4]}<extra></extra>'
            ),
        ))
    
    fig.update_layout(
        template="simple_white",
        height=max(400, len(sorted_items) * 20),
        xaxis_title='Variety',
        xaxis2=dict(
            overlaying='x',
            side='top',
            type='category',
            tickmode='array',
            tickvals=sorted_varieties,
            ticktext=sorted_varieties,
            showgrid=False,
            showline=True,
            ticks='outside',
            showticklabels=True,
        ),
        yaxis_title='Lexical Item',
        margin=dict(t=40, b=40, l=200, r=100),
    )
    return fig


##############
## Export helpers
##############

def build_lexical_raw_export(lexical_raw, informants, participant_ids, items, include_sociodem=True):
    """Wide raw-value export: rows = selected participants, columns = selected items (as stored, e.g. '2'/'ND'/'NX')."""
    items = [i for i in items if i in lexical_raw.columns]
    data = lexical_raw.loc[lexical_raw['InformantID'].isin(participant_ids), ['InformantID'] + items].copy()
    if include_sociodem:
        socio_cols = [c for c in LEXICAL_SOCIODEM_COLUMNS if c in informants.columns]
        data = informants.loc[:, socio_cols].merge(data, on='InformantID', how='right')
    return data


def create_export_log_lexical(export_kind, participants, items, result_df, mode='apparent_time',
                               series_by=None, sort_by=None, extra_lines=None):
    """Generate a plain-text log file to accompany a lexical data export ZIP."""
    from datetime import datetime
    try:
        from version import __version__
    except ImportError:
        __version__ = "unknown"

    lines = [
        "BSLVC Lexical Data Export Log",
        "==================================",
        "",
        "Export Information:",
        "-------------------",
        f"Export Kind: {export_kind}",
        f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Application Version: {__version__}",
        f"Database Version: {retrieve_data.get_database_version()}",
        "",
        "Export Settings:",
        "----------------",
        f"Time Axis Mode: {mode}",
    ]
    if series_by is not None:
        lines.append(f"Series Grouping: {series_by}")
    if sort_by is not None:
        lines.append(f"Facet Sort Order: {sort_by}")
    if extra_lines:
        lines.extend(extra_lines)
    lines.extend([
        "",
        "Data Selection:",
        "---------------",
        f"Number of Participants: {len(participants)}",
        f"Number of Items: {len(items)}",
        "",
        f"Participant IDs ({len(participants)} total):",
        ", ".join(sorted(participants)),
        "",
        f"Item Codes ({len(items)} total):",
        ", ".join(sorted(items)),
        "",
        "Data Dimensions:",
        "----------------",
        f"Rows: {result_df.shape[0]}",
        f"Columns: {result_df.shape[1]}",
    ])
    return "\n".join(lines)



