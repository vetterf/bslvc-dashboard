#### BSLVC Interactive Data Dashboard ####

# Configure environment for Docker compatibility
import os
import tempfile

# Set matplotlib cache directory to avoid permission issues in Docker
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))
# Ensure matplotlib uses a writable cache directory
if not os.path.exists(os.environ['MPLCONFIGDIR']):
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# to do:
# getGroupingsforCache => returns dict, should return dataframe, so that it can be converted to json
# Add tab for auxiliary plots ("Age/gender","distinct items")
# Add Card for better visualization
# move setting from grid to aside (works better on smaller screens)
# Add a "loading" spinner

import os
import dash

from dash import Dash,_dash_renderer, html, dcc, Input, Output, State, page_container, callback, get_relative_path,DiskcacheManager, CeleryManager
from dataclasses import dataclass

import io

import plotly.express as px
import pandas as pd


from sqlalchemy import create_engine
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import pages.data.retrieve_data as rd
from dash.exceptions import PreventUpdate
from uuid import uuid4
import diskcache
import pickle
from flask import request

############################################################################################################

#_dash_renderer._set_react_version("18.2.0")
launch_uid = uuid4()

def get_icon(icon):
    return DashIconify(icon=icon, height=16)

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(
    cache, cache_by=[lambda: launch_uid], expire=60
)

def invalidate_cache_if_requested():
    """Check for cache invalidation URL parameter and clear cache if requested"""
    try:
        # Check if the clear_cache parameter is present in the URL
        if request and request.args.get('clear_cache') == 'true':
            # Clear main diskcache
            cache.clear()
            print("Main cache cleared via URL parameter")
            
            # Try to clear plot cache if available
            try:
                from pages.grammar import plot_cache
                plot_cache.clear()
                print("Plot cache cleared via URL parameter")
            except ImportError:
                print("Plot cache not available for clearing")
            except Exception as e:
                print(f"Error clearing plot cache: {e}")
            
            # Try to clear LRU caches from grammar functions
            try:
                from pages import grammar
                lru_functions = [
                    'get_grammar_data_cached',
                    'get_grammar_data_pairs_cached', 
                    'get_informants_cached',
                    'get_grammar_data_raw_cached',
                    'get_grammar_data_pairs_raw_cached',
                    'get_grammar_meta_cached',
                    'get_grammar_meta_pairs_cached',
                    'get_grammar_items_cols_cached',
                    'get_grammar_items_cols_pairs_cached',
                    'get_initial_umap_plot',
                    'get_initial_item_plot'
                ]
                
                for func_name in lru_functions:
                    if hasattr(grammar, func_name):
                        func = getattr(grammar, func_name)
                        if hasattr(func, 'cache_clear'):
                            func.cache_clear()
                            
                print("LRU caches cleared via URL parameter")
            except ImportError:
                print("Grammar module not available for LRU cache clearing")
            except Exception as e:
                print(f"Error clearing LRU caches: {e}")
            
            # Clear UMAP presets directory
            try:
                import shutil
                umap_presets_dir = os.path.join("pages", "data", "umap_presets")
                if os.path.exists(umap_presets_dir):
                    # Remove all files in the directory but keep the directory
                    for filename in os.listdir(umap_presets_dir):
                        file_path = os.path.join(umap_presets_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    print("UMAP presets directory cleared via URL parameter")
                else:
                    print("UMAP presets directory does not exist")
            except Exception as e:
                print(f"Error clearing UMAP presets directory: {e}")
            
            return True
    except Exception as e:
        print(f"Error clearing cache: {e}")
        # If we can't access request (e.g., during initialization), just continue
        pass
    return False

app = Dash(__name__,external_stylesheets=dmc.styles.ALL, use_pages=True,suppress_callback_exceptions=True,background_callback_manager=background_callback_manager)
app.title = 'BSLVC'
server = app.server

# Add cache invalidation middleware
@server.before_request
def check_cache_invalidation():
    invalidate_cache_if_requested()

def create_main_link(icon, label, href):
    return dmc.Anchor(
        dmc.Group(
            [
                DashIconify(
                    icon=icon,
                    width=23,
                ) if icon else None,
                dmc.Text(label, size="sm"),
            ]
        ),
        # use get_relative_path when hosting to services like
        # Dash Enterprise or pycafe that uses pathname prefixes.
        # See the dash docs for mor info
        href=get_relative_path(href),
        variant="text",
        mb=5,
        underline=False,
    )



navbar = dmc.AppShellNavbar(
            id="navbar",
            children=[
            dmc.NavLink(
                label="Home",
                href="/",
                leftSection=get_icon(icon="tabler:home"),
                active="partial",
            ),
            dmc.NavLink(
                label="Getting Started",
                href="/getting-started",
                leftSection=get_icon(icon="tabler:rocket"),
                disabled=False,
                active="partial", 
            ),
            dmc.NavLink(
                label="About",
                href="/about",
                leftSection=get_icon(icon="tabler:info-square"),
                disabled=False,
                active="partial", 
            ),
            dmc.Divider(label="Data & Analysis", labelPosition="left", mt="md", mb="xs"),
            dmc.NavLink(
                label="Data Overview",
                href="/data-overview",
                leftSection=get_icon(icon="tabler:database"),
                active="partial",
            ),
            dmc.NavLink(
                label="Grammar Sets",
                href="/grammar",
                leftSection=get_icon(icon="tabler:library"),
                active="partial",                                                                                                                                                    
            ),
            dmc.NavLink(
                label="Lexical Sets",
                leftSection=get_icon(icon="tabler:book-2"),
                disabled=True,
                style={"color": "gray", "cursor": "not-allowed"},
            ),
            dmc.Divider(label="External resources", labelPosition="left", mt="md", mb="xs"),
            dmc.Anchor(
                dmc.NavLink(
                    label="BSLVC Project Homepage",
                    leftSection=get_icon(icon="tabler:external-link"),
                ),
                href="https://www.uni-bamberg.de/en/eng-ling/forschung/the-bslvc-project-dfg-funded/",
                target="_blank",
                underline=False
            ),
            dmc.Anchor(
                dmc.NavLink(
                    label="OSF Repository",
                    leftSection=get_icon(icon="tabler:database-export"),
                ),
                href="https://osf.io/4buef/",
                target="_blank",
                underline=False
            ),
            dmc.Anchor(
                dmc.NavLink(
                    label="Documentation",
                    leftSection=get_icon(icon="tabler:book-2"),
                ),
                href="https://github.com/vetterf/bslvc-dashboard/documentation",
                target="_blank",
                underline=False
            ),
            ],
            p="md"
        )


header = dmc.Group(children=[
                      dmc.Group(
                          children=[
                                dmc.Burger(
                                    id="mobile-burger",
                                    size="sm",
                                    hiddenFrom="sm",
                                    opened=False,
                                ),
                                dmc.Burger(
                                    id="desktop-burger",
                                    size="sm",
                                    visibleFrom="sm",
                                    opened=True,
                                ),
                              dmc.Image(
                                  src="/assets/img/bslvc_logo.png",
                                  h=40,
                                  w="auto",
                                  fit="contain"
                              ),
                              dmc.Title("BSLVC Dashboard", c="black", order=4)
                              ],align="center",justify="flex-start", gap="md")
                ],
                h="100%",
                px="md",
                align="center",
                justify="flex-start",
            )

app_shell = dmc.AppShell(
    [
        dmc.AppShellHeader(header, px=25,
                bg="#f8f8f8"),
        navbar,
        dmc.AppShellMain(page_container),
        dmc.AppShellFooter(
            dmc.Group(
                [
                    dmc.Text([
                        "Copyright © 2025 by ",
                        dmc.Anchor(
                            "Fabian Vetter",
                            href="https://www.uni-bamberg.de/en/eng-ling/staff/fabian-vetter/",
                            target="_blank",
                            c="blue"
                        ),
                        " | Licensed under ",
                        dmc.Anchor(
                            "MIT License",
                            href="https://opensource.org/licenses/MIT",
                            target="_blank",
                            c="blue"
                        ),
                        " | ",
                        dmc.Anchor(
                            "Chair of English Linguistics",
                            href="https://www.uni-bamberg.de/en/eng-ling/",
                            target="_blank",
                            c="blue"
                        ),
                        " @ ",
                        dmc.Anchor(
                            "University of Bamberg",
                            href="https://www.uni-bamberg.de",
                            target="_blank",
                            c="blue"
                        ),
                        " | ",
                        dmc.Anchor(
                            "DFG Grant: 548274092",
                            href="https://www.dfg.de/en",
                            target="_blank",
                            c="blue"
                        ),
                        " | v0.1.2"
                    ], size="sm", c="dimmed"),
                ],
                justify="center",
                align="center",
                h="100%"
            ),
            h=40,
            px="md"
        )
    ],
    header={"height": 50},
    padding="xl",
    navbar={
        "width": 200,
        "breakpoint": "sm",
        "collapsed": {"mobile": True},
    },
    footer={
        "breakpoint": "md",
        "collapsed": {"desktop": False, "mobile": True},
    },
    id="appshell",
)


app.layout = dmc.MantineProvider(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="cache-status", storage_type="session", data={"cleared": False}),
        dcc.Store(id="theme-store", storage_type="session", data="light"),
        dcc.Store(id='GrammarData', data=[{'imputed':[],'normal':[]}],storage_type='memory'),
        dcc.Store(id='LexcialData', data=[{'imputed':[],'normal':[]}],storage_type='memory'),
        dcc.Store(id='SurveyCount', data=[],storage_type='memory'),
        dcc.Store(id='AllData', data=[],storage_type='memory'),
        dcc.Store(id="grammar_running",storage_type='memory'),
        dcc.Store(id="home_running",storage_type='memory'),
        dcc.Store(id="lexical_running",storage_type='memory'),
        dcc.Store(id="navbar-opened", data=False, storage_type='session'),
        dmc.NotificationProvider(),
        app_shell
    ],
    id="mantine-provider",
     forceColorScheme="light",
     theme={
         "primaryColor": "indigo",
         "colors":{
            "indigo": [
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                ],
            "blue": [
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                "#1f77b4",
                ]
         },
         "fontFamily": "'Inter', sans-serif",
         "components": {
             "Button": {"defaultProps": {"fw": 400}},
             "Alert": {"styles": {"title": {"fontWeight": 500}}},
             "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
             "Badge": {"styles": {"root": {"fontWeight": 500}}},
             "Progress": {"styles": {"label": {"fontWeight": 500}}},
             "RingProgress": {"styles": {"label": {"fontWeight": 500}}},
             "CodeHighlightTabs": {"styles": {"file": {"padding": 12}}},
             "Table": {
                 "defaultProps": {
                     "highlightOnHover": True,
                     "withTableBorder": True,
                     "verticalSpacing": "sm",
                     "horizontalSpacing": "md",
                 }
             },
         },
     },
)

@callback(
    Output("appshell", "navbar"),
    Input("mobile-burger", "opened"),
    Input("desktop-burger", "opened"),
    State("appshell", "navbar"),
)
def toggle_navbar(mobile_opened, desktop_opened, navbar):
    navbar["collapsed"] = {
        "mobile": not mobile_opened,
        "desktop": not desktop_opened,
    }
    return navbar

@callback(
    Output("cache-status", "data"),
    Input("url", "search"),
    State("cache-status", "data"),
    prevent_initial_call=False
)
def handle_cache_invalidation(search, cache_status):
    """Handle cache invalidation via URL parameter"""
    if search and "clear_cache=true" in search:
        try:
            # Clear main diskcache
            cache.clear()
            print("Main cache cleared via URL parameter")
            
            # Try to clear plot cache if available
            try:
                from pages.grammar import plot_cache
                plot_cache.clear()
                print("Plot cache cleared via URL parameter")
            except ImportError:
                print("Plot cache not available for clearing")
            except Exception as e:
                print(f"Error clearing plot cache: {e}")
            
            # Try to clear LRU caches from grammar functions
            try:
                from pages import grammar
                lru_functions = [
                    'get_grammar_data_cached',
                    'get_grammar_data_pairs_cached', 
                    'get_informants_cached',
                    'get_grammar_data_raw_cached',
                    'get_grammar_data_pairs_raw_cached',
                    'get_grammar_meta_cached',
                    'get_grammar_meta_pairs_cached',
                    'get_grammar_items_cols_cached',
                    'get_grammar_items_cols_pairs_cached',
                    'get_initial_umap_plot',
                    'get_initial_item_plot'
                ]
                
                for func_name in lru_functions:
                    if hasattr(grammar, func_name):
                        func = getattr(grammar, func_name)
                        if hasattr(func, 'cache_clear'):
                            func.cache_clear()
                            
                print("LRU caches cleared via URL parameter")
            except ImportError:
                print("Grammar module not available for LRU cache clearing")
            except Exception as e:
                print(f"Error clearing LRU caches: {e}")
            
            # Clear UMAP presets directory
            try:
                import shutil
                umap_presets_dir = os.path.join("pages", "data", "umap_presets")
                if os.path.exists(umap_presets_dir):
                    # Remove all files in the directory but keep the directory
                    for filename in os.listdir(umap_presets_dir):
                        file_path = os.path.join(umap_presets_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    print("UMAP presets directory cleared via URL parameter")
                else:
                    print("UMAP presets directory does not exist")
            except Exception as e:
                print(f"Error clearing UMAP presets directory: {e}")
            
            return {"cleared": True, "timestamp": str(pd.Timestamp.now())}
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return cache_status
    return cache_status


if __name__ == '__main__':
    app.run(debug=True)