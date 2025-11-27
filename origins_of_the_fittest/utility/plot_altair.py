from typing import Any, cast

import altair as alt

greys = {
    "verylight": "#F0F0F0",
    "light": "#DDD",
    "mid": "#777",
    "dark": "#444",
    "blackish": "#333",
}
legend_styling = {
    "titleColor": greys["dark"],
    "titleFontSize": 14,
    "titleFontWeight": 400,
    "labelColor": greys["dark"],
    "labelFontSize": 12,
    "labelFont": "Helvetica Neue, sans serif",
    "titleFont": "Helvetica Neue, sans serif",
}
config = {
    "config": {
        "padding": 20,
        "view": {"stroke": None, "fill": "white"},
        "title": {
            "font": "Helvetica Neue, sans serif",
            "fontSize": 14,
            "color": greys["dark"],
            "padding": 10,
            "align": "left",
            "anchor": "start",
            "dx": 10,
        },
        "header": {
            "anchor": "start",
            "labelFontSize": 14,
            "titleColor": greys["dark"],
            "labelColor": greys["dark"],
        },
        "axis": {
            "labelFont": "Helvetica Neue, sans serif",
            "titleFont": "Helvetica Neue, sans serif",
            "grid": False,
            "gridColor": greys["verylight"],
            "labelColor": greys["mid"],
            "tickColor": greys["mid"],
            "titleColor": greys["dark"],
            "domainColor": greys["mid"],
            "labelFontSize": 12,
            "titleFontSize": 14,
            "titleFontWeight": 400,
            "tickSize": 6,
            "titlePadding": 8,
            "labelFlush": False,
        },
        "facet": {
            "titleFont": "Helvetica Neue, sans serif",
            "titleFontSize": 14,
            "color": greys["dark"],
            "spacing": 20,
            "padding": 20,
        },
        "concat": {
            "spacing": 14,
            "axis": {"title": None},
            "title": {
                "align": "left",
                "anchor": "left",
            },
        },
        "legend": legend_styling,
        "background": None,
    }
}
legend_styling_dark = {
    "titleColor": greys["light"],
    "titleFontSize": 14,
    "titleFontWeight": 400,
    "labelColor": greys["light"],
    "labelFontSize": 12,
    "labelFont": "Helvetica Neue, sans serif",
    "titleFont": "Helvetica Neue, sans serif",
}

config_dark = {
    "config": {
        "padding": 20,
        "view": {"stroke": greys["light"], "fill": None},
        "title": {
            "font": "Helvetica Neue, sans serif",
            "fontSize": 14,
            "color": greys["light"],
            "padding": 10,
            "align": "left",
            "anchor": "start",
            "dx": 10,
        },
        "header": {
            "anchor": "start",
            "labelFontSize": 14,
            "titleColor": greys["light"],
            "labelColor": greys["light"],
        },
        "axis": {
            "labelFont": "Helvetica Neue, sans serif",
            "titleFont": "Helvetica Neue, sans serif",
            "grid": False,
            "gridColor": greys["dark"],
            "labelColor": greys["verylight"],
            "tickColor": greys["verylight"],
            "titleColor": greys["light"],
            "domainColor": greys["verylight"],
            "labelFontSize": 12,
            "titleFontSize": 14,
            "titleFontWeight": 400,
            "tickSize": 6,
            "titlePadding": 8,
            "labelFlush": False,
        },
        "facet": {
            "titleFont": "Helvetica Neue, sans serif",
            "titleFontSize": 14,
            "color": greys["light"],
            "spacing": 20,
            "padding": 20,
        },
        "concat": {
            "spacing": 14,
            "axis": {"title": None},
            "title": {
                "align": "left",
                "anchor": "left",
            },
        },
        "legend": legend_styling_dark,
        "background": None,
    }
}


legend_styling_neutral = {
    "titleColor": greys["mid"],
    "titleFontSize": 14,
    "titleFontWeight": 400,
    "labelColor": greys["mid"],
    "labelFontSize": 12,
    "labelFont": "Helvetica Neue, sans serif",
    "titleFont": "Helvetica Neue, sans serif",
}

config_neutral = {
    "config": {
        "padding": 20,
        "view": {"stroke": None, "fill": "white"},
        "title": {
            "font": "Helvetica Neue, sans serif",
            "fontSize": 14,
            "color": greys["mid"],
            "padding": 10,
            "align": "left",
            "anchor": "start",
            "dx": 10,
        },
        "header": {
            "anchor": "start",
            "labelFontSize": 14,
            "titleColor": greys["mid"],
            "labelColor": greys["mid"],
        },
        "axis": {
            "labelFont": "Helvetica Neue, sans serif",
            "titleFont": "Helvetica Neue, sans serif",
            "grid": False,
            "gridColor": greys["mid"],
            "labelColor": greys["mid"],
            "tickColor": greys["mid"],
            "titleColor": greys["mid"],
            "domainColor": greys["mid"],
            "labelFontSize": 12,
            "titleFontSize": 14,
            "titleFontWeight": 400,
            "tickSize": 6,
            "titlePadding": 8,
            "labelFlush": False,
        },
        "facet": {
            "titleFont": "Helvetica Neue, sans serif",
            "titleFontSize": 14,
            "color": greys["mid"],
            "spacing": 20,
            "padding": 20,
        },
        "concat": {
            "spacing": 14,
            "axis": {"title": None},
            "title": {
                "align": "left",
                "anchor": "left",
            },
        },
        "legend": legend_styling_neutral,
        "background": None,
    }
}


def altair_helvetica_theme(version="light"):
    """Register and enable the custom Altair theme."""

    theme_plugin = cast(Any, lambda: config)
    theme_plugin_dark = cast(Any, lambda: config_dark)
    theme_plugin_neutral = cast(Any, lambda: config_neutral)

    alt.themes.register("helvetica", theme_plugin)
    alt.themes.register("helvetica_dark", theme_plugin_dark)
    alt.themes.register("helvetica_neutral", theme_plugin_neutral)

    if version == "light":
        alt.themes.enable("helvetica")
    elif version == "dark":
        alt.themes.enable("helvetica_dark")
    else:
        alt.themes.enable("helvetica_neutral")
