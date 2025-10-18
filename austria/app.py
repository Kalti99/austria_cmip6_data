import os
import numpy as np
import xarray as xr
from flask import Flask, render_template, request
import plotly.graph_objects as go

# Optional SciPy for linear interpolation
try:
    import scipy as _scipy  # noqa: F401
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


BASE_DIR = os.path.dirname(__file__)

SCENARIOS = {
    "ssp126": os.path.join(BASE_DIR, "meantemp_ssp126_austria.nc"),
    "ssp245": os.path.join(BASE_DIR, "meantemp_ssp245_austria.nc"),
    "ssp585": os.path.join(BASE_DIR, "meantemp_ssp585_austria.nc"),
}


def _find_var(ds: xr.Dataset) -> str:
    if "tas" in ds.data_vars:
        return "tas"
    for v, da in ds.data_vars.items():
        if {"time", "lat", "lon"}.issubset(set(da.dims)):
            return v
    return list(ds.data_vars)[0]


def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = (da.attrs.get("units") or "").lower()
    if units in ("k", "kelvin"):
        out = da - 273.15
        out.attrs["units"] = "°C"
        return out
    if units in ("c", "degc", "°c", "celsius"):
        da.attrs["units"] = "°C"
        return da
    return da


def _domain_mean_series(path: str, agg: str = "monthly", unit: str = "C"):
    ds = xr.open_dataset(path)
    var = _find_var(ds)
    da = ds[var]

    if unit.upper() == "C":
        da = _to_celsius(da)

    weights = np.cos(np.deg2rad(ds["lat"]))
    mean_ts = da.weighted(weights).mean(dim=("lat", "lon"))

    if agg == "annual":
        mean_ts = mean_ts.groupby("time.year").mean("time")
        years = mean_ts["year"].values
        x_plot = years
        t_num = years.astype(float)
    else:
        # Keep monthly as strings for plotting (CFTime-safe),
        # but compute numeric decimal years for trend calculation.
        time_index = mean_ts["time"]
        years = time_index.dt.year.values
        months = time_index.dt.month.values
        t_num = years + (months - 0.5) / 12.0
        x_plot = np.array([str(v) for v in time_index.values])

    y = mean_ts.values
    units_label = mean_ts.attrs.get("units", "")
    return x_plot, y, units_label, t_num


def _point_series(path: str, lat: float, lon: float, agg: str = "monthly", unit: str = "C", method: str = "nearest"):
    ds = xr.open_dataset(path)
    var = _find_var(ds)
    da = ds[var]

    if unit.upper() == "C":
        da = _to_celsius(da)

    # Interpolate or pick nearest grid point (avoid SciPy requirement for nearest)
    method = method if method in ("nearest", "linear") else "nearest"
    if method == "linear" and not HAS_SCIPY:
        method = "nearest"
    if method == "nearest":
        ts = da.sel(lat=lat, lon=lon, method="nearest")
    else:
        try:
            ts = da.interp(lat=lat, lon=lon, method="linear")
        except Exception:
            ts = da.sel(lat=lat, lon=lon, method="nearest")

    if agg == "annual":
        ts = ts.groupby("time.year").mean("time")
        years = ts["year"].values
        x_plot = years
        t_num = years.astype(float)
    else:
        time_index = ts["time"]
        years = time_index.dt.year.values
        months = time_index.dt.month.values
        t_num = years + (months - 0.5) / 12.0
        x_plot = np.array([str(v) for v in time_index.values])

    y = ts.values
    units_label = ts.attrs.get("units", "")
    return x_plot, y, units_label, t_num


def build_figure(agg: str = "monthly", unit: str = "C", lat: float | None = None, lon: float | None = None, method: str = "nearest") -> go.Figure:
    fig = go.Figure()
    colors = {"ssp126": "#2ca02c", "ssp245": "#1f77b4", "ssp585": "#d62728"}
    units_label = ""
    for scen, path in SCENARIOS.items():
        if not os.path.exists(path):
            continue
        if lat is not None and lon is not None:
            x, y, units_label, t_num = _point_series(path, lat=lat, lon=lon, agg=agg, unit=unit, method=method)
        else:
            x, y, units_label, t_num = _domain_mean_series(path, agg=agg, unit=unit)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name=scen.upper(), line=dict(color=colors.get(scen)))
        )

        # Add linear trendline (least squares)
        msk = np.isfinite(t_num) & np.isfinite(y)
        if msk.sum() >= 2:
            a, b = np.polyfit(t_num[msk], y[msk], 1)  # slope per year
            y_fit = a * t_num + b
            slope_decade = a * 10.0
            units_txt = units_label or ("°C" if unit.upper() == "C" else "K")
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_fit,
                    mode="lines",
                    name=f"{scen.upper()} Trend ({slope_decade:+.2f} {units_txt}/Dekade)",
                    line=dict(color=colors.get(scen), dash="dash"),
                )
            )

    title_text = (
        f"Austria point temperature @ lat={lat:.3f}, lon={lon:.3f} — {agg} ({unit.upper()})"
        if lat is not None and lon is not None
        else f"Austria mean near-surface air temperature — {agg} ({unit.upper()})"
    )
    fig.update_layout(
        title=title_text,
        xaxis_title="Time" if agg == "monthly" else "Year",
        yaxis_title=f"Temperature [{units_label}]" if units_label else "Temperature",
        template="plotly_white",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _year_bounds() -> tuple[int, int] | None:
    # Inspect the first present file to extract available year range
    for _, path in SCENARIOS.items():
        if os.path.exists(path):
            ds = xr.open_dataset(path)
            t = ds["time"]
            try:
                years = t.dt.year.values
                return int(years.min()), int(years.max())
            except Exception:
                continue
    return None


def build_year_detail_figure(year: int, unit: str = "C", lat: float | None = None, lon: float | None = None, method: str = "nearest") -> go.Figure:
    # Shows monthly values within the selected year for each scenario (at point or domain mean)
    fig = go.Figure()
    colors = {"ssp126": "#2ca02c", "ssp245": "#1f77b4", "ssp585": "#d62728"}
    months_labels = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    units_label = ""

    for scen, path in SCENARIOS.items():
        if not os.path.exists(path):
            continue
        if lat is not None and lon is not None:
            x_all, y_all, units_label, tnum = _point_series(path, lat=lat, lon=lon, agg="monthly", unit=unit, method=method)
            # y_all corresponds to monthly sequence across all years; filter by chosen year
            ds = xr.open_dataset(path)
            var = _find_var(ds)
            da = ds[var]
            if unit.upper() == "C":
                da = _to_celsius(da)
            # Safe selection/interpolation without requiring SciPy
            _method = method if method in ("nearest", "linear") else "nearest"
            if _method == "linear" and not HAS_SCIPY:
                _method = "nearest"
            if _method == "nearest":
                ts = da.sel(lat=lat, lon=lon, method="nearest")
            else:
                try:
                    ts = da.interp(lat=lat, lon=lon, method="linear")
                except Exception:
                    ts = da.sel(lat=lat, lon=lon, method="nearest")
            ts_year = ts.where(ts["time"].dt.year == year, drop=True)
            if ts_year.size == 0:
                continue
            months = ts_year["time"].dt.month.values
            y = ts_year.values
        else:
            # domain mean monthly, then subset by year
            ds = xr.open_dataset(path)
            var = _find_var(ds)
            da = ds[var]
            if unit.upper() == "C":
                da = _to_celsius(da)
            weights = np.cos(np.deg2rad(ds["lat"]))
            mean_ts = da.weighted(weights).mean(dim=("lat", "lon"))
            ts_year = mean_ts.where(mean_ts["time"].dt.year == year, drop=True)
            if ts_year.size == 0:
                continue
            months = ts_year["time"].dt.month.values
            y = ts_year.values
            units_label = ts_year.attrs.get("units", "")

        # Place into 12-month vector (keep missing months as NaN)
        y_month = np.full(12, np.nan, dtype=float)
        for m, val in zip(months, y):
            idx = int(m) - 1
            if 0 <= idx < 12:
                y_month[idx] = float(val)

        fig.add_trace(
            go.Scatter(x=list(range(1, 13)), y=y_month, mode="lines+markers", name=scen.upper(), line=dict(color=colors.get(scen)))
        )

    fig.update_layout(
        title=(f"Monatswerte {year} — Punkt lat={lat:.3f}, lon={lon:.3f}" if lat is not None and lon is not None else f"Monatswerte {year} — Österreich-Mittel"),
        xaxis=dict(title="Monat", tickmode="array", tickvals=list(range(1,13)), ticktext=months_labels),
        yaxis_title=f"Temperature [{units_label}]" if units_label else "Temperature",
        template="plotly_white",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def create_app() -> Flask:
    app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

    @app.route("/")
    def index():
        agg = request.args.get("agg", "monthly").lower()
        if agg not in ("monthly", "annual"):
            agg = "monthly"
        unit = request.args.get("unit", "C").upper()
        if unit not in ("C", "K"):
            unit = "C"
        # Lat/Lon inputs (defaults to Vienna approx.)
        def _parse_float(val, default):
            try:
                return float(val)
            except Exception:
                return default
        lat = _parse_float(request.args.get("lat", "48.208"), 48.208)
        lon = _parse_float(request.args.get("lon", "16.374"), 16.374)

        # Keep within Austria crop bounds
        lat = max(46.0, min(50.0, lat))
        lon = max(9.0, min(18.0, lon))

        method = request.args.get("method", "nearest").lower()
        if method not in ("nearest", "linear"):
            method = "nearest"

        # Year selection and secondary figure (monthly breakdown for selected year)
        year_bounds = _year_bounds()
        year_default = year_bounds[0] if year_bounds else None
        year_param = request.args.get("year", str(year_default) if year_default else "")
        try:
            year = int(year_param) if year_param else None
        except Exception:
            year = year_default
        if year is not None and year_bounds:
            year = max(year_bounds[0], min(year_bounds[1], year))

        fig = build_figure(agg=agg, unit=unit, lat=lat, lon=lon, method=method)
        fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        fig2_html = None
        if year is not None:
            fig2 = build_year_detail_figure(year=year, unit=unit, lat=lat, lon=lon, method=method)
            fig2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

        return render_template(
            "index.html",
            fig_html=fig_html,
            fig2_html=fig2_html,
            agg=agg,
            unit=unit,
            lat=lat,
            lon=lon,
            method=method,
            year=year,
            year_bounds=year_bounds,
            files_present={k: os.path.exists(v) for k, v in SCENARIOS.items()},
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
