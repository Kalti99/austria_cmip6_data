Austria CMIP6 Temperature (Flask demo)
=====================================

Small Flask app to explore CMIP6 near-surface air temperature for Austria.

Features
- Cropped NetCDFs for Austria (ssp126/ssp245/ssp585)
- Point selection by latitude/longitude within Austria
- Monthly or annual aggregation, unit switch (K/°C)
- Linear trendlines with slope per decade
- Year detail plot with monthly values

Run locally
1) Move into the project folder:
   - `cd austria`
2) Install dependencies (user scope):
   - `python -m pip install --user -r ..\\requirements.txt`
3) Start the app:
   - `python app.py`
4) Open in browser:
   - `http://127.0.0.1:5000`

Notes
- Linear interpolation requires SciPy; otherwise the app falls back to nearest grid point. To enable linear: `python -m pip install --user scipy`.
- The full original dataset folder is ignored via `.gitignore`.

Repo layout
- `austria/app.py` — Flask app
- `austria/templates/index.html` — HTML template
- `austria/static/style.css` — Styles
- `austria/*.nc` — Cropped NetCDFs (Austria)
- `.gitignore`, `requirements.txt`, `README.md`

