import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def run_ddf_analysis(mask_path, months, plot_maps=True, plot_corr=True):
    """
    Run correlation + regression analysis for a given spatial subset.

    Parameters
    ----------
    mask_path : str
        Path to shapefile/GeoPackage containing the mask polygon(s).
    months : int or list of int
        Month(s) to include in the analysis.
    plot_maps : bool
        Whether to plot predictors + DDF side-by-side.
    plot_corr : bool
        Whether to plot the correlation matrix.
    """

    # --- Load mask polygon ---
    mask_gdf = gpd.read_file(mask_path).to_crs("EPSG:4326")

    def clip_to_shape(da, gdf):
        """Clip a rioxarray DataArray/Dataset to the mask polygon(s)."""

        # Ensure geometry is only polygons
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

        # Dissolve multiple features into one
        if len(gdf) > 1:
            gdf = gdf.dissolve()

        # Ensure spatial dims are 'y' and 'x'
        dim_map = {}
        if "lat" in da.dims: dim_map["lat"] = "y"
        if "lon" in da.dims: dim_map["lon"] = "x"
        if dim_map:
            da = da.rename(dim_map)

        # Write CRS if missing
        if da.rio.crs is None:
            da = da.rio.write_crs(gdf.crs)

        return da.rio.clip(gdf.geometry.values, gdf.crs, drop=True)

    # --- Load SWE ---
    swe = xr.open_zarr(
        "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/snow/snow_issykkul_1999_2016.zarr"
    )["SWE"]
    swe = clip_to_shape(swe, mask_gdf)

    # --- Load SRAD ---
    srad = xr.open_dataset(
        "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/CHELSA/rsds/chelsa_rsds_monthly_clim_1979_2019.nc"
    )
    srad = srad.rio.write_crs("EPSG:4326", inplace=False)
    srad = srad["rsds"].astype("float32")
    srad = clip_to_shape(srad, mask_gdf)
    srad_sel = srad.sel(month=months).mean("month")

    # --- Load DEM ---
    dem = rxr.open_rasterio(
        "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/MERIT_DEM/MERIT30_dem.tif",
        masked=True
    ).squeeze(drop=True)
    sf = float(dem.attrs.get("scale_factor", 1.0) or 1.0)
    off = float(dem.attrs.get("add_offset", 0.0) or 0.0)
    dem = (dem * sf + off).astype("float32")
    dem = clip_to_shape(dem, mask_gdf)

    # --- Load DDF ---
    ddf_monthly = xr.open_dataset(
        "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/ERA5L/Mean_DDF_500m_monthly_1999_2017.nc"
    )["ddf_monthly"]
    if ddf_monthly.rio.crs is None:
        ddf_monthly = ddf_monthly.rio.write_crs(swe.rio.crs)
    ddf_monthly = clip_to_shape(ddf_monthly, mask_gdf)
    ddf_sel = ddf_monthly.sel(month=months).mean("month")

    # --- Reproject DEM to match DDF ---
    dem_proj = dem.rio.reproject_match(
        ddf_sel, resampling=Resampling.bilinear, nodata=np.nan
    ).astype("float32")
    dem_proj.name = "elev"

    # --- Terrain derivatives ---
    dy = float(abs(dem_proj.y[1] - dem_proj.y[0]))
    dx = float(abs(dem_proj.x[1] - dem_proj.x[0]))
    gy, gx = np.gradient(dem_proj.values, dy, dx)
    slope = xr.DataArray(np.hypot(gx, gy), coords=dem_proj.coords, dims=dem_proj.dims, name="slope")
    aspect = np.arctan2(-gy, -gx) % (2 * np.pi)
    northness = xr.DataArray(np.cos(aspect), coords=dem_proj.coords, dims=dem_proj.dims, name="northness")
    eastness = xr.DataArray(np.sin(aspect), coords=dem_proj.coords, dims=dem_proj.dims, name="eastness")

    # --- SRAD to DDF grid ---
    srad_proj = srad_sel.rio.reproject_match(ddf_sel).astype("float32").rename("srad_mean")

    # --- Stack predictors ---
    preds = xr.Dataset({
        "elev": dem_proj,
        "slope": slope.astype("float32"),
        "northness": northness.astype("float32"),
        "eastness": eastness.astype("float32"),
        "srad_mean": srad_proj
    }).where(np.isfinite(ddf_sel))

    A = preds.to_array("var").stack(y_x=("y", "x")).transpose("y_x", "var")
    ddf_vec = ddf_sel.stack(y_x=("y", "x"))

    valid = np.isfinite(ddf_vec) & np.isfinite(A).all(dim="var")
    A2v = A.where(valid).dropna(dim="y_x", how="any")
    yvec = ddf_vec.where(valid).dropna(dim="y_x", how="any")

    X = A2v.values
    y = yvec.values.astype("float32")
    feature_names = list(preds.data_vars)

    # --- Merge for plotting ---
    ds_all = xr.merge([preds, ddf_sel.to_dataset(name="ddf")], join="inner")

    # --- Plot maps ---
    if plot_maps:
        vars_to_plot = ["ddf"] + list(preds.data_vars)
        ncols = 3
        nrows = int(np.ceil(len(vars_to_plot) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for i, var in enumerate(vars_to_plot):
            r, c = divmod(i, ncols)
            vmin = -1 if var in ("northness", "eastness") else None
            vmax = 1 if var in ("northness", "eastness") else None
            ds_all[var].plot(ax=axes[r, c], robust=True, cmap="viridis", vmin=vmin, vmax=vmax,
                             cbar_kwargs={"label": var})
            axes[r, c].set_title(var)
        for j in range(i + 1, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis("off")
        plt.tight_layout()
        plt.show()

    # --- Correlation matrix ---
    if plot_corr:
        df = ds_all.to_array("var").stack(sample=("y", "x")).transpose("sample", "var").to_pandas().dropna()
        corr = df.corr(method="pearson")
        fig, ax = plt.subplots(figsize=(6 + 0.4 * len(corr), 5 + 0.3 * len(corr)))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Pearson r")
        ax.set_title("Correlation matrix")
        plt.tight_layout()
        plt.show()
        if "ddf" in corr.columns:
            print("\nCorrelation with DDF:")
            print(corr["ddf"].drop(labels=["ddf"]).sort_values(ascending=False))

    gc.collect()
    # keep track of positions so we can map predictions back
    idx = A2v["y_x"].values  # stacked indices of valid pixels
    grid = ddf_sel  # 2-D template to unstack back to map

    return X, y, feature_names, idx, grid


## Run analysis

X, y, features, idx, grid = run_ddf_analysis(
    # mask_path="/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/kyzylsuu/shp/kyzylsuu.shp",
    mask_path="/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/issyk_kul/shp/issyk_kul_outline_exLake_hydrosheds.shp",
    # mask_path="/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/chong_aksuu/chong_aksuu_catchment.shp",
    # mask_path = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/cholpon_ata/cholpon_ata_catchment.shp",

    months=[4, 5, 6, 7, 8, 9]
)

## Fit model

def fit_and_plot_ddf(
    X, y, features,
    use_polynomial=True,
    features2keep=("elev", "srad_mean", "slope"),
    idx=None,            # stacked ("y_x") indices of valid pixels (from run_ddf_analysis)
    grid=None,           # 2D DataArray template (the DDF grid you modeled)
    test_size=0.2,
    random_state=42
):
    """
    Fit a linear or degree-2 polynomial model and make visual comparisons.
    If `idx` and `grid` are provided, also plot Actual, Predicted, Residual maps + parity plot together.
    """

    # --- keep only selected predictors ---
    keep_idx = [i for i, f in enumerate(features) if f in set(features2keep)]
    if len(keep_idx) == 0:
        raise ValueError(f"None of features2keep {features2keep} in provided features {features}")
    Xk = X[:, keep_idx]
    features_keep = [features[i] for i in keep_idx]
    print("Using predictors:", features_keep)

    # --- build model pipeline ---
    if use_polynomial:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("linreg", LinearRegression())
        ])
        print("\nFitting polynomial regression (degree=2)...")
    else:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("linreg", LinearRegression())
        ])
        print("\nFitting simple linear regression...")

    # --- split, fit, evaluate ---
    Xtr, Xte, ytr, yte = train_test_split(Xk, y, test_size=test_size, random_state=random_state)
    model.fit(Xtr, ytr)
    yhat_te = model.predict(Xte)

    r2  = r2_score(yte, yhat_te)
    rmse = float(np.sqrt(mean_squared_error(yte, yhat_te)))
    mae  = float(np.mean(np.abs(yte - yhat_te)))
    print(f"Holdout R²={r2:.3f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # --- print formula (expanded if polynomial) ---
    if use_polynomial:
        poly = model.named_steps["poly"]
        lin  = model.named_steps["linreg"]
        names_poly = poly.get_feature_names_out(features_keep)
        coefs = lin.coef_
        intercept = lin.intercept_
        print("\nPolynomial Regression Formula (standardized inputs):")
        print(f"DDF = {intercept:.6f} + " + " + ".join(
            f"{c:.6f}*{n}" for c, n in zip(coefs, names_poly)
        ))
    else:
        lin  = model.named_steps["linreg"]
        coefs = lin.coef_
        intercept = lin.intercept_
        print("\nLinear Regression Formula (standardized inputs):")
        print(f"DDF = {intercept:.6f} + " + " + ".join(
            f"{c:.6f}*{n}" for c, n in zip(coefs, features_keep)
        ))

    # --- maps + parity plot (if possible) ---
    ddf_pred = resid = None
    if (idx is not None) and (grid is not None):
        # predict ALL valid pixels using full Xk
        yhat_all = model.predict(Xk).astype("float32")

        vec = grid.stack(y_x=("y","x"))
        pred_vec = xr.full_like(vec, np.nan, dtype="float32")
        pred_vec.loc[dict(y_x=idx)] = yhat_all
        ddf_pred = pred_vec.unstack("y_x").rename("ddf_pred")

        ddf_act = grid.rename("ddf_act")
        resid   = (ddf_pred - ddf_act).rename("residual")

        # figure setup
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

        # actual
        both = xr.concat([ddf_act, ddf_pred], "z")
        vmin = float(both.quantile(0.02))
        vmax = float(both.quantile(0.98))
        ddf_act.plot(ax=axes[0,0], cmap="viridis", vmin=vmin, vmax=vmax,
                     cbar_kwargs={"label":"DDF [mm K⁻¹ d⁻¹]"})
        axes[0,0].set_title("Actual DDF")

        # predicted
        ddf_pred.plot(ax=axes[0,1], cmap="viridis", vmin=vmin, vmax=vmax,
                      cbar_kwargs={"label":"DDF [mm K⁻¹ d⁻¹]"})
        axes[0,1].set_title("Predicted DDF")

        # residuals
        rmin = float(resid.quantile(0.02))
        rmax = float(resid.quantile(0.98))
        resid.plot(ax=axes[1,0], cmap="coolwarm", vmin=rmin, vmax=rmax,
                   cbar_kwargs={"label":"Residual (pred - act)"})
        axes[1,0].set_title("Residuals")

        # parity plot in bottom-right
        axes[1,1].scatter(yte, yhat_te, s=6, alpha=0.6)
        lims = [min(yte.min(), yhat_te.min()), max(yte.max(), yhat_te.max())]
        axes[1,1].plot(lims, lims, 'k--', lw=1)
        axes[1,1].set_xlabel("Actual DDF (test)")
        axes[1,1].set_ylabel("Predicted DDF (test)")
        axes[1,1].set_title(f"Parity plot\nR²={r2:.3f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        axes[1,1].grid(True)

        plt.show()
    else:
        print("Skipping map comparison (idx/grid not provided).")

    return model, ddf_pred, resid

# Fit model with polynomial terms
model = fit_and_plot_ddf(X, y, features,
                         use_polynomial=True,
                         features2keep=["elev", "slope", "srad_mean"],
                         idx=idx,
                         grid=grid)



## NEEDS TO BE FIXED (see chat).



# Find best function

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt

def fit_and_plot_ddf_models(X, y, features, model_type="linear", features2keep=None,
                            degree=2, cv_folds=5, idx=None, grid=None):
    """
    Fit and compare DDF prediction models.

    Parameters
    ----------
    X, y : np.ndarray
        Predictors and target arrays.
    features : list of str
        Predictor names.
    model_type : {"linear", "poly", "loglinear", "powerlaw"}
        Type of regression to fit.
    features2keep : list of str, optional
        Subset of predictors to use.
    degree : int
        Polynomial degree (for 'poly' only).
    cv_folds : int
        Number of CV folds for evaluation.
    idx : np.ndarray, optional
        Flattened index of valid pixels (to map predictions back to grid).
    grid : tuple, optional
        (ny, nx) shape of the original grid for plotting maps.
    """

    # Keep only selected features
    if features2keep:
        keep_idx = [i for i, f in enumerate(features) if f in features2keep]
        X = X[:, keep_idx]
        features = [features[i] for i in keep_idx]

    # Transform predictors for log-based models
    if model_type in ("loglinear", "powerlaw"):
        if np.any(X <= 0):
            raise ValueError("Log-based models require all predictors > 0")
        X_trans = np.log(X)
    else:
        X_trans = X.copy()

    # Transform target for powerlaw
    if model_type == "powerlaw":
        if np.any(y <= 0):
            raise ValueError("Power-law model requires all target values > 0")
        y_trans = np.log(y)
    else:
        y_trans = y.copy()

    # Build model
    if model_type == "linear":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("linreg", LinearRegression())
        ])
    elif model_type == "poly":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression())
        ])
    elif model_type in ("loglinear", "powerlaw"):
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Cross-validation
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_trans, y_trans, cv=cv, scoring="r2")
    rmse_scores = np.sqrt(-cross_val_score(model, X_trans, y_trans, cv=cv, scoring="neg_mean_squared_error"))
    print(f"\nModel: {model_type}")
    print(f"Mean CV R²: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Mean CV RMSE: {rmse_scores.mean():.3f}")

    # Fit full model
    model.fit(X_trans, y_trans)

    # Predictions (transform back if needed)
    y_pred = model.predict(X_trans)
    if model_type == "powerlaw":
        y_pred = np.exp(y_pred)

    # Formula
    if model_type in ("linear", "loglinear", "powerlaw"):
        coefs = model.coef_
        intercept = model.intercept_
        if model_type == "linear":
            print("\nLinear formula:")
            print(f"DDF = {intercept:.4f} + " + " + ".join(
                [f"{coef:.4f}*{name}" for coef, name in zip(coefs, features)]
            ))
        elif model_type == "loglinear":
            print("\nLog-linear formula:")
            print(f"DDF = {intercept:.4f} + " + " + ".join(
                [f"{coef:.4f}*log({name})" for coef, name in zip(coefs, features)]
            ))
        elif model_type == "powerlaw":
            print("\nPower-law formula:")
            print("DDF = exp( {:.4f} + ".format(intercept) +
                  " + ".join([f"{coef:.4f}*log({name})" for coef, name in zip(coefs, features)]) + " )")
    elif model_type == "poly":
        poly = model.named_steps["poly"]
        linreg = model.named_steps["linreg"]
        names_poly = poly.get_feature_names_out(features)
        coefs = linreg.coef_
        intercept = linreg.intercept_
        print("\nPolynomial formula:")
        print(f"DDF = {intercept:.4f} + " + " + ".join(
            [f"{coef:.4f}*{name}" for coef, name in zip(coefs, names_poly)]
        ))

    # Plotting
    if grid is None or idx is None:
        # Simple parity plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y, y_pred, s=5, alpha=0.6)
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        plt.plot(lims, lims, 'k--', lw=1)
        plt.xlabel("Actual DDF")
        plt.ylabel("Predicted DDF")
        plt.title(f"DDF Prediction ({model_type})")
        plt.grid(True)
        plt.show()
    else:
        # grid is a DataArray or tuple
        if hasattr(grid, "shape"):
            ny, nx = grid.shape
        else:
            ny, nx = grid

        y_map = np.full(ny * nx, np.nan, dtype=np.float32)
        pred_map = np.full(ny * nx, np.nan, dtype=np.float32)

        # Ensure idx is integer type
        idx_int = np.asarray(idx, dtype=int)

        y_map[idx_int] = y
        pred_map[idx_int] = y_pred

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(y_map.reshape((ny, nx)), cmap="viridis")
        axes[0].set_title("Actual DDF")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(pred_map.reshape((ny, nx)), cmap="viridis")
        axes[1].set_title("Predicted DDF")
        plt.colorbar(im1, ax=axes[1])

        axes[2].scatter(y, y_pred, s=5, alpha=0.6)
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        axes[2].plot(lims, lims, 'k--', lw=1)
        axes[2].set_xlabel("Actual DDF")
        axes[2].set_ylabel("Predicted DDF")
        axes[2].grid(True)
        axes[2].set_title("Parity plot")

        plt.suptitle(f"DDF Prediction ({model_type})")
        plt.tight_layout()
        plt.show()

    return model


model = fit_and_plot_ddf_models(
    X, y, features,
    model_type="powerlaw",    # "linear", "poly", "loglinear", "powerlaw"
    features2keep=["elev", "srad_mean", "slope"],
    degree=2,
    idx=idx,            # from your run_ddf_analysis
    grid=grid            # shape of your DDF grid
)
