"""Goa Airbnb nightly-price model.

Input : data/goa_listings.json (Apify tri_angle/airbnb-scraper, run 10 Nov 2026: Goa listings priced for the
        same stay, 20 to 22 January 2027, 2 adults, INR)
Output: model/model.json, model/meta.json, model/listings.parquet (features, predictions, SHAP)

Same approach as the Los Angeles project: XGBoost on log nightly price, SHAP explanations, 20% of listings held
out. The nightly price is the base rate before taxes ("2 nights x Rs 3,000"), so taxes and fees do not blur it.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "model"
OUT.mkdir(exist_ok=True)
THREADS = 4

AMENITIES = {
    "Pool": [r"\bpool\b"],
    "Private pool": [r"private.*pool"],
    "Hot tub": [r"hot tub", r"jacuzzi", r"bathtub"],
    "Free parking": [r"free parking", r"free .*parking"],
    "Air conditioning": [r"air conditioning", r"\bac\b", r"split type"],
    "Wifi": [r"\bwifi\b", r"wi-fi"],
    "Kitchen": [r"^kitchen$", r"\bkitchen\b"],
    "Washing machine": [r"washing machine", r"\bwasher\b"],
    "Dedicated workspace": [r"dedicated workspace"],
    "Beach access": [r"beach access", r"beachfront"],
    "Sea view": [r"sea view", r"ocean view", r"beach view"],
    "Garden": [r"garden", r"backyard"],
    "Patio or balcony": [r"patio", r"balcony"],
    "Self check-in": [r"self check-in", r"keypad", r"smart lock", r"lockbox"],
    "Pets allowed": [r"pets allowed"],
    "Breakfast": [r"breakfast"],
    "Gym": [r"\bgym\b", r"exercise"],
    "TV": [r"^tv\b", r"\btv\b"],
    "BBQ grill": [r"bbq", r"barbecue"],
    "Room service or housekeeping": [r"housekeeping", r"cleaning available", r"room service"],
}
AMENITY_RE = {k: re.compile("|".join(v), re.I) for k, v in AMENITIES.items()}
PLACES = {  # reference points for distance features
    "dist_calangute_km": (15.5439, 73.7553), "dist_panaji_km": (15.4909, 73.8278),
    "dist_palolem_km": (15.0100, 74.0232), "dist_airport_dabolim_km": (15.3808, 73.8314),
    "dist_airport_mopa_km": (15.7440, 73.8606),
}
NUMERIC = ["guests", "bedrooms", "beds", "bathrooms", "shared_bath", "is_studio", "latitude", "longitude", *PLACES,
           "dist_coast_km", "rating", "rating_cleanliness", "rating_location", "rating_value", "reviews",
           "superhost", "host_years", "amenity_count", "north_goa"] + ["am_" + k for k in AMENITIES]
CATEGORICAL = ["room_type", "property_type_g", "area"]

# A rough Goa coastline (lat, lon), north to south, for a "distance to the sea" feature.
COAST = np.array([(15.72, 73.69), (15.66, 73.71), (15.60, 73.73), (15.56, 73.75), (15.52, 73.76), (15.49, 73.80),
                  (15.45, 73.80), (15.40, 73.80), (15.35, 73.86), (15.30, 73.90), (15.25, 73.92), (15.20, 73.94),
                  (15.15, 73.95), (15.10, 73.97), (15.05, 74.00), (15.01, 74.02), (14.95, 74.05)])


def hav(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat0 - lat) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon0 - lon) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def money(s):
    if not isinstance(s, str):
        return np.nan
    m = re.search(r"([\d,]+(?:\.\d+)?)", s)
    return float(m.group(1).replace(",", "")) if m else np.nan


def nightly(item):
    d = (((item.get("price") or {}).get("breakDown") or {}).get("basePrice") or {}).get("description") or ""
    m = re.search(r"(\d+)\s*nights?\s*x\s*₹?\s*([\d,]+(?:\.\d+)?)", d)
    if m:
        return float(m.group(2).replace(",", ""))
    total = money((item.get("price") or {}).get("price"))
    return total / 2 if total == total else np.nan


def parse_items(items):
    rows = []
    for it in items:
        sub = " | ".join((it.get("subDescription") or {}).get("items") or []).lower()

        def num(pat, default=np.nan):
            m = re.search(pat, sub)
            return float(m.group(1)) if m else default
        bedrooms = num(r"(\d+)\s*bedroom", 0.0 if "studio" in sub else np.nan)
        beds = num(r"(\d+)\s*(?:queen |king |double |single |sofa )?beds?")
        baths = num(r"(\d+(?:\.\d)?)\s*(?:private |shared )?(?:attached )?bath")
        if np.isnan(baths) and "half-bath" in sub:
            baths = 0.5
        if np.isnan(baths) and "bath" in sub:
            baths = 1.0
        av = [v.get("title", "") for g in (it.get("amenities") or []) for v in (g.get("values") or [])
              if v.get("available", True)]
        am_text = " || ".join(av)
        r = it.get("rating") or {}
        host = it.get("host") or {}
        tah = host.get("timeAsHost") or {}
        crumbs = [b.get("linkText", "") for b in (it.get("breadcrumbs") or [])]
        coord = it.get("coordinates") or {}
        row = dict(
            id=str(it.get("id")), title=it.get("title") or "", url=it.get("url"), location=it.get("location") or "",
            property_type=it.get("propertyType") or "", room_type=it.get("roomType") or "",
            guests=it.get("personCapacity"), bedrooms=bedrooms, beds=beds, bathrooms=baths,
            shared_bath=float("shared" in sub), is_studio=float("studio" in sub),
            latitude=coord.get("latitude"), longitude=coord.get("longitude"),
            rating=r.get("guestSatisfaction"), rating_cleanliness=r.get("cleanliness"),
            rating_location=r.get("location"), rating_value=r.get("value"), reviews=r.get("reviewsCount") or 0,
            superhost=float(bool(host.get("isSuperHost"))),
            host_years=(tah.get("years") or 0) + (tah.get("months") or 0) / 12,
            amenity_count=len(av), north_goa=float(any("North Goa" in c for c in crumbs)),
            price=nightly(it))
        for k, rx in AMENITY_RE.items():
            row["am_" + k] = float(any(rx.search(a) for a in av))
        rows.append(row)
    return pd.DataFrame(rows)


def build(df):
    df = df.drop_duplicates("id").copy()
    # a private or shared room is one bedroom; Airbnb only lists the bed for these
    room = df["room_type"].isin(["Private room", "Shared room", "Hotel room"])
    df.loc[room & df["bedrooms"].isna(), "bedrooms"] = 1
    df = df[df["price"].between(500, 100000) & df["latitude"].between(14.8, 15.85) & df["longitude"].between(73.6, 74.4)]
    for k, (la, lo) in PLACES.items():
        df[k] = hav(df["latitude"], df["longitude"], la, lo)
    d = np.stack([hav(df["latitude"].values, df["longitude"].values, la, lo) for la, lo in COAST], axis=1)
    df["dist_coast_km"] = d.min(axis=1)
    df["area"] = df["location"].str.replace(r"\s*District.*$", "", regex=True).str.split(",").str[0].str.strip()
    top = df["area"].value_counts()
    df["area"] = df["area"].where(df["area"].isin(top[top >= 12].index), "Other")
    pt = df["property_type"].value_counts()
    df["property_type_g"] = df["property_type"].where(df["property_type"].isin(pt[pt >= 10].index), "Other")
    return df.reset_index(drop=True)


def frame(df, cats):
    X = df[NUMERIC].astype("float32").copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=cats[c])
    return X


def main():
    items = json.loads((DATA / "goa_listings.json").read_text(encoding="utf-8"))
    df = build(parse_items(items))
    print(f"{len(items)} scraped, {len(df)} usable listings")
    cats = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}
    X, y = frame(df, cats), np.log(df["price"].values)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=7)
    params = dict(n_estimators=3000, learning_rate=0.03, max_depth=5, min_child_weight=3, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=3.0, tree_method="hist", enable_categorical=True,
                  max_cat_to_onehot=1, nthread=THREADS, early_stopping_rounds=150, eval_metric="mae")
    model = xgb.XGBRegressor(**params)
    model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[te], y[te])], verbose=False)
    pt, tt = np.exp(model.predict(X.iloc[te])), df["price"].values[te]
    ape = np.abs(pt - tt) / tt
    base = np.exp(np.median(y[tr]))
    ss = 1 - np.sum((np.log(tt) - np.log(pt)) ** 2) / np.sum((np.log(tt) - np.log(tt).mean()) ** 2)
    res = dict(median_ape=float(np.median(ape)), within20=float(np.mean(ape <= 0.2)), mae=float(np.mean(np.abs(pt - tt))),
               r2_log=float(ss), baseline_mae=float(np.mean(np.abs(base - tt))),
               baseline_median_ape=float(np.median(np.abs(base - tt) / tt)))
    print({k: round(v, 3) for k, v in res.items()}, "trees", model.best_iteration + 1)
    model.save_model(OUT / "model_holdout.json")

    best = model.best_iteration + 1
    params.update(n_estimators=best, early_stopping_rounds=None)
    final = xgb.XGBRegressor(**params)
    final.fit(X, y, verbose=False)
    final.save_model(OUT / "model.json")
    ex = shap.TreeExplainer(final)
    sv = ex.shap_values(X)
    out = df.copy()
    out["pred_price"] = np.exp(final.predict(X))
    out["in_test"] = False
    out.loc[te, "in_test"] = True
    out.loc[te, "pred_price"] = pt
    out = pd.concat([out, pd.DataFrame(sv.astype("float32"), columns=["shap_" + c for c in X.columns])], axis=1)
    out.to_parquet(OUT / "listings.parquet", index=False)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    meta = dict(results=res, n_scraped=len(items), n_listings=int(len(df)), n_test=int(len(te)), trees=int(best),
                base_value=float(ex.expected_value), numeric=NUMERIC, categorical=CATEGORICAL, categories=cats,
                amenities=list(AMENITIES), importance=imp.to_dict(), places=PLACES,
                stay="20 to 22 January 2027, 2 adults", scraped="10 November 2026")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("top features:", list(imp.index[:10]))


if __name__ == "__main__":
    main()
