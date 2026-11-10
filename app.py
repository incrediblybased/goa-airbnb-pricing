"""Goa Airbnb pricing engine. Run: streamlit run app.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import xgboost as xgb

import ui
from train import COAST, PLACES, hav

ROOT = Path(__file__).parent
ui.apply("Goa Airbnb pricing")

GROUPS = {
    "Location": ["latitude", "longitude", *PLACES, "dist_coast_km", "area", "north_goa", "rating_location"],
    "Size and layout": ["guests", "bedrooms", "beds", "bathrooms", "shared_bath", "is_studio"],
    "Property type": ["room_type", "property_type_g"],
    "Reviews and reputation": ["rating", "rating_cleanliness", "rating_value", "reviews", "superhost", "host_years"],
}
FIXED = {"Sea view", "Beach access", "Garden"}     # cannot be added by a host


@st.cache_resource
def load():
    meta = json.loads((ROOT / "model" / "meta.json").read_text())
    m, h = xgb.XGBRegressor(), xgb.XGBRegressor()
    m.load_model(ROOT / "model" / "model.json")
    h.load_model(ROOT / "model" / "model_holdout.json")
    df = pd.read_parquet(ROOT / "model" / "listings.parquet")
    return meta, m, h, shap.TreeExplainer(m), shap.TreeExplainer(h), df


meta, model, holdout, EX, EXH, L = load()
NUMERIC, CATEGORICAL, CATS, AMEN = meta["numeric"], meta["categorical"], meta["categories"], meta["amenities"]
GROUPS["Amenities"] = ["am_" + a for a in AMEN] + ["amenity_count"]
R = meta["results"]


def frame(rows):
    X = rows[NUMERIC].astype("float32").copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(rows[c], categories=CATS[c])
    return X


def rupees(x, sign=False):
    s = f"₹{abs(x):,.0f}"
    return (("+" if x >= 0 else "-") + s) if sign else s


def explain(row, use_holdout=False):
    ex = EXH if use_holdout else EX
    sv = ex.shap_values(frame(row))[0]
    by = dict(zip(NUMERIC + CATEGORICAL, sv))
    base, price = float(np.exp(ex.expected_value)), float(np.exp(ex.expected_value + sv.sum()))
    tot = sv.sum()
    parts = {g: (price - base) * sum(by.get(c, 0) for c in cs) / tot if abs(tot) > 1e-9 else 0 for g, cs in GROUPS.items()}
    return base, price, parts


def upgrades(row, mdl):
    cur = float(np.exp(mdl.predict(frame(row)))[0])
    trials = []
    for a in AMEN:
        if a in FIXED or row["am_" + a].iloc[0] == 1:
            continue
        r = row.copy()
        r["am_" + a] = 1
        r["amenity_count"] = r["amenity_count"] + 1
        trials.append((a, r))
    if not trials:
        return pd.DataFrame(columns=["Amenity", "Per night"])
    new = np.exp(mdl.predict(frame(pd.concat([t[1] for t in trials], ignore_index=True))))
    return pd.DataFrame({"Amenity": [t[0] for t in trials], "Per night": new - cur}).sort_values("Per night", ascending=False)


def breakdown_chart(base, price, parts):
    items = sorted(parts.items(), key=lambda kv: -abs(kv[1]))
    fig = go.Figure(go.Waterfall(
        orientation="h", y=["Typical Goa listing"] + [k for k, _ in items] + ["This listing"],
        x=[base] + [v for _, v in items] + [price], measure=["absolute"] + ["relative"] * len(items) + ["total"],
        text=[rupees(base)] + [rupees(v, True) for _, v in items] + [rupees(price)], textposition="outside",
        increasing=dict(marker=dict(color=ui.POSITIVE)), decreasing=dict(marker=dict(color=ui.NEGATIVE)),
        totals=dict(marker=dict(color=ui.INK)), connector=dict(line=dict(color=ui.RULE, width=1))))
    ui.plotly_layout(fig, height=360)
    run = np.cumsum([base] + [v for _, v in items])
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickprefix="₹", range=[0, max(run.max(), price) * 1.25])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def upgrades_table(row, mdl):
    up = upgrades(row, mdl)
    up = up[up["Per night"] >= 50].head(8)
    if up.empty:
        st.markdown('<p class="note">No missing amenity on our list adds more than ₹50 a night here.</p>', unsafe_allow_html=True)
        return
    body = "".join(f"<tr><td>{r.Amenity}</td><td class='num pos'>{rupees(r['Per night'], True)}</td>"
                   f"<td class='num'>{rupees(r['Per night'] * 10, True)}</td></tr>" for _, r in up.iterrows())
    st.markdown("<table class='plain'><tr><th>Add this</th><th class='num'>Per night</th><th class='num'>Per 10 nights booked</th></tr>"
                f"{body}</table>", unsafe_allow_html=True)


# ---------------------------------------------------------------- header
st.markdown(f'<div class="kicker">Goa, {meta["n_listings"]:,} live Airbnb listings collected {meta["scraped"]}, '
            f'priced for {meta["stay"]}</div>', unsafe_allow_html=True)
st.markdown("# What should your Goa listing charge a night?")
st.markdown('<p class="lede">A model trained on real Goa Airbnb listings estimates a fair nightly rate, shows which parts '
            'of the listing add or remove money, and lists the amenities worth adding.</p>', unsafe_allow_html=True)

t_new, t_live, t_market, t_acc = st.tabs(["Price a listing", "Check a live listing", "What moves price in Goa", "How accurate it is"])

with t_new:
    areas = (L.groupby("area").agg(n=("id", "size"), lat=("latitude", "median"), lon=("longitude", "median"),
                                   north=("north_goa", "median")).sort_values("n", ascending=False))
    area_list = [a for a in areas.index if a != "Other"]
    med = L[NUMERIC].median()
    with st.form("new"):
        c1, c2, c3 = st.columns([1, 1, 1.5])
        with c1:
            st.markdown("### The place")
            area = st.selectbox("Area", sorted(area_list), index=sorted(area_list).index("Calangute") if "Calangute" in area_list else 0)
            room = st.selectbox("Rental type", CATS["room_type"], index=CATS["room_type"].index("Entire home/apt"))
            ptypes = CATS["property_type_g"]
            prop = st.selectbox("Property type", ptypes, index=ptypes.index("Entire rental unit") if "Entire rental unit" in ptypes else 0)
            guests = st.number_input("Guests", 1, 16, 4)
            bedrooms = st.number_input("Bedrooms", 0, 10, 2)
            beds = st.number_input("Beds", 1, 16, 2)
            baths = st.number_input("Bathrooms", 0.5, 8.0, 2.0, step=0.5)
        with c2:
            st.markdown("### Reviews")
            rating = st.slider("Average rating", 3.0, 5.0, 4.8, 0.05)
            n_rev = st.number_input("Number of reviews", 0, 1000, 25)
            superhost = st.checkbox("Superhost")
            years = st.number_input("Years hosting", 0, 15, 3)
        with c3:
            st.markdown("### Amenities")
            defaults = {"Air conditioning", "Wifi", "Free parking", "Kitchen", "TV", "Patio or balcony"}
            ca, cb = st.columns(2)
            half = (len(AMEN) + 1) // 2
            picked = [a for i, a in enumerate(AMEN) if (ca if i < half else cb).checkbox(a, value=a in defaults, key="n_" + a)]
        st.form_submit_button("Estimate the price")

    a = areas.loc[area]
    row = pd.DataFrame([med])
    row["latitude"], row["longitude"], row["north_goa"] = a["lat"], a["lon"], a["north"]
    for k, (la, lo) in PLACES.items():
        row[k] = hav(row["latitude"].values, row["longitude"].values, la, lo)
    row["dist_coast_km"] = min(hav(row["latitude"].values, row["longitude"].values, la, lo)[0] for la, lo in COAST)
    row["guests"], row["bedrooms"], row["beds"], row["bathrooms"] = guests, bedrooms, beds, baths
    row["shared_bath"], row["is_studio"] = 0, float(bedrooms == 0)
    row["rating"] = rating if n_rev else np.nan
    for c in ["rating_cleanliness", "rating_value"]:
        row[c] = min(5.0, rating + 0.02) if n_rev else np.nan
    row["rating_location"] = L.loc[L["area"] == area, "rating_location"].median() if n_rev else np.nan
    row["reviews"], row["superhost"], row["host_years"] = n_rev, float(superhost), years
    for am in AMEN:
        row["am_" + am] = float(am in picked)
    row["amenity_count"] = med["amenity_count"] + len(picked) - 6
    row["room_type"], row["property_type_g"], row["area"] = room, prop, area

    base, price, parts = explain(row)
    sim = L[(L["area"] == area) & (L["room_type"] == room) & (L["bedrooms"] == bedrooms)]["price"]
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown('<div class="panel"><div class="kicker">Estimated nightly rate</div>'
                    f'<div class="figure accent">{rupees(price)}</div>'
                    f'<p class="note">Typical range {rupees(price * (1 - R["median_ape"]))} to {rupees(price * (1 + R["median_ape"]))}, '
                    f'from the model\'s median error of {R["median_ape"]:.0%}. Base rate before taxes and fees.</p></div>',
                    unsafe_allow_html=True)
        if len(sim) >= 5:
            st.markdown(f'<p class="note">{len(sim)} similar listings in {area} charge a median of {rupees(sim.median())}.</p>',
                        unsafe_allow_html=True)
        st.markdown("### Amenities worth adding")
        upgrades_table(row, model)
    with right:
        st.markdown("### Where the price comes from")
        breakdown_chart(base, price, parts)

with t_live:
    st.markdown('<p class="lede">Real listings, most underpriced first. Listings held out from training are scored by a model '
                'that never saw them.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        ar2 = st.selectbox("Area", sorted(L["area"].unique()), key="ar2",
                           index=sorted(L["area"].unique()).index("Anjuna") if "Anjuna" in set(L["area"]) else 0)
        only = st.checkbox("Only listings held out from training", value=True)
    pool = L[L["area"] == ar2]
    if only:
        pool = pool[pool["in_test"]]
    pool = pool.assign(gap=pool["pred_price"] - pool["price"]).sort_values("gap", ascending=False)
    with c2:
        if pool.empty:
            st.info("No listings match. Untick the held-out filter.")
            st.stop()
        lab = {i: f"{str(r['title'])[:60]}  ({rupees(r['price'])} now)" for i, r in pool.iterrows()}
        pick = st.selectbox("Listing", list(lab), format_func=lab.get)
    r = pool.loc[[pick]]
    held = bool(r["in_test"].iloc[0])
    base, fair, parts = explain(r, use_holdout=held)
    actual = float(r["price"].iloc[0])
    gap = fair - actual
    verdict = ("priced below the model's estimate" if gap > 0.15 * actual else
               "priced above the model's estimate" if gap < -0.15 * actual else "priced in line with similar listings")
    ui.stats([(rupees(actual), "Current nightly rate"), (rupees(fair), "Model's fair rate"),
              (rupees(gap, True), "Difference per night"), (rupees(gap * 10, True), "Per 10 nights booked")])
    st.markdown(f'<p>This listing is {verdict}. <a href="{r["url"].iloc[0]}" target="_blank">Open it on Airbnb</a>.</p>',
                unsafe_allow_html=True)
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("### Where its price comes from")
        breakdown_chart(base, fair, parts)
    with right:
        st.markdown("### Amenities worth adding")
        upgrades_table(r, holdout if held else model)

with t_market:
    st.markdown('<p class="lede">Averages across all listings: how much each factor moves the price compared with an otherwise '
                'typical Goa listing.</p>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("### Amenity premiums")
        rows = []
        for am in AMEN:
            have = L["am_" + am] == 1
            if have.sum() < 25 or (~have).sum() < 25:
                continue
            col = "shap_am_" + am
            rows.append((am, np.exp(L.loc[have, col].mean() - L.loc[~have, col].mean()) - 1, have.mean()))
        amd = pd.DataFrame(rows, columns=["a", "eff", "share"]).sort_values("eff", ascending=False)
        fig = go.Figure(go.Bar(x=amd["eff"] * 100, y=amd["a"], orientation="h",
                               marker_color=[ui.POSITIVE if v >= 0 else ui.NEGATIVE for v in amd["eff"]],
                               text=[f"{v:+.1%}" for v in amd["eff"]], textposition="outside",
                               customdata=amd["share"] * 100,
                               hovertemplate="%{y}: %{x:.1f}% effect, in %{customdata:.0f}% of listings<extra></extra>"))
        ui.plotly_layout(fig, height=560)
        fig.update_yaxes(autorange="reversed")
        lo, hi = min(0, amd["eff"].min() * 100), amd["eff"].max() * 100
        fig.update_xaxes(ticksuffix="%", range=[min(lo * 1.8, -6), hi * 1.3])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("### Areas")
        ar = L[L["area"] != "Other"].groupby("area").agg(eff=("shap_area", "mean"), med=("price", "median"), n=("id", "size"))
        ar["eff"] = np.exp(ar["eff"]) - 1
        ar = ar.sort_values("eff", ascending=False)
        vague = {"Goa": "Goa, area not given", "North Goa": "North Goa, area not given", "South Goa": "South Goa, area not given"}
        body = "".join(f"<tr><td>{vague.get(n, n)}</td><td class='num {'pos' if v.eff >= 0 else 'neg'}'>{v.eff:+.0%}</td>"
                       f"<td class='num'>{rupees(v.med)}</td><td class='num'>{int(v.n)}</td></tr>" for n, v in ar.iterrows())
        st.markdown("<table class='plain'><tr><th>Area</th><th class='num'>Effect of the area name</th><th class='num'>Median rate</th>"
                    f"<th class='num'>Listings</th></tr>{body}</table><p class='note'>The effect is on top of distance to the sea, "
                    "Calangute, Panaji and the airports, which the model counts separately.</p>", unsafe_allow_html=True)
    st.markdown("### What matters most, overall")
    imp = pd.Series(meta["importance"])
    grp = pd.Series({g: imp.reindex(cs).fillna(0).sum() for g, cs in GROUPS.items()}).sort_values()
    fig = go.Figure(go.Bar(x=grp.values / grp.sum() * 100, y=grp.index, orientation="h", marker_color=ui.INK,
                           text=[f"{v / grp.sum():.0%}" for v in grp.values], textposition="outside"))
    ui.plotly_layout(fig, height=280)
    fig.update_xaxes(ticksuffix="%", range=[0, grp.max() / grp.sum() * 125])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with t_acc:
    ui.stats([(f"{R['median_ape']:.1%}", "Median error on unseen listings"),
              (f"{R['within20']:.0%}", "Estimates within 20% of the real rate"),
              (rupees(R["mae"]), "Average error per night"),
              (f"{R['r2_log']:.2f}", "R squared on log price")])
    st.markdown(f"""
<div class="panel">
<h3>Method</h3>
<p>Data: {meta['n_listings']:,} Goa Airbnb listings collected on {meta['scraped']} with the Apify Airbnb scraper (official Apify
partner actor), all priced for the same stay ({meta['stay']}) so rates are comparable. The rate used is the nightly base price
before taxes and fees. Inside Airbnb, the source for the Los Angeles project, does not cover India. {meta['n_test']:,} listings
were held out for testing.</p>
<p>Model: XGBoost on the log of the nightly rate, {meta['trees']:,} trees. Inputs: guests, bedrooms, beds, bathrooms, rental and
property type, area, distance to the sea, Calangute, Panaji and both airports, North or South Goa, ratings, reviews, superhost,
years hosting and {len(AMEN)} amenity groups.</p>
<p>A guess of the median rate for every listing has a median error of {R['baseline_median_ape']:.0%}; the model's is {R['median_ape']:.0%}.</p>
<p class="note">Limits: about 1,100 listings from the main tourist areas, one set of dates in the January high season, and
asking prices rather than booked prices. Rates change sharply around Christmas and New Year and in the monsoon. Treat the
estimate as a starting point.</p>
</div>""", unsafe_allow_html=True)
