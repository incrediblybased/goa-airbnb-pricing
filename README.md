# Goa Airbnb pricing engine

Fair nightly rate for a Goa listing, where the price comes from in rupees, the amenities worth adding, and real
listings ranked from most underpriced.

- **Data:** 1,100 live Goa Airbnb listings collected on 10 November 2026, all priced for the same stay
  (20 to 22 January 2027, 2 adults), across Calangute, Anjuna, Candolim, Panaji, Vagator, Siolim, Benaulim, Palolem and Margao.
- **Model:** XGBoost on log nightly base rate with size, type, area, distances to the sea and key places, ratings,
  reviews, superhost status and 20 amenity groups; SHAP explains each estimate.
- **Test:** 220 listings held out. Median error 18.7% (guessing the median rate: 48.2%), R squared 0.76 on log price.
- **Limits:** one January high-season stay, asking prices rather than booked prices.

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`. `train.py` is included for reference;
the raw scraped file is not.
