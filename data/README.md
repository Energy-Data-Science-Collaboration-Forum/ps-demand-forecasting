# Gathering Data

Previous explorations of the data have shown that historical trends in gas demand do not change significantly. We have therefor chosen to limit historical data to the period from January 2019 up to November 2022. 

If you managed to create your python environment as per the instructions in the main _README.md_ then you are ready to now also collect the data. Simply run `python src/get_data.py` and it will gather the necessary datasets and place them in the **data** folder.

You can adjust the time periods for which to download data by changing the dates in `get_data.py` (line 194-197).

## Elexon Electricity Demand Day-Ahead

National Grid Electricity System Operator (ESO) provide Elexon with forecasts of the Total Electricity Demand (TED) multiple times in a day. This forecast comes in different flavours but we're mostly interested in the Transmission System Demand Forecast (a.k.a DATF). Our historical analysis showed this forecast was the most useful for further modeling. 

The reason why electricity demand is important for the gas consumed by powerstations is because powerstations react to the electricity market. The hypothesis is that when electricity demand is high you will need powerstations to fulfil that demand, thus requiring more gas to be burned.

The electricity market operates at a half-hourly granularity. The gas market however does not so we aggregate the half-hourly values to a day value.

**Note**: We're using the API from Elexon's Insights Solution where the TSDF dataset starts from February 2022. The API from Elexon's BM Reports has more historical data. BM Reports will be replaced by Insights Solution in the near future (if you are from the near future, let us know what's going on).

## Elexon Wind Forecast

ESO also provide Elexon with forecasts for the wind-powered electricity generation which is also updated multiple times in a day. The forecast is at the hourly level (for reasons unknown to us) and we aggregate it to a day value. 

They hypothesis for using this data is that powerstations often function as a reserve for renewable energy generators. If there isn't any wind power then the gap in electricity generation will be filled by powerstations.

## Elexon Electricity Actual Generation

Since the electricity market is mostly at the half-hourly granularity, the data is more up to date. We can therefor use this more recent data as features in our models. One such example is the actual electricity generation. We don't necessarily need all values for a day, the values up to a certain point can also be predictive of what's going to happen further into the future.

The actual generation data is split up by fuel type but still at the half-hourly level.

## MIPI Gas Actuals

Historical or actual gas demand is published on the MIPI platform on a daily basis. It is split up in a number of components and we download all of them even though we are only interested in the Powerstation (PS) component. 