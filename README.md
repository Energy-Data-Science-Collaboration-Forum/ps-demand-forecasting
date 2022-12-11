# Gas Demand Forecasting
[National Grid Gas Transmission (GT)](https://www.nationalgrid.com/gas-transmission/) own, manage, and operate the national transmission network in Great Britain, making gas available when and where it’s needed. We do not produce or sell gas ourselved but instead are required by law to develop, maintain, and operate economic and efficient networks and to facilitate competition in the supply of gas. 

As part of our role as operator of the network we are [incentivised](https://www.nationalgrid.com/gas-transmission/about-us/system-operator-incentives/demand-forecasting) to publish gas demand forecasts over a range of timescales. This is to assist the industry to make efficient decisions in balancing their supply and demand positions. It is therefore important for these values to be as accurate as possible. 

We are now making the work we have done to improve the forecasts publicly available through this repository. **This is only for the purpose of collaboration and is not a representation of how the published forecast is created.**

## Setup

To run the code in this repository or contribute new code please ensure you have all dependencies installed as follows (note this requires an [Anaconda](https://www.anaconda.com/) installation):

```
conda create -f environment.yml
# or pip install -r requirements.txt
conda activate gas_demand_forecasting
```

We have included sample data files in the _data_ folder to get you started. Full versions of these datasets are publicly available, please view the README in the _data_ folder for instructions on how to get them.

## Run Model Training

The demand forecast we publish is for the entire network, i.e. the National Transmission System (NTS). However to create this forecast we first look at the distinct components that make up the NTS. The largest two are Powerstations (PS) and Local Distribution Zones (LDZ). We therefore include code to create a model for LDZ gas demand and one for PS gas demand. 

Once your environment is setup you can run `python main.py` from the command line to re-run the training and calculate model performance. 

## Results

The last run with the full datasets was on -- date here -- with the following results:

|Model|MAE|MAPE|
---|---|---|
|PS|3.68|15.0|

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

The code in this repository is licensed under the terms of the MIT license.