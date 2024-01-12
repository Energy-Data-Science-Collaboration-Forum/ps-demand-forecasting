# Gas Demand Forecasting
[National Gas Transmission (GT)](https://www.nationalgas.com/) own, manage, and operate the national transmission network in Great Britain, making gas available when and where it’s needed. We do not produce or sell gas ourselved but instead are required by law to develop, maintain, and operate economic and efficient networks and to facilitate competition in the supply of gas.

As part of our role as operator of the network we are [incentivised](https://www.nationalgas.com/about-us/system-operator-incentives/demand-forecasting) to publish gas demand forecasts over a range of timescales. This is to assist the industry to make efficient decisions in balancing their supply and demand positions. It is therefore important for these values to be as accurate as possible.

We are now making the work we have done to improve the forecasts publicly available through this repository. **This is only for the purpose of collaboration and is not a representation of how the published forecast is created.** This is also a work in progress, it may at times not be complete or fail to work but just drop us a message or [create an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) and we'll try to resolve it.


## Setup

To run the code in this repository or contribute new code please ensure you have all dependencies installed as follows (note this requires an [Anaconda](https://www.anaconda.com/) installation):

```
conda create -f environment.yml
# or pip install -r requirements.txt
conda activate gas_demand_forecasting
```

We have included sample data files with made up data in the _data_ folder to get you started. Full versions of these datasets are publicly available, please view the README in the _data_ folder for instructions on how to get them. If you wish to run with the sample data be sure to update the `main.py` script with their name.


## Run Model Training

The demand forecast we publish is for the entire network, i.e. the National Transmission System (NTS). However to create this forecast we first look at the distinct components that make up the NTS. The largest two are Powerstations (PS) and Local Distribution Zones (LDZ). Here we include code to create a model for **PS** gas demand.

Once your environment is setup you can run `python main.py` from the command line to re-run the training and calculate model performance. Please note that the models are also automatically trained as per the schedule in *.github/workflows/train_models.yml*.

## Results

The last run with the full datasets was on 16th March 2023 with data up to and including January 2023 with the following results:

|Model|MAE|MAPE|
---|---|---|
|GLM_63|5.96|0.12|
|PS_GAM|8.53|0.17|

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

The code in this repository is licensed under the terms of the MIT license.
