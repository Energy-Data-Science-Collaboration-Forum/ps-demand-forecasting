import logging
import datetime as dt
import pandas as pd
import requests
from os import path
from requests import Session

from zeep import Client
from zeep.helpers import serialize_object
from zeep.transports import Transport


logger = logging.getLogger(__name__)

FORMAT = "%Y%m%d_%H%M%S"
GAS_ACTUAL_DATA_ITEMS = [
    "NTS Volume Offtaken, Industrial Offtake Total",
    "NTS Volume Offtaken, Interconnector Exports Total",
    "NTS Volume Offtaken, LDZ Offtake Total",
    "NTS Volume Offtaken, Powerstations Total",
    "NTS Volume Offtaken, Storage Injection Total",
]

MIPI_URL = "http://marketinformation.natgrid.co.uk/MIPIws-public/public/publicwebservice.asmx?wsdl"
ELEXON_TED_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/TSDF?boundary=N&"
ELEXON_WIND_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR?"
ELEXON_ELEC_GEN_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH?"


def get_gas_actuals_from_mipi(output_dir, from_date, to_date):
    """Retrieve actual gas demand data at an aggregated level (see GAS_ACTUAL_DATA_ITEMS)
    between the given dates. Data is written to the given output directory with a timestamped file name.

    Args:
        output_dir (str): directory path to write the output to
        from_date (str): Lower bound for the applicable date of the dataset, yyyy-mm-dd format
        to_date (str): Upper bound for the applicable date of the dataset, yyyy-mm-dd format
    """
    actual_data = get_mipi_data(GAS_ACTUAL_DATA_ITEMS, from_date, to_date)
    if len(actual_data) > 0:
        df = pd.concat(actual_data)
        df = df.rename(columns={"DATA_ITEM": "TYPE"})
        df["Value"] = pd.to_numeric(df["Value"])
        df.to_csv(
            path.join(output_dir, f"gas_actuals_{dt.datetime.now().strftime(FORMAT)}.csv"),
            index=False,
        )
    else:
        logger.warn("No Actuals Data Returned")


def get_mipi_data(item_names, from_date, to_date):
    """Retrieve data from MIPI for the given data sets (item names) and between the given dates

    Args:
        item_names (list): List of strings corresponding to datasets in MIPI
        from_date (str): Lower bound for the applicable date of the dataset, yyyy-mm-dd format
        to_date (str): Upper bound for the applicable date of the dataset, yyyy-mm-dd format

    Returns:
        list: A list of DataFrames, each DataFrame represents a dataset
    """
    session = Session()
    client = Client(MIPI_URL, transport=Transport(session=session))

    body = {
        "LatestFlag": "Y",
        "ApplicableForFlag": "Y",
        "FromDate": from_date,
        "ToDate": to_date,
        "DateType": "GASDAY",
    }
    result = []
    for item in item_names:

        logger.debug(
            f"MIPI LDZ Actual : Gathering {item} data, from {from_date} to {to_date}",
        )

        body["PublicationObjectNameList"] = {"string": item}
        r = client.service.GetPublicationDataWM(body)
        if r is not None:
            data_dic = [
                serialize_object(d)
                for d in r[0].PublicationObjectData["CLSPublicationObjectDataBE"]
            ]
            df = pd.DataFrame(data=data_dic, columns=data_dic[0].keys())
            df["DATA_ITEM"] = item
            result.append(df)
        else:
            logger.warning(f"No Data for: {item}")

    return result


def get_ted_forecast_from_elexon(output_dir, from_date, to_date):
    """Retrieve Total Electricity Demand forecast published between the given dates (duplicates may be present).
    Data is written to the given output directory with a timestamped file name.

    Args:
        output_dir (str): directory path to write the output to
        from_date (str): Lower bound for the published date of the dataset, yyyy-mm-dd format
        to_date (str): Upper bound for the published date of the dataset, yyyy-mm-dd format
    """
    result = get_elexon_data_from_api("TED Forecast", ELEXON_TED_URL, from_date, to_date)

    if len(result) > 0:
        result = pd.concat(result)
        result.to_csv(
            f"data/elexon_ted_forecast_{dt.datetime.now().strftime(FORMAT)}.csv",
            index=False,
        )


def get_wind_forecast_from_elexon(output_dir, from_date, to_date):
    """Retrieve wind forecast published between the given dates (duplicates may be present).
    Data is written to the given output directory with a timestamped file name.

    Args:
        output_dir (str): directory path to write the output to
        from_date (str): Lower bound for the published date of the dataset, yyyy-mm-dd format
        to_date (str): Upper bound for the published date of the dataset, yyyy-mm-dd format
    """
    result = get_elexon_data_from_api("Wind Forecast", ELEXON_WIND_URL, from_date, to_date)

    if len(result) > 0:
        result = pd.concat(result)
        result.to_csv(
            path.join(output_dir, f"elexon_wind_forecast_{dt.datetime.now().strftime(FORMAT)}.csv"),
            index=False,
        )


def get_electricity_actuals_from_elexon(output_dir, from_date, to_date):
    result = get_elexon_data_from_api("Electricity Actuals", ELEXON_ELEC_GEN_URL, from_date, to_date)

    if len(result) > 0:
        result = pd.concat(result)
        result.to_csv(
            path.join(output_dir, f"elexon_electricity_actuals_{dt.datetime.now().strftime(FORMAT)}.csv"),
            index=False,
        )


def get_elexon_data_from_api(name, input_url, from_date, to_date):
    """Retrieve data from the Elexon BMRS API using the given url and for the given dates

    Args:
        name (str): Name for the dataset, used for logging
        input_url (str): API url, see https://developer.data.elexon.co.uk/api-details#api=prod-insol-insights-api
        from_date (str): Lower bound for the published date of the dataset, yyyy-mm-dd format
        to_date (str): Upper bound for the published date of the dataset, yyyy-mm-dd format

    Returns:
        list: List of DataFrames with the data. It's a list because the data is retrieved in chunks.
    """

    # The Elexon API requires the dates to be not more than 7 days apart
    # so we will have to retrieve the data in chunks
    dates = pd.date_range(from_date, to_date, freq="7D")
    dates = dates.to_series().dt.strftime("%Y-%m-%d").to_list()
    if dates[-1] != to_date:
        dates += [to_date]

    hdr = {"Cache-Control": "no-cache"}

    result = []
    for i in range(len(dates) - 1):

        from_dt = dates[i]
        to_dt = dates[i + 1]

        url = (
            input_url
            + f"publishDateTimeFrom={from_dt}&publishDateTimeTo={to_dt}&format=json"
        )       

        response = requests.get(url, headers=hdr)

        if response.status_code == 200:
            the_data = pd.json_normalize(response.json(), record_path="data")
            result.append(the_data)
        else:
            logger.warning(
                f"Error retrieving {name}, status code: {response.status_code}"
            )
            break

    return result


if __name__ == "__main__":
    data_dir = "data"
    get_gas_actuals_from_mipi(data_dir, "2022-01-01", "2022-07-01")
    get_ted_forecast_from_elexon(data_dir, "2022-01-01", "2022-07-01")
    get_wind_forecast_from_elexon(data_dir, "2022-01-01", "2022-07-01")
    get_electricity_actuals_from_elexon(data_dir, "2022-01-01", "2022-07-01")
