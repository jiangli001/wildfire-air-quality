"""
Asynchronous Open-Meteo API client for fetching historical weather data.
"""
import logging
from typing import List, Dict, Tuple, Optional
import asyncio
import aiohttp
import pandas as pd
from tenacity import (
    retry,
    stop_after_delay,
    retry_if_exception_type,
    before_sleep_log,
    wait_fixed
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_delay(10000),
    wait=wait_fixed(60),
    retry=(
        retry_if_exception_type(aiohttp.ClientError) |
        retry_if_exception_type(asyncio.TimeoutError) |
        retry_if_exception_type(aiohttp.ServerTimeoutError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def fetch_weather_data(
    session: aiohttp.ClientSession,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    site_name: str,
    site_id: float,
    hourly_vars: Optional[List[str]] = None,
    timezone: str = "America/Los_Angeles",
    models: Optional[List[str]] = None
) -> Dict:
    """
    Fetch weather data from Open-Meteo API asynchronously with retry mechanism.

    Retries up to 5 times with exponential backoff (1s, 2s, 4s, 8s, 10s) on:
    - Network/connection errors
    - Timeout errors
    - Server errors (5xx)
    - Rate limiting (429)

    Args:
        session: aiohttp client session
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
        hourly_vars: List of hourly variables to fetch
        timezone: Timezone for the data
        models: List of models to use

    Returns:
        Dictionary containing the API response

    Raises:
        aiohttp.ClientError: After all retry attempts are exhausted
    """
    if hourly_vars is None:
        hourly_vars = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "precipitation", "wind_speed_10m", "wind_direction_10m",
            "direct_radiation_instant", "direct_radiation",
            "pressure_msl", "surface_pressure", "cloud_cover"
        ]

    if models is None:
        models = ["best_match"]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "models": ",".join(models),
        "timezone": timezone,
    }

    timeout = aiohttp.ClientTimeout(total=30)
    logger.info("Fetching weather data for lat=%s, lon=%s, start=%s, end=%s",\
                latitude, longitude, start_date, end_date)
    async with session.get(url, params=params, timeout=timeout) as response:
        # Retry on 5xx server errors and 429 rate limiting
        if response.status >= 500 or response.status == 429:
            response.raise_for_status()

        # For other errors (4xx), raise immediately without retry
        response.raise_for_status()
        json_response = await response.json()
        json_response['site_name'] = site_name
        json_response['site_id'] = site_id
        return json_response


def json_to_df(data: Dict) -> pd.DataFrame:
    """
    Parse Open-Meteo API JSON response into a pandas DataFrame.

    Args:
        data: JSON response from the API

    Returns:
        pandas DataFrame with time index and weather variables as columns
    """
    hourly = data.get('hourly', {})
    df = pd.DataFrame(hourly)
    df['time'] = pd.to_datetime(df['time'])
    df['site_name'] = data.get('site_name', '')
    df['site_id'] = data.get('site_id', None)
    return df


async def fetch_multiple_locations(
    coordinates: List[Tuple[float, float, str, str, str, float]]
) -> List[pd.DataFrame]:
    """
    Fetch weather data for multiple locations/date ranges asynchronously.

    Args:
        coordinates: List of tuples containing (latitude, longitude, start_date, end_date)

    Returns:
        List of pandas DataFrames, one for each location/date range
    """
    connector = aiohttp.TCPConnector(limit=20) # Limit concurrent connections
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_weather_data(session, lat, lon, start, end, site_name=site_name, site_id=site_id)
            for lat, lon, start, end, site_name, site_id in coordinates
        ]
        responses = await asyncio.gather(*tasks)

    return [json_to_df(response) for response in responses]


async def main():
    """Example usage of the Open-Meteo API client."""
    logger.info("Reading fire incidents data from csv.")
    df = pd.read_csv("../data/processed_fire_incidents.csv")
    df["start_date"] = pd.to_datetime(df["start_date"], format="%m/%d/%y").dt.strftime("%Y-%m-%d")
    df["end_date"] = pd.to_datetime(df["end_date"], format="%m/%d/%y").dt.strftime("%Y-%m-%d")

    # convert to list of tuples
    logger.info("Converting fire incidents data to list of tuples.")
    locations = list(df.itertuples(index=False, name=None))
    print(locations)
    logger.info("Fetching weather data for multiple locations.")
    dataframes = await fetch_multiple_locations(locations)
    logger.info("Saving weather data to csv.")
    pd.concat(dataframes).to_csv("../data/weather_data_new.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())
