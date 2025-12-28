from typing import List, Dict, Optional, Any
import asyncio
import aiohttp
import pandas as pd

async def make_async_api_request(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    error_context: str = ""
) -> Optional[Dict]:
    """
    Generic async API request function with comprehensive error handling.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                # Handle specific 400 error with more detail if possible
                if response.status == 400:
                    print(f"API rejected request (400). Context: {error_context}")
                    return None

                response.raise_for_status()
                data = await response.json()
                return data

    except aiohttp.ClientResponseError as http_err:
        print(f"HTTP error occurred: {http_err} - Status Code: {http_err.status}")
        if error_context:
            print(f"Context: {error_context}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if error_context:
            print(f"Context: {error_context}")
        return None


async def get_loc_ids(api_key: str, query: str) -> List[str]:
    """
    Calls the Weather.com location search API and extracts location info 
    INCLUDING locationType to help filter valid stations.
    """
    base_url = "https://api.weather.com/v3/location/search"
    params = {
        "apiKey": api_key,
        "language": "en-US",
        "query": query,
        "locationType": "city,airport,postCode,pws",
        "format": "json"
    }

    data = await make_async_api_request(
        url=base_url,
        params=params,
        error_context=f"location search query: {query}"
    )


    if not data:
        return []
    return data['location']['locId']



async def get_historical_weather_data(
        loc_id: str,
        api_key: str,
        start_date: str,
        units: str = "e"
    ) -> Optional[Dict]:
    """
    Calls the Weather.com API to get historical weather observations.
    Requires a station-based LocID (Airport/PWS) or supported PostCode.
    """
    url = f"https://api.weather.com/v1/location/{loc_id}/observations/historical.json"

    params = {
        "apiKey": api_key,
        "units": units,
        "startDate": start_date
    }

    headers = {"Accept": "application/json"}

    data = await make_async_api_request(
        url=url,
        params=params,
        headers=headers,
        error_context=f"historical weather for loc_id: {loc_id}, date: {start_date}"
    )

    return data


def historical_weather_to_dataframe(weather_data: Dict) -> pd.DataFrame:
    """Transforms historical weather JSON response into a pandas DataFrame."""
    if not weather_data or "observations" not in weather_data:
        print("No observations found in weather data")
        return pd.DataFrame()

    observations = weather_data["observations"]
    return pd.DataFrame(observations)


async def fetch_and_process_weather(
    api_key: str,
    query: str,
    start_date: str,
    units: str = "e"
) -> Optional[pd.DataFrame]:
    """
    Complete workflow: Smartly selects the best location ID (Airport/PWS) 
    to ensure historical data availability.
    """
    print(f"Searching for location: {query}")
    locations = await get_loc_ids(api_key, query)

    if not locations:
        print(f"No locations found for query: {query}")
        return None

    loc_id = locations[0]

    weather_data = await get_historical_weather_data(loc_id, api_key, start_date, units)

    if not weather_data:
        print("Failed to fetch weather data")
        return None

    df = historical_weather_to_dataframe(weather_data)
    df.to_csv(f"historical_weather_{loc_id}_{start_date}.csv", index=False)

    return df


async def main():
    # Use your actual key here
    API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
    START_DATE = "20251123"

    queries = [
        "34.40318098412745, -110.82531176461298",              # Zipcode (Should pick postCode type)
        # "London",             # City (Should now auto-select an Airport like Heathrow)
        # "34.0522,-118.2437"   # Coordinates (Should now pick nearest Airport/PWS)
    ]

    tasks = [fetch_and_process_weather(API_KEY, query, START_DATE) for query in queries]
    results = await asyncio.gather(*tasks)

    for query, df in zip(queries, results):
        print(f"\n{'='*60}")
        print(f"Results for query: {query}")
        print(f"{'='*60}")
        if df is not None and not df.empty:
            print(f"Shape: {df.shape}")
            print(f"First row station: {df.iloc[0].get('obs_name', 'Unknown')}")
        else:
            print("No data available")

if __name__ == "__main__":
    asyncio.run(main())
