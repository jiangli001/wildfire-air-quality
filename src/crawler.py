
"""
Asynchronous web crawler for California Air Resources Board (CARB) air quality data.
Downloads PM2.5 hourly data in 2-week chunks to avoid the 80k row limit.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import logging
from urllib.parse import urlencode

import pandas as pd
import aiohttp

# pylint: disable=logging-not-lazy,logging-fstring-interpolation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Crawler:
    """Async crawler for CARB air quality and meteorological data."""

    BASE_URL = "https://www.arb.ca.gov/aqmis2/display.php"

    # Configuration for different data types
    # Add new data types here following the same structure
    DATA_TYPE_CONFIGS = {
        'PM25HR': {
            'param': 'PM25HR',
            'units': '001',
            'ptype': 'aqd',
        },
        'TEMP': {
            'param': 'TEMP',
            'units': '017',
            'ptype': 'met',
        },
        'HUMIDITY': {
            'param': 'RELHUM',
            'units': '019',
            'ptype': 'met',
        },
        'WINDSPD': {
            'param': 'WINSPD',
            'units': '011',
            'ptype': 'met',
        },
        'PKSPD': {
            'param': 'PKSPD',
            'units': '011',
            'ptype': 'met',
        },
        'PRECIP': {
            'param': 'PRECIP',
            'units': '029',
            'ptype': 'met',
        },
        'SORAD': {
            'param': 'SORAD',
            'units': '079',
            'ptype': 'met',
        },
        'DEWPNT': {
            'param': 'DEWPNT',
            'units': '017',
            'ptype': 'met',
        },
    }

    # Base parameters that remain constant across all data types
    BASE_PARAMS = {
        'filefmt': 'csv',
        'download': 'y',
        'o3area': '',
        'o3pa8': '',
        'county_name': '--COUNTY--',
        'latitude': 'A-Whole State',
        'basin': '--AIR BASIN--',
        'o3switch': 'new',
        'hours': 'all',
        'report': 'PICKDATA',
        'btnsubmit': 'Update Display',
        'qselect': 'Screened',
        'submit': 'Only+if+Checked',
        'datafmt': 'dvd',
        'statistic': '',
        'order': 'name',
    }

    def __init__(
        self,
        data_type: str = 'PM25HR',
        output_dir: str = "./data",
        max_concurrent: int = 20,
    ):
        """
        Initialize the crawler.

        Args:
            data_type: Type of data to download (e.g., 'PM25HR', 'TEMP')
            output_dir: Directory to save downloaded files
            max_concurrent: Maximum concurrent downloads

        Raises:
            ValueError: If data_type is not supported
        """
        if data_type not in self.DATA_TYPE_CONFIGS:
            available = ', '.join(self.DATA_TYPE_CONFIGS.keys())
            raise ValueError(
                f"Unsupported data_type '{data_type}'. "
                f"Available types: {available}"
            )

        self.data_type = data_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _split_date_range_by_year(
        self,
        start_date: datetime,
        end_date: datetime,
        site_id: int
    ) -> List[Tuple[datetime, datetime, int]]:
        """
        Split a date range into multiple ranges if it spans multiple years.

        Args:
            start_date: Start date
            end_date: End date
            site_id: Site ID

        Returns:
            List of (start_date, end_date, site_id) tuples, one per year
        """
        ranges = []
        current_start = start_date

        while current_start <= end_date:
            # Get the last day of the current year
            year_end = datetime(current_start.year, 12, 31)
            # Use the earlier of year_end or end_date
            current_end = min(year_end, end_date)

            ranges.append((current_start, current_end, site_id))

            # Move to January 1 of next year
            if current_end < end_date:
                current_start = datetime(current_start.year + 1, 1, 1)
            else:
                break

        return ranges

    def generate_date_ranges(self) -> List[Tuple[datetime, datetime, int]]:
        """
        Generate list of date ranges.

        Returns:
            List of (start_date, end_date, site_id) tuples
        """

        df = pd.read_csv("./site_and_fire_mapped.csv")
        cols = ['incident_dateonly_created', 'incident_dateonly_extinguished', 'site']

        df['incident_dateonly_created'] = pd.to_datetime(
            df['incident_dateonly_created'], errors='raise', format="%m/%d/%y"
        )
        df['incident_dateonly_extinguished'] = pd.to_datetime(
            df['incident_dateonly_extinguished'], errors='raise', format="%m/%d/%y"
        )
        # Filter out rows where any of the three columns is empty
        df_new = df[cols].dropna(subset=cols)
        # Convert rows to list of tuples (site_id, start_date, end_date)
        initial_ranges = list(df_new.itertuples(index=False, name=None))

        # Split ranges that span multiple years
        result = []
        for start_date, end_date, site_id in initial_ranges:
            result.extend(self._split_date_range_by_year(start_date, end_date, site_id))

        logger.info(f"Generated {len(result)} date ranges to download ({len(initial_ranges)} original ranges)")
        return result

    def build_url(self, start_date: datetime, end_date: datetime, site_id: int) -> str:
        """
        Build the download URL for a specific date range.

        Args:
            start_date: Start date
            end_date: End date
            site_id: Site ID

        Returns:
            Complete URL string
        """
        # Get data type configuration
        data_config = self.DATA_TYPE_CONFIGS[self.data_type]

        # Generate filename in the format: {PARAM}_PICKDATA_YYYY-M-D
        fname = f"{data_config['param']}_PICKDATA_{start_date.year}-{start_date.month}-{end_date.day}"
        first_date = f"{start_date.year}-{start_date.month}-{start_date.day}"

        # Start with base parameters
        params = self.BASE_PARAMS.copy()

        # Add data type specific parameters
        params.update(data_config)

        # Add date and site specific parameters
        params.update({
            'sitelist': str(site_id),
            'fname': fname,
            'first_date': first_date,
            'year': str(start_date.year),
            'start_mon': str(start_date.month),
            'start_day': str(start_date.day),
            'mon': str(end_date.month),
            'day': str(end_date.day),
        })

        return f"{self.BASE_URL}?{urlencode(params)}"

    def get_filename(self, start_date: datetime, end_date: datetime, site_id: int) -> str:
        """
        Generate filename for the downloaded data.

        Args:
            start_date: Start date
            end_date: End date
            site_id: Site ID

        Returns:
            Filename string
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{self.data_type}_{site_id}_{start_str}_to_{end_str}.csv"

    def _should_skip_download(self, filepath: Path, filename: str) -> bool:
        """
        Check if download should be skipped (file already exists).

        Args:
            filepath: Path to the file
            filename: Name of the file for logging

        Returns:
            True if file should be skipped, False otherwise
        """
        if filepath.exists():
            logger.info(f"File already exists: {filename}, skipping")
            return True
        return False

    async def _fetch_data(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> bytes:
        """
        Fetch data from URL.

        Args:
            session: aiohttp session
            url: URL to fetch

        Returns:
            Response content as bytes

        Raises:
            aiohttp.ClientError: On HTTP errors
            asyncio.TimeoutError: On timeout
        """
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as response:
            response.raise_for_status()
            return await response.read()

    def _save_content(self, filepath: Path, content: bytes, filename: str) -> None:
        """
        Save content to file.

        Args:
            filepath: Path to save the file
            content: Content to write
            filename: Name of the file for logging
        """
        with open(filepath, 'wb') as f:
            f.write(content)
        logger.info(f"Successfully saved: {filename} ({len(content)} bytes)")

    async def _handle_download_error(
        self,
        error: Exception,
        attempt: int,
        retry_count: int,
        filename: str
    ) -> bool:
        """
        Handle download error with retry logic.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-indexed)
            retry_count: Total number of retry attempts
            filename: Name of the file for logging

        Returns:
            True if should continue retrying, False if max retries reached
        """
        is_last_attempt = attempt == retry_count - 1

        if isinstance(error, asyncio.TimeoutError):
            logger.warning(f"Timeout on attempt {attempt + 1} for {filename}")
        elif isinstance(error, aiohttp.ClientError):
            logger.warning(f"HTTP error on attempt {attempt + 1} for {filename}: {error}")
        else:
            logger.error(f"Unexpected error downloading {filename}: {error}")

        if is_last_attempt:
            logger.error(f"Failed to download {filename} after {retry_count} attempts")
            return False

        # Exponential backoff
        await asyncio.sleep(2 ** attempt)
        return True

    async def _download_with_retries(
        self,
        session: aiohttp.ClientSession,
        url: str,
        filepath: Path,
        filename: str,
        start_date: datetime,
        end_date: datetime,
        retry_count: int
    ) -> bool:
        """
        Download data with retry logic.

        Args:
            session: aiohttp session
            url: URL to download from
            filepath: Path to save the file
            filename: Name of the file for logging
            start_date: Start date for logging
            end_date: End date for logging
            retry_count: Number of retries on failure

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(retry_count):
            try:
                logger.info(
                    f"Downloading {start_date.strftime('%Y/%m/%d')} to "
                    f"{end_date.strftime('%Y/%m/%d')} (attempt {attempt + 1}/{retry_count})"
                )

                content = await self._fetch_data(session, url)
                self._save_content(filepath, content, filename)
                return True

            except (asyncio.TimeoutError, aiohttp.ClientError, Exception) as e: # pylint: disable=broad-except
                should_retry = await self._handle_download_error(
                    e, attempt, retry_count, filename
                )
                if not should_retry:
                    return False

        return False

    async def download_chunk(
        self,
        session: aiohttp.ClientSession,
        start_date: datetime,
        end_date: datetime,
        site_id: int,
        retry_count: int = 3
    ) -> bool:
        """
        Download data for a specific date range.

        Args:
            session: aiohttp session
            start_date: Start date
            end_date: End date
            retry_count: Number of retries on failure

        Returns:
            True if successful, False otherwise
        """
        url = self.build_url(start_date, end_date, site_id)
        filename = self.get_filename(start_date, end_date, site_id)
        filepath = self.output_dir / filename

        if self._should_skip_download(filepath, filename):
            return True

        async with self.semaphore:
            return await self._download_with_retries(
                session, url, filepath, filename,
                start_date, end_date, retry_count
            )

    async def crawl(self):
        """
        Main crawling method - downloads all data chunks asynchronously.
        """
        date_ranges = self.generate_date_ranges()
        # date_ranges = [(datetime(2025, 11, 1), datetime(2025, 11, 2), 5210)]

        logger.info(f"Starting download of {len(date_ranges)} chunks")
        logger.info(f"Max concurrent downloads: {self.max_concurrent}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")

        # Create aiohttp session with connection pooling
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Create download tasks
            tasks = [
                self.download_chunk(session, start_date, end_date, site_id)
                for start_date, end_date, site_id in date_ranges
            ]

            # Execute all downloads
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successes and failures
            successes = sum(1 for r in results if r is True)
            failures = len(results) - successes

            logger.info(f"\n{'='*60}")
            logger.info("Download complete!")
            logger.info(f"Successful: {successes}/{len(results)}")
            logger.info(f"Failed: {failures}/{len(results)}")
            logger.info(f"Files saved to: {self.output_dir.absolute()}")
            logger.info(f"{'='*60}")


async def main():
    """
    Main entry point.
    """
    crawler = Crawler(
        data_type='WINDSPD',
        output_dir="./wind_speed",
        max_concurrent=20,
    )
    # await crawler.crawl()
    crawler.generate_date_ranges()


if __name__ == "__main__":
    asyncio.run(main())
