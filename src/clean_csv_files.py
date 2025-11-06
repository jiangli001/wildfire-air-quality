"""
Script to clean CSV files by removing the "Quality Flag Definition" section
and any trailing empty lines.
"""

from sys import argv
from pathlib import Path
import logging

# pylint: disable=logging-not-lazy,logging-fstring-interpolation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FILE_DIR = argv[1]

def clean_csv_file(filepath: Path) -> bool:
    """
    Remove "Quality Flag Definition" section and trailing empty lines from a CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Read all lines from the file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the index where "Quality Flag Definition" starts
        cleaned_lines = []
        for line in lines:
            # Stop when we encounter the "Quality Flag Definition" line
            if "Quality Flag Definition" in line:
                break
            cleaned_lines.append(line)

        # Remove trailing empty lines
        while cleaned_lines and cleaned_lines[-1].strip() == '':
            cleaned_lines.pop()

        # Ensure the last line ends with a newline
        if cleaned_lines and not cleaned_lines[-1].endswith('\n'):
            cleaned_lines[-1] += '\n'

        # Check if file was modified
        original_count = len(lines)
        cleaned_count = len(cleaned_lines)

        if original_count == cleaned_count:
            logger.info(f"No changes needed for {filepath.name}")
            return False

        # Write the cleaned content back to the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        removed_lines = original_count - cleaned_count
        logger.info(
            f"Cleaned {filepath.name}: removed {removed_lines} lines "
            f"({original_count} -> {cleaned_count})"
        )
        return True

    except Exception as e: # pylint: disable=broad-except
        logger.error(f"Error processing {filepath.name}: {e}")
        return False


def clean_all_csv_files(data_dir: str) -> None:
    """
    Clean all CSV files in the specified directory.

    Args:
        data_dir: Path to the directory containing CSV files
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Directory not found: {data_path}")
        return

    # Find all CSV files
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {data_path}")
        return

    logger.info(f"Found {len(csv_files)} CSV files to process")
    logger.info(f"{'='*60}")

    # Process each file
    modified_count = 0
    for csv_file in sorted(csv_files):
        if clean_csv_file(csv_file):
            modified_count += 1

    # Summary
    logger.info(f"{'='*60}")
    logger.info("Processing complete!")
    logger.info(f"Total files processed: {len(csv_files)}")
    logger.info(f"Files modified: {modified_count}")
    logger.info(f"Files unchanged: {len(csv_files) - modified_count}")


if __name__ == "__main__":
    clean_all_csv_files(FILE_DIR)
