import logging
import sys
from src.pocketbase_client import PocketBaseExtractor

logger = logging.getLogger("ETL_Logger")

def main():
    logger.info("Starting ETL Pipeline")
    logger.info("-" * 50)

    try:
        extractor = PocketBaseExtractor()
        extractor.run_extract()

        from src import data_processing


        logger.info("ETL pipeline completed")

    except Exception as exception:
        logger.critical(f"ETL pipeline failed with error: {exception}", exc_info = True)
        sys.exit(1)

if __name__ == "__main__":
    main()