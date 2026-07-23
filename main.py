import logging
from src.pocketbase_client import PocketBaseExtractor

logger = logging.getLogger("ETL_Logger")

def main():
    logger.info("Starting ETL Pipeline")

    extractor = PocketBaseExtractor()
    extractor.run_extract()

    from src import data_processing


    logger.info("ETL pipeline completed")

if __name__ == "__main__":
    main()