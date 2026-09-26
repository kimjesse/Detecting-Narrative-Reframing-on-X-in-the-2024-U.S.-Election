from DataFiles.preprocesssing import run_pipeline
import logging

if __name__ == "__main__":
    logging.basicConfig(filename='../example.log', level=logging.DEBUG)
    run_pipeline()