import logging
import os
import datetime

log_dir = "logs"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
log_file_name = datetime.datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")
log_file_path = os.path.join(log_dir, log_file_name)


logging.basicConfig(
    level = logging.DEBUG,
    format= "%(filename)s - %(asctime)s - %(levelname)s - %(funcName)s - Line %(lineno)d - %(message)s",
    datefmt="%d-%b-%Y %H:%M:%S",
    handlers= [
        logging.FileHandler(log_file_path, mode="w"), ## Create log file for each run
        logging.StreamHandler()  ## Print log to console
    ]
)

sam_logger = logging.getLogger("SAM_Logger")
sam_logger.info("SAM model Logger has been started.")