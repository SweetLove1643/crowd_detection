# cấu hình log + che credential trong log
import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

FORMAT = "%(asctime)s | %(levelname)-7s | %(threadName)-16s | %(name)-14s | %(message)s"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB / file
BACKUP_COUNT = 3



class MaskCredentials(logging.Filter):
    """Che user:pass trong URL (rtsp://user:pass@...) ở MỌI dòng log."""
    _pat = re.compile(r"(\w+://)[^@/\s]+@")



    def filter(self, record):
        record.msg = self._pat.sub(r"\1***:***@", record.getMessage())
        record.args = None
        return True

def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter(FORMAT)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(MaskCredentials())

    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    console.setFormatter(formatter)
    console.addFilter(MaskCredentials())

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.addHandler(file_handler)
    root.addHandler(console)

def log_startup_banner() -> None:
    logging.getLogger("main").info(
        "===== Khởi động hệ thống (pid=%d, Python %s) =====",
        os.getpid(),
        sys.version.split()[0],
    )
