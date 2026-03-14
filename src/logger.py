# ============================================================
# AeroGuard — Central Logger
# Yeh file poore project mein logging handle karti hai
# Do jagah simultaneously log hoga:
#   1. Terminal (color ke saath — development mein helpful)
#   2. logs/ folder mein daily rotating files
# Usage: from src.logger import logger
#        logger.info("Kuch hua")
#        logger.error("Kuch galat hua")
# ============================================================

import os
import sys
from loguru import logger
from dotenv import load_dotenv
import yaml

# .env file se environment variables load karo
load_dotenv()

# .env se logs directory path lo, agar nahi mili toh default "./logs"
LOGS_DIR = os.getenv("LOGS_DIR", "./logs")

# logs directory banao agar exist nahi karti
# exist_ok=True matlab agar pehle se hai toh error mat do
os.makedirs(LOGS_DIR, exist_ok=True)

# config.yaml se logging level padho
# iska fayda — sirf config.yaml change karo, code nahi
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

LOG_LEVEL = config["logging"]["level"]  # "DEBUG" ya "INFO"

# Loguru ka default handler remove karo
# Warna duplicate logs aayenge
logger.remove()

# ---- Handler 1: Terminal (Console) ----
# Yeh developer ko real-time colored output deta hai
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    # Format explanation:
    # {time} = timestamp | {level} = DEBUG/INFO/ERROR
    # {name} = file ka naam | {message} = actual log message
    # <green>, <cyan> = loguru color tags — terminal mein color dikhate hain
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> | "
        "<white>{message}</white>"
    ),
    colorize=True   # Windows terminal mein bhi colors enable karo
)

# ---- Handler 2: File (logs/ folder) ----
# Yeh permanent record ke liye hai — debugging aur audit ke liye
logger.add(
    # Har din ki alag file — aeroguard_2025-01-15.log
    os.path.join(LOGS_DIR, "aeroguard_{time:YYYY-MM-DD}.log"),
    level=LOG_LEVEL,
    # File mein color tags nahi — plain text
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name} | "
        "{message}"
    ),
    rotation="10 MB",       # 10MB hone pe nayi file shuru
    retention="30 days",    # 30 din se purani files delete
    compression="zip"       # purani files compress karo disk space bachane
)