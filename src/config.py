from dotenv import load_dotenv
import os

load_dotenv()

REDIS_HOST = os.environ.get("REDISHOST", "localhost")
REDIS_PORT = int(os.environ.get("REDISPORT", 6379))

CASHED_TTL_SECONDS = 30
