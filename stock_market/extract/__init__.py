from pathlib import Path
import sys
from dotenv import load_dotenv

__version__ = "23.0.0"

ROOT = Path(__file__).resolve().parent.parent.parent

env_path = ROOT / ".env"
load_dotenv(dotenv_path=env_path)
sys.path.append(str(ROOT))