from src.extract_features import extract_features
from src.config import YEAR, EVENT, SESSION_TYPE

if __name__ == "__main__":
    extract_features(YEAR, EVENT, SESSION_TYPE)

