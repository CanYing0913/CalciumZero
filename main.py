import sys
from time import time

# from src import src_detection as s0
# from src import src_stabilizer as s1
# from src import src_caiman as s2
from src import src
import os
from pathlib import Path


def main():
    """
    Main pipeline function.
    """
    pipeline = src.pipeline()
    pipeline.parse()
    pipeline.run()


if __name__ == "__main__":
    main()
    sys.exit(0)
