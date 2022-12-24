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
    try:
        pipeline.parse()
        pipeline.run()
    except:
        # Upon exceptions, update log so user can inspect
        if pipeline.log is not None:
            pipeline.log.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
