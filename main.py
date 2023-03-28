import sys
from time import time

import src.src_pipeline
import src.src_pipeline as pipe
import os
from pathlib import Path


def main():
    """
    Main pipeline function.
    """
    pipeline = pipe.Pipeline()
#     try:
    pipeline.parse()
    pipeline.run()
#     except:
#         print(f"{sys.exc_info()[0]} occurred.")
    # Upon exceptions, update log so user can inspect
    if pipeline.log is not None:
        print("[INFO] Closing log...")
        pipeline.log.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
