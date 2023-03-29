import sys
import src.src_pipeline as pipe


def main():
    pipeline = pipe.Pipeline()
    try:
        pipeline.parse()
        pipeline.run()
    finally:
        # Upon exceptions, update log so user can inspect
        if pipeline.log is not None:
            print("[INFO] Closing log...")
            pipeline.log.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
