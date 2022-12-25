import os
import sys
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', required=True, type=str, help='Path to your input (folder).')

    return parser.parse_args()


def main():
    arguments = parse()
    if not os.path.exists(arguments.input):
        raise FileNotFoundError(f"")
    input_path = arguments.input

    container_name = ''
    input("press enter to start docker run")
    cmd = f'docker run --name {container_name} -v "{input_path}":/tmp/mnt -i -t pipeline {arg}'
    os.system(cmd)


if __name__ == "__main__":
    main()
