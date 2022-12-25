import os
import sys


def main():
    # # Run docker once to get docker container ID
    # os.system("docker run -it my_app")
    arg = sys.argv
    if "-in" not in arg:
        os.system('echo "No input specified. Program exits."')
        os.exit(1)
    idx = arg.index("-in")+1
    input_path = arg[idx]
    arg = arg[1:idx-1] + arg[idx+1:]
    print(input_path)
    arg = ' '.join(arg)
    print(arg)
    input("press enter to start docker run")
    os.system(f"docker run -v {input_path}:/tmp/CaImAn/ -t my_app:latest {arg}")
    # # Copy inputs to container
    # os.system(f"sudo docker cp {cid}:/tmp/input/")
    # Pass parameter to container
"""
docker run --name test -v "/mnt/c/Users/canyi/Desktop/CanYing/test_docker/mnt":/tmp/mnt -i -t pipeline -input "/tmp/mnt/in/case1.tif"
"""
if __name__ == "__main__":
    main()
