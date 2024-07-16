from GUI import GUI
from multiprocessing import freeze_support


def main():
    app = GUI()
    app.gui()


if __name__ == "__main__":
    freeze_support()
    main()
