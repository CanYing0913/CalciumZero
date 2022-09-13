

if __name__ == "__main__":
    with open("testBrain_eddy.ecclog", "r") as fin:
        inlines = fin.readlines()

    with open("testBrain_eddy_log.txt", "w") as fout:
        count = 0
        for i, inline in enumerate(inlines):
            if "Final result: " in inline:
                fout.write(inlines[i + 1])
                fout.write(inlines[i + 2])
                fout.write(inlines[i + 3])
                fout.write(inlines[i + 4])
                fout.write("separate\n")
                i += 4
