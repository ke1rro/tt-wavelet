LENGTH = 10000000

with open("pad_signal.txt", "w") as f:
    lst = [str(i) for i in range(1, LENGTH + 1)]
    f.write(" ".join(lst))
