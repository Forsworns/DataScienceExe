import matplotlib.pyplot as plt
from configs import *

def show_baseline():
    plt.figure(1,figsize=(8,4))
    plt.subplot(121)
    plt.bar()
    plt.subplot(122)
    plt.bar()
    plt.suptitle(BASELINE)
    plt.show()


if __name__ == "__main__":
    show_baseline()