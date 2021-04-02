import matplotlib.pyplot as plt


def parse_files(fp):
    loss_arr = []
    accuracy_arr = []

    with open(fp, "r+") as data:
        lines = data.readlines()
        for ix, line in enumerate(lines):
            line = line.strip()
            if "accuracy: " in line:
                line = line.split(", ")
                try:
                    _, loss, acc = line
                    loss = float(loss.split(': ')[-1])
                    accuracy = float(acc.split(': ')[-1])
                    loss_arr.append(loss)
                    accuracy_arr.append(accuracy)
                except:
                    pass
    return loss_arr, accuracy_arr

def plot_lsts(loss, accuracy, title):
    steps = [x for x in range(len(loss))]
    plt.plot(steps, loss, label="Loss")
    plt.title(f"{title}")
    plt.xlabel("Steps")
    plt.ylabel("Loss/Accuracy")
    plt.plot(steps, accuracy, label="Accuracy")
    plt.legend()
    plt.savefig(f"{title}.jpg", format='jpg', dpi=1200)
    plt.show()

loss_arr, accuracy_arr = parse_files("/Users/hamnamoieez/Desktop/Projects/self-attention-image-recognition/code/pairwise.txt")
plot_lsts(loss_arr, accuracy_arr, "Pairwise_SAN10")