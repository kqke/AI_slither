from pickle import load
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# cnn = load(open("C:\\Users\\Roee\\PycharmProjects\\AI\\project\\snake\\records\\CNN_1.pkl", "rb"))


def autolabel(rect):
    """Attach a text label above each bar in *rects*, displaying its height."""
    height = rect.get_height()
    plt.gca().annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

gamma = ".50"
rewards = "food +1, dying -5, killing +3"
suptitle = "Statistics after 64k iterations\ngamma: {}\nrewards: {}".format(gamma, rewards)
s_i = [.331, .214, -.317]
f_i = [.193, .19, .037]
d_i = [.003, .010, .072]
k_i = [.469, .000, .000]
names = ["CNN", "GREEDY", "RANDOM"]
n = len(names)

width = 0.9

x = np.arange(n)

plt.figure()
plt.suptitle(suptitle)

plt.subplot(2, 2, 1)
plt.title("score / iter")
for i in range(n):
    rects = plt.bar(x[i]-width/2, s_i[i], width, label=names[i])
    autolabel(rects[0])

plt.subplot(2, 2, 2)
plt.title("food / iter")
for i in range(n):
    rects = plt.bar(x[i]-width/2, f_i[i], width, label=names[i])
    autolabel(rects[0])

plt.subplot(2, 2, 3)
plt.title("die / iter")
for i in range(n):
    rects = plt.bar(x[i]-width/2, d_i[i], width, label=names[i])
    autolabel(rects[0])

plt.subplot(2, 2, 4)
plt.title("kill / iter")
for i in range(n):
    rects = plt.bar(x[i]-width/2, k_i[i], width, label=names[i])
    autolabel(rects[0])


plt.legend()
plt.show()

# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()



# autolabel(rects1)
# autolabel(rects2)
#
# plt.show()
