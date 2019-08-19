from pickle import load
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import *


s = 18
def autolabel(rect):
    """Attach a text label above each bar in *rects*, displaying its height."""
    height = rect.get_height()
    plt.gca().annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                size=s)

main_name = "CNN"
rewards = "food +{}, dying {}, killing +{}".format(SCORE_FOOD, SCORE_DEAD, SCORE_KILLING)
suptitle = "Statistics after 64k iterations\ngamma: {}\nrewards: {}".format(GAMMA, rewards)
s_i = [.235, .177, .136]
f_i = [.194, .192, .165]
d_i = [.003, .010, .012]
k_i = [.010, .003, .001]
names = ["CNN", "NN", "GREEDY"]
n = len(names)

width = 0.9

x = np.arange(n)

plt.figure()
plt.suptitle(suptitle, size=s)

plt.subplot(2, 2, 1)
plt.title("score / iter", size=s)
for i in range(n):
    rects = plt.bar(x[i]-width/2, s_i[i], width, label=names[i])
    autolabel(rects[0])

plt.subplot(2, 2, 2)
plt.title("food / iter", size=s)
for i in range(n):
    rects = plt.bar(x[i]-width/2, f_i[i], width, label=names[i])
    autolabel(rects[0])

plt.subplot(2, 2, 3)
plt.title("die / iter", size=s)
for i in range(n):
    rects = plt.bar(x[i]-width/2, d_i[i], width, label=names[i])
    autolabel(rects[0])

plt.subplot(2, 2, 4)
plt.title("kill / iter", size=s)
for i in range(n):
    rects = plt.bar(x[i]-width/2, k_i[i], width, label=names[i])
    autolabel(rects[0])


plt.legend(fontsize=s)
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
