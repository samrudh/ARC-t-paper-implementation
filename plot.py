import matplotlib.pyplot as plt

plt.bar(range(len(accuracies)), list(accuracies.values()), align='center')
plt.xticks(range(len(accuracies)), list(accuracies.keys()))
plt.xticks(rotation=45)

plt.show()