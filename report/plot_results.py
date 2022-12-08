import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_file = 'report/results.csv'
df = pd.read_csv(path_file)


# Set figure default figure size
plt.rcParams["figure.figsize"] = (10, 6)

fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(211)
ax1.set_ylabel('Floor hit rate [%]')
# ax1.set_xlabel('Dataset')
ax1.set_title('Classification Results')

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2 * np.pi * t)
ax1.plot(df.iloc[:, 0], df.iloc[:, 1], linestyle=":", marker="o", label="kNN")
ax1.plot(df.iloc[:, 0], df.iloc[:, 2], linestyle="-.", marker="o", label="CNNLoc")
ax1.plot(df.iloc[:, 0], df.iloc[:, 3], linestyle="--", marker="o", label="ELM")
ax1.plot(df.iloc[:, 0], df.iloc[:, 4], linestyle="-", marker="o", label="CNN-ELM")
# ax1.legend(loc='lower left')

# Fixing random state for reproducibility
np.random.seed(19680801)

ax2 = fig.add_subplot(212)
ax2.plot(df.iloc[:, 0], df.iloc[:, 5], linestyle=":", marker="o", label="kNN")
ax2.plot(df.iloc[:, 0], df.iloc[:, 6], linestyle="-.", marker="o", label="CNNLoc")
ax2.plot(df.iloc[:, 0], df.iloc[:, 7], linestyle="--", marker="o", label="ELM")
ax2.plot(df.iloc[:, 0], df.iloc[:, 8], linestyle="-", marker="o", label="CNN-ELM")
ax2.set_yscale('log')
ax2.set_xlabel('Dataset')
ax2.set_ylabel('Time [sec]')

# Put a legend below current axis
plt.subplots_adjust(bottom=0.2, top=0.90)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=5)

plt.savefig('fig_ClassificationResults.pdf')
plt.show()
