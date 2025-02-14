import matplotlib.pyplot as plt

data = open("mnist_train.csv", "r")

# read second line
data.readline()
line = data.readline().split(',')

# print label
print(line[0])

# plot data
a = [[int(line[j + i * 28 + 1])/255 for j in range(28)] for i in range(28)]
plt.imshow(a, cmap='binary')
plt.show()

data.close()