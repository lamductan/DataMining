import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def readInput(inputFilename):
	data = np.loadtxt(inputFilename, delimiter=',')
	return data

def euclideDistance(x1, x2, n):
	d = (x1 - x2).reshape([1, n])
	A = d.dot(d.T)
	return np.sqrt(A[0,0])

class KMean:
	def __init__(self, inputFilename, n, k):
		self.n = n
		self.k = k
		self.X = readInput(inputFilename)
		self.m = self.X.shape[0]
		self.Y = np.zeros(self.m, dtype = np.int)
		self.centroids = np.zeros([self.k, self.n])

	def isConverge(self, old_Y):
		return np.array_equal(self.Y, old_Y)

	def visualize(self, step=None):
		if step is None:
			fig = plt.figure('Clustering result by K-Mean')
		else:
			fig = plt.figure('Step {0}'.format(step))
		ax = plt.subplot(111)
		color = ['r', 'g', 'b']
		for i in range(self.m):
			ax.scatter(self.X[i,0], self.X[i,1], c=color[int(self.Y[i])])

		for i in range(self.k):
			ax.scatter(self.centroids[i,0], self.centroids[i,1], marker='x', c=color[i], s=150)
		plt.show()

	def train(self):
		distance = np.zeros(self.k)
		for i in range(self.k):
			#self.centroids[i] = self.X[np.random.randint(0, self.m)]
			self.centroids[i] = self.X[i]

		iStep = 0
		while True:
			print('Step %d' %(iStep))
			print(self.centroids)
			#self.visualize(iStep)
			
			old_Y = self.Y.copy()
			cnt = np.zeros(self.k)
			for i in range(self.m):
				for j in range(self.k):
					distance[j] = euclideDistance(self.X[i], self.centroids[j], self.n)
				self.Y[i] = np.argmin(distance)

			self.printResult()
			print()

			self.centroids = np.zeros([self.k, self.n])
			for i in range(self.m):
				c = self.Y[i]
				self.centroids[c] += self.X[i]
				cnt[c] += 1
			for i in range(self.k):
				for j in range(self.n):
					self.centroids[i,j] /= cnt[i]

			if self.isConverge(old_Y):
				break
			iStep += 1

	def printResult(self):
		for i in range(self.m):
			print('(', end='')
			for j in range(self.n):
				print('%d' % (self.X[i,j]), end=' ')
			print('): %d', self.Y[i])

if __name__ == '__main__':
	model = KMean('input.txt', 2, 3)
	model.train()
	#model.printResult()
	model.visualize()