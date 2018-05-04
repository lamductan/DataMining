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

class KMedoids:
	def __init__(self, inputFilename, n, k):
		self.n = n
		self.k = k
		self.X = readInput(inputFilename)
		self.m = self.X.shape[0]
		self.Y = np.zeros(self.m, dtype = np.int)
		self.isMedoid = np.zeros(self.m)

		#a = np.arange(self.m)
		#np.random.shuffle(a)
		#self.medoids = a[:self.k]
		self.medoids = np.array([15,9,5])

		for idx in self.medoids:
			self.isMedoid[idx] = 1

		self.distance = np.zeros([self.m, self.m])
		for i in range(self.m):
			for j in range(self.m):
				self.distance[i,j] = euclideDistance(self.X[i], self.X[j], self.n)
		for i in range(self.m):
			self.Y[i] = self.assign(i)

		self.cost = self.computeCost()

	def isConverge(self, old_Medoids):
		return np.array_equal(self.medoids, old_Medoids)

	def visualize(self, step=None):
		if step is None:
			fig = plt.figure('Clustering result by K-Medoids')
		else:
			fig = plt.figure('Step {0}'.format(step))
		ax = plt.subplot(111)
		color = ['r', 'g', 'b']
		for i in range(self.m):
			ax.scatter(self.X[i,0], self.X[i,1], c=color[int(self.Y[i])])

		for i in range(self.k):
			ax.scatter(self.X[self.medoids[i],0], self.X[self.medoids[i],1], marker='x', c=color[i], s=150)
		plt.show()

	def assign(self, idx):
		dis = np.zeros(self.k)
		for i in range(self.k):
			dis[i] = self.distance[idx, self.medoids[i]]
		return np.argmin(dis)

	def computeCost(self):
		cost = 0.0
		for i in range(self.m):
			cost += self.distance[i, self.medoids[self.Y[i]]]
		return cost

	def train(self):
		distance = np.zeros(self.k)
			
		iStep = 0
		f = open('output-kmedoids.csv', 'w')
		self.printResult1(f, iStep)

		while True:
			iStep += 1
			self.visualize(iStep)
			
			old_Medoids = self.medoids.copy()
			for i in range(self.k):
				medoids = self.medoids[i]
				for o in range(self.m):
					old_Y = self.Y.copy()
					if self.isMedoid[o] == 1:
						continue
					else:
						self.isMedoid[medoids] = 0
						self.isMedoid[o] = 1
						self.medoids[i] = o
						for j in range(self.m):
							self.Y[j] = self.assign(j)
						cost = self.computeCost()
						if cost < self.cost:
							self.cost = cost
							medoids = o
						else:
							self.isMedoid[medoids] = 1
							self.isMedoid[o] = 0
							self.medoids[i] = medoids
							self.Y = old_Y.copy()

			self.printResult1(f, iStep)
			print(file = f)

			if self.isConverge(old_Medoids):
				break

	def printResult(self):
		print('Medoids:')
		for i in range(self.k):
			print('M[%d] = (%d, %d)' % (i, self.X[self.medoids[i],0], self.X[self.medoids[i],1]))
		print('Label:')
		for i in range(self.m):
			print('(', end='')
			for j in range(self.n):
				print('%d' % (self.X[i,j]), end=' ')
			print('): %d', self.Y[i])

	def printResult1(self, outputFile, step = None):
		f = outputFile
		if step != None:
			print('Step: %d' %(step), file = f)

		print('Medoids', file = f)
		for i in range(self.k):
			print('%d,%d,%d' % (self.medoids[i], self.X[self.medoids[i],0], self.X[self.medoids[i],1]), file = f)
		print('Cost=%0.3f' % (self.cost), file = f)
		print('Label:', file = f)
		print('x,y,label', file = f)
		for i in range(self.m):
			for j in range(self.n):
				print('%d' % (self.X[i,j]), end=',', file = f)
			print(self.Y[i], file = f)

		print(file = f)

if __name__ == '__main__':
	model = KMedoids('input.txt', 2, 3)
	model.train()
	#model.printResult()
	model.visualize()


#	legend('Cluster 1', 'Cluster 2', 'Cluster 3',... 
 #      'Medoids', 'Location', 'NW');
