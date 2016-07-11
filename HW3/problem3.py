from sklearn.cluster import KMeans, SpectralClustering
from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc


#Kmeans segmentation
#Tree image color clustering
trees = misc.imread('trees.png')
trees = np.array(trees, dtype=np.float64) / 255
w, h, d = original_shape = tuple(trees.shape)
assert d == 3
image_array = np.reshape(trees, (w * h, d))

def getLabels(n_colors, image_array):
	kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
	labels = kmeans.predict(image_array)
	return kmeans, labels

def newImage(kmeans, labels, w, h):
    d = kmeans.cluster_centers_.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = kmeans.cluster_centers_[labels[label_idx]]
            label_idx += 1
    return image

kmeansK3, labelsK3 = getLabels(3, image_array)
kmeansK5, labelsK5 = getLabels(5, image_array)
kmeansK10, labelsK10 = getLabels(10, image_array)

def showImage():
	plt.figure(1)
	plt.clf()
	plt.axis('off')
	plt.title('Image after Kmeans (k = 3)')
	plt.imshow(newImage(kmeansK3, labelsK3, w, h))
	plt.figure(2)
	plt.clf()
	plt.axis('off')
	plt.title('Image after Kmeans (k = 5)')
	plt.imshow(newImage(kmeansK5, labelsK5, w, h))
	plt.figure(3)
	plt.clf()
	plt.axis('off')
	plt.title('Image after Kmeans (k = 10)')
	plt.imshow(newImage(kmeansK10, labelsK10, w, h))
	plt.show()



#Spectral Clustering Vs kmeans

samplesNum = 2000
noisy_circles = datasets.make_circles(n_samples=samplesNum, factor=.5,noise=.05)
noisy_moons = datasets.make_moons(n_samples=samplesNum, noise=.05)


colors = np.array([x for x in 'rg'])
colors = np.hstack([colors] * 20)

clusteringType = ['KMeans', 'Spectral Clustering']
datasets = [noisy_circles, noisy_moons]
i = 1
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    # create clustering estimators
    two_means = cluster.KMeans(n_clusters=2)
    spectral = cluster.SpectralClustering(n_clusters=2,eigen_solver='arpack',affinity="nearest_neighbors")
    clustering_algorithms = [two_means, spectral]

    plt.figure(i)
    plt.subplot(3, 1, 1)
    plt.title('Original data', size=12)
    plt.scatter(X[:, 0], X[:, 1], color=colors[y].tolist(), s=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(()) 

    i += 1
    plot_num = 2

    for name, algorithm in zip(clusteringType, clustering_algorithms):
        # predict cluster memberships
        algorithm.fit(X)
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)
        # plot        
        plt.subplot(3, 1, plot_num)
        plt.title(name, size=12)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

plt.show()




