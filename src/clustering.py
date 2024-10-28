from linear_algebra import Vector, vector_mean, squared_distance, distance
from typing import List, Tuple, NamedTuple, Union, Callable
import random, tqdm, itertools
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def num_differences(v1, v2):
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])


assert num_differences([1, 2, 3], [2, 1, 3]) == 2
assert num_differences([1, 2], [1, 2]) == 0


def cluster_means(k, inputs, assignments):
    # clusters[i] ocntains the inputs whose assigments is i
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)

    # if a cluster is empty, just use a random point
    return [
        vector_mean(cluster) if cluster else random.choice(inputs)
        for cluster in clusters
    ]


class KMeans:
    def __init__(self, k) -> None:
        self.k = k
        self.mean = None

    def classify(self, input):
        """Return the index of the cluster closest to the input"""
        return min(range(self.k), key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):
        # Start with random assignments
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Compute means and find new assignments
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]

                # Check how many assignments changed and if we're done
                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    return

                # Otherwise keep the new assignments, and compute new means
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"Changed: {num_changed} / {len(inputs)}")


class Leaf(NamedTuple):
    value: Vector


leaf1 = Leaf([10, 20])
leaf2 = Leaf([30, -15])


class Merged(NamedTuple):
    children: tuple
    order: int


merged = Merged((leaf1, leaf2), order=1)

Cluster = Union[Leaf, Merged]


def get_values(cluster):
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value for child in cluster.children for value in get_values(child)]


assert get_values(merged) == [[10, 20], [30, -15]]


def cluster_distance(cluster1, cluster2, distance_agg):
    """Compute all the pairwise distances between cluster1 and cluster2 and
    apply the aggregation function _distance_agg_ to the resulting list"""

    return distance_agg(
        [distance(v1, v2) for v1 in get_values(cluster1) for v2 in get_values(cluster2)]
    )


def get_merge_order(cluster):
    if isinstance(cluster, Leaf):
        return float("inf")
    else:
        return cluster.order


def get_children(cluster):
    if isinstance(cluster, Leaf):
        raise TypeError("Leaf has no children")
    else:
        return cluster.children


def bottom_up_cluster(inputs, distance_agg=min):
    clusters = [Leaf(input) for input in inputs]

    def pair_distance(pair):
        return cluster_distance(pair[0], pair[1], distance_agg)

    # as long as we have more than one cluster left...
    while len(clusters) > 1:
        # find the two closest clusters
        c1, c2 = min(
            (
                (cluster1, cluster2)
                for i, cluster1 in enumerate(clusters)
                for cluster2 in clusters[:i]
            ),
            key=pair_distance,
        )

        # remove them from the list of clusters
        clusters = [c for c in clusters if c != c1 and c != c2]

        # merge them, using merge_order = # of clusters left
        merged_cluster = Merged((c1, c2), order=len(clusters))

        # and add their merge
        clusters.append(merged_cluster)

    # when there's only one cluster left, return it
    return clusters[0]


def generate_clusters(base_cluster, num_clusters):
    # start with a lsit with just the base cluster
    clusters = [base_cluster]

    # as long as we don't have enough clusters yet...
    while len(clusters) < num_clusters:
        # choose the last_merged of our clusters
        next_cluster = min(clusters, key=get_merge_order)
        # remove it from the list
        clusters = [c for c in clusters if c != next_cluster]

        # and add its children to the list (i.e., unmerge it)
        clusters.extend(get_children(next_cluster))

    # once we have enough clusters
    return clusters


def main():

    inputs: List[List[float]] = [
        [-14, -5],
        [13, 13],
        [20, 23],
        [-19, -11],
        [-9, -16],
        [21, 27],
        [-49, 15],
        [26, 13],
        [-46, 5],
        [-34, -1],
        [11, 15],
        [-49, 0],
        [-22, -16],
        [19, 28],
        [-12, -8],
        [-13, -19],
        [-41, 8],
        [-11, -6],
        [-25, -9],
        [-18, -3],
    ]

    random.seed(12)  # so you get the same results as me
    clusterer = KMeans(k=3)
    clusterer.train(inputs)
    means = sorted(clusterer.means)  # sort for the unit test

    assert len(means) == 3

    # Check that the means are close to what we expect.
    assert squared_distance(means[0], [-44, 5]) < 1
    assert squared_distance(means[1], [-16, -10]) < 1
    assert squared_distance(means[2], [18, 20]) < 1

    random.seed(0)
    clusterer = KMeans(k=2)
    clusterer.train(inputs)
    means = sorted(clusterer.means)

    assert len(means) == 2
    assert squared_distance(means[0], [-26, -5]) < 1
    assert squared_distance(means[1], [18, 20]) < 1

    from matplotlib import pyplot as plt

    def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
        """finds the total squared error from k-means clustering the inputs"""
        clusterer = KMeans(k)
        clusterer.train(inputs)
        means = clusterer.means
        assignments = [clusterer.classify(input) for input in inputs]

        return sum(
            squared_distance(input, means[cluster])
            for input, cluster in zip(inputs, assignments)
        )

    # now plot from 1 up to len(inputs) clusters

    ks = range(1, len(inputs) + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]

    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.title("Total Error vs. # of Clusters")
    # plt.show()

    plt.gca().clear()

    image_path = r"girl_with_book.jpg"  # wherever your image is


    # .tolist() converts a numpy array to a Python list

    clusterer = KMeans(5)

    def recolor(pixel: Vector) -> Vector:
        cluster = clusterer.classify(pixel)  # index of the closest cluster
        return clusterer.means[cluster]  # mean of the closest cluster

    base_cluster = bottom_up_cluster(inputs)

    three_clusters = [
        get_values(cluster) for cluster in generate_clusters(base_cluster, 3)
    ]

    # sort smallest to largest
    tc = sorted(three_clusters, key=len)
    assert len(tc) == 3
    assert [len(c) for c in tc] == [2, 4, 14]
    assert sorted(tc[0]) == [[11, 15], [13, 13]]

    plt.close()

    for i, cluster, marker, color in zip(
        [1, 2, 3], three_clusters, ["D", "o", "*"], ["r", "g", "b"]
    ):
        xs, ys = zip(*cluster)  # magic unzipping trick
        plt.scatter(xs, ys, color=color, marker=marker)

        # put a number at the mean of the cluster
        x, y = vector_mean(cluster)
        plt.plot(x, y, marker="$" + str(i) + "$", color="black")

    plt.title("User Locations -- 3 Bottom-Up Clusters, Min")
    plt.xlabel("blocks east of city center")
    plt.ylabel("blocks north of city center")
    # plt.show()

    plt.gca().clear()
    plt.close()

    base_cluster_max = bottom_up_cluster(inputs, max)
    three_clusters_max = [
        get_values(cluster) for cluster in generate_clusters(base_cluster_max, 3)
    ]

    for i, cluster, marker, color in zip(
        [1, 2, 3], three_clusters_max, ["D", "o", "*"], ["r", "g", "b"]
    ):
        xs, ys = zip(*cluster)  # magic unzipping trick
        plt.scatter(xs, ys, color=color, marker=marker)

        # put a number at the mean of the cluster
        x, y = vector_mean(cluster)
        plt.plot(x, y, marker="$" + str(i) + "$", color="black")

    plt.title("User Locations -- 3 Bottom-Up Clusters, Max")
    plt.xlabel("blocks east of city center")
    plt.ylabel("blocks north of city center")
    plt.gca().clear()


if __name__ == "__main__":
    main()
