from collections import Counter, defaultdict
from linear_algebra import distance, Vector
from typing import NamedTuple, Dict
import random, requests, csv
from matplotlib import pyplot as plt
from machine_learning import split_data, train_test_split
import tqdm
from sklearn import datasets

def raw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

assert raw_majority_vote(['a','b','c','b']) == 'b'

def majority_vote(labels):
    """Assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])
    
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])

assert majority_vote(['a','b','c','b','a']) == 'b'

from linear_algebra import distance, Vector

class LabeledPoint(NamedTuple):
    point: Vector
    label: str
    
def knn_classify(k, labeled_points, new_point):
    # Order the labeled points from nearest to farthest.
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))
    
    # Find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    # and let them vote.
    return majority_vote(k_nearest_labels)

def random_point(dim):
    return [random.random() for _ in range(dim)]

def random_distances(dim, num_pairs):
    return [distance(random_point(dim), random_point(dim))
            for _ in range(num_pairs)]
    
def main():
    iris = datasets.load_iris()
    iris_data = [parse_iris_row(row) for row in iris]
    #print(iris)
    def parse_iris_row(row):
        """
        sepal_length, sepal_width, petal_length, petal_width, class
        """
        measurements = [float(value) for value in row[:-1]]
        label = row[-1].split("-")[-1]
        
        return LabeledPoint(measurements, label)
        
        
     # We'll also group just the points by species/label so we can plot them.
    points_by_species = defaultdict(list)
    for iris in iris_data:
        points_by_species[iris.label].append(iris.point)
        
    metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
    pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
    marks = ['+', '.', 'x']  # we have 3 classes, so 3 markers
    
    fig, ax = plt.subplots(2, 3)
    
    for row in range(2):
        for col in range(3):
            i, j = pairs[3 * row + col]
            ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
    
            for mark, (species, points) in zip(marks, points_by_species.items()):
                xs = [point[i] for point in points]
                ys = [point[j] for point in points]
                ax[row][col].scatter(xs, ys, marker=mark, label=species)
    
    ax[-1][-1].legend(loc='lower right', prop={'size': 6})
    plt.show()
    
    plt.savefig('../images/iris_scatter.png')
    plt.gca().clear()
    
    random.seed(12)
    iris_train, iris_test = split_data(iris_data, 0.7)
    assert len(iris_train) == 0.7 * 150
    assert len(iris_test) == 0.3 * 150
    
    # track how many times we see (predicted, actual)
    confusion_matrix = defaultdict(int)
    num_correct = 0
    
    for iris in iris_test:
        predicted = knn_classify(5, iris_train, iris.point)
        actual = iris.label
        
        if predicted == actual:
            num_correct += 1
            
        confusion_matrix[(predicted, actual)] += 1
        
    pct_correct = num_correct / len(iris_test)
    print(pct_correct, confusion_matrix)
    
    dimesions = range(1, 101)
    
    avg_distances = []
    min_distances = []
    
    random.seed(0)
    for dim in tqdm.tqdm(dimesions, desc="Curse of dimensionality"):
        distances = random_distances(dim, 10000)
        avg_distances.append(sum(distances) / 10000)
        min_distances.append(min(distances))
        
    min_avg_ratio = [min_dist / avg_dist for min_dist, avg_dist in zip(min_distances, avg_distances)]
    
if __name__ == "__main__": main()    