#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

struct Point {
    double x;
    double y;
    int label;
};

double distance(const Point& p1, const Point& p2) {
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

int knn(const std::vector<Point>& dataset, const Point& query, int k) {
    std::vector<std::pair<double, int>> distances; // (distance, index) pairs

    // Calculate distances from query point to all points in the dataset
    #pragma omp parallel for
    for (size_t i = 0; i < dataset.size(); ++i) {
        double dist = distance(dataset[i], query);
        #pragma omp critical
        distances.push_back({dist, static_cast<int>(i)});
    }

    // Sort distances in ascending order
    std::sort(distances.begin(), distances.end());

    // Count the labels of the k nearest neighbors
    std::vector<int> labelCounts(2, 0); // Assuming binary classification
    for (int i = 0; i < k; ++i) {
        int index = distances[i].second;
        int label = dataset[index].label;
        labelCounts[label]++;
    }

    // Determine the majority label among the k nearest neighbors
    int majorityLabel = (labelCounts[0] > labelCounts[1]) ? 0 : 1;

    return majorityLabel;
}

int main() {
    std::vector<Point> dataset = {
        {2.0, 3.0, 0},
        {5.0, 4.0, 0},
        {9.0, 6.0, 0},
        {3.0, 7.0, 1},
        {8.0, 1.0, 1},
        {7.0, 2.0, 1}
    };

    Point query = {4.0, 5.0, -1}; // Query point with unknown label
    int k = 3; // Number of nearest neighbors to consider

    int predictedLabel = knn(dataset, query, k);

    std::cout << "Predicted label for query point: " << predictedLabel << std::endl;

    return 0;
}
