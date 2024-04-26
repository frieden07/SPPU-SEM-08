#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>

struct Point {
    double x, y;
};

double euclidean_distance(const Point& a, const Point& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

void k_means_clustering(std::vector<Point>& points, std::vector<Point>& centroids) {
    int num_points = points.size();
    int num_centroids = centroids.size();
    std::vector<int> cluster_assignment(num_points);

    bool converged = false;
    while (!converged) {
        // Assign each point to the closest centroid
        #pragma omp parallel for
        for (int i = 0; i < num_points; ++i) {
            double min_distance = std::numeric_limits<double>::max();
            int closest_centroid = 0;

            for (int j = 0; j < num_centroids; ++j) {
                double distance = euclidean_distance(points[i], centroids[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }

            cluster_assignment[i] = closest_centroid;
        }

        // Update centroids based on cluster assignments
        std::vector<int> cluster_sizes(num_centroids, 0);
        std::vector<Point> new_centroids(num_centroids, {0, 0});

        #pragma omp parallel for
        for (int i = 0; i < num_points; ++i) {
            int cluster = cluster_assignment[i];
            #pragma omp atomic
            cluster_sizes[cluster]++;
            #pragma omp atomic
            new_centroids[cluster].x += points[i].x;
            #pragma omp atomic
            new_centroids[cluster].y += points[i].y;
        }

        for (int i = 0; i < num_centroids; ++i) {
            if (cluster_sizes[i] > 0) {
                centroids[i].x = new_centroids[i].x / cluster_sizes[i];
                centroids[i].y = new_centroids[i].y / cluster_sizes[i];
            }
        }

        // Check for convergence
        // (not parallelized for simplicity)
        converged = true;
        for (int i = 0; i < num_centroids; ++i) {
            if (new_centroids[i].x != centroids[i].x || new_centroids[i].y != centroids[i].y) {
                converged = false;
                break;
            }
        }
    }
}

int main() {
    // Example data points
    std::vector<Point> points = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};

    // Initial centroids
    std::vector<Point> centroids = {{1, 2}, {3, 4}};

    // Perform k-means clustering
    k_means_clustering(points, centroids);

    return 0;
}
