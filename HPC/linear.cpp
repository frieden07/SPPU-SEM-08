#include <iostream>
#include <vector>
#include <omp.h>

void linear_regression(std::vector<double>& x, std::vector<double>& y) {
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x_squared = 0;
    int n = x.size();

    #pragma omp parallel for reduction(+:sum_x,sum_y,sum_xy,sum_x_squared)
    for (int i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x_squared += x[i] * x[i];
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;

    std::cout << "Slope: " << slope << ", Intercept: " << intercept << std::endl;
}

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};

    linear_regression(x, y);

    return 0;
}
