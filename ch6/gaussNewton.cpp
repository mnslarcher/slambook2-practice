#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;   // ground-truth values
  double ae = 2.0, be = -1.0, ce = 5.0;  // initial estimation
  int N = 100;                           // num of data points
  double w_sigma = 1.0;                  // sigma of the noise
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;  // random number generator

  vector<double> x_data, y_data;
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) +
                     rng.gaussian(w_sigma * w_sigma));
  }

  // Start Gauss-Newton interations
  int iterations = 100;
  double cost = 0, lastCost = 0;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (int iter = 0; iter < iterations; iter++) {
    Matrix3d H = Matrix3d::Zero();  // Hessian = J^T W^{-1} J in Gauss-Newton
    Vector3d b = Vector3d::Zero();  // bias
    cost = 0;
    for (int i = 0; i < N; i++) {
      double xi = x_data[i], yi = y_data[i];  // the i-th data
      double error = yi - exp(ae * xi * xi + be * xi + ce);
      Vector3d J;                                          // jacobian
      J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
      J[1] = -xi * exp(ae * xi * xi + be * xi + ce);       // de/db
      J[2] = -exp(ae * xi * xi + be * xi + ce);            // de/dc

      H += inv_sigma * inv_sigma * J * J.transpose();
      b += -inv_sigma * inv_sigma * error * J;

      cost += error * error;
    }
    // solve Hx=b
    Vector3d dx = H.ldlt().solve(b);
    if (isnan(dx[0])) {
      cout << "Result is nan!" << endl;
      break;
    }

    ae += dx[0];
    be += dx[1];
    ce += dx[2];

    lastCost = cost;
    cout << "Total cost: " << cost << ", \t\tupdate: " << dx.transpose()
         << "\t\testimated params: " << ae << "," << be << "," << ce << endl;
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used =
      chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "Solve time cost = " << time_used.count() << " seconds. " << endl;
  cout << "Estimated abc = " << ae << ", " << be << ", " << ce << endl;

  return 0;
}