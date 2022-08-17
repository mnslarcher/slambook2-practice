#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
  Quaterniond q(R);
  Sophus::SO3d SO3_R(R);
  Sophus::SO3d SO3_q(q);

  cout << "SO3 from matrix:\n" << SO3_R.matrix() << endl;
  cout << "SO3 from quaternion:\n" << SO3_q.matrix() << endl;
  cout << "They are equal." << endl;
  return 0;
}