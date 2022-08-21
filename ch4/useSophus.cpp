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
  cout << "\nSO3 from quaternion:\n" << SO3_q.matrix() << endl;
  cout << "\nThey are equal." << endl;

  // use logarithmic map to get the Lie algebra
  Vector3d so3 = SO3_R.log();
  cout << "\nso3 = " << so3.transpose() << endl;
  // hat is from vector to skew-symmetric matrix
  cout << "\nso3 hat:\n" << Sophus::SO3d::hat(so3) << endl;
  // inversely from matrix to vector
  cout << "\nso3 hat vee = "
       << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

  // update by perturbation model
  Vector3d update_so3(1e-4, 0, 0);  // this is a small update
  Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
  cout << "\nSO3 updated:\n" << SO3_updated.matrix() << endl;

  // Simmilar for SE(3)
  Vector3d t(1, 0, 0);        // translation 1 along x
  Sophus::SE3d SE3_Rt(R, t);  // construction SE3 from R, t
  Sophus::SE3d SE3_qt(q, t);  // or q, t
  cout << "\nSE3 from R, t:\n" << SE3_Rt.matrix() << endl;
  cout << "\nSE3 from q, t:\n" << SE3_qt.matrix() << endl;
  // Lie Algebra 6d vector, we give a typedef
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3_Rt.log();
  cout << "\nse3 = " << se3.transpose() << endl;
  // The output shows Sophus puts the translation at first in se(3), then
  // rotation. Save as SO(3) we have hat and vee
  cout << "\nse3 hat:\n" << Sophus::SE3d::hat(se3) << endl;
  cout << "\nse3 hat vee = "
       << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;
  // Finally the update
  Vector6d update_se3;
  update_se3.setZero();
  update_se3(0, 0) = 1e-4;
  Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
  cout << "\nSE3 updated:\n" << SE3_updated.matrix() << endl;
  return 0;
}