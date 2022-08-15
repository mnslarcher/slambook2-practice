#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  // Pose of robot x is qx (rotation), tx (transalation) respect to world
  // coordinate system W
  // Txw is the world to robot x transformation matrix
  Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
  // IMPORTANT: to use them to rotate a vector, they must be normalized
  q1.normalize();
  q2.normalize();

  Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);

  // p1 is a point in the robot 1 coordinate system
  Vector3d p1(0.5, 0, 0.2);

  Isometry3d T1w(q1), T2w(q2);
  T1w.pretranslate(t1);
  T2w.pretranslate(t2);

  // p2 is p1 in the robot 2 coordinate system
  Vector3d p2 = T2w * T1w.inverse() * p1;
  cout << endl << p2.transpose() << endl;

  return 0;
}