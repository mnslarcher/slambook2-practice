#include <cmath>
#include <iostream>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

int main(int argc, char **argv) {
  Matrix3d rotation_matrix = Matrix3d::Identity();
  // Counterclockwise rotation of 45° around the z-axis
  AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));  // d = double
  cout.precision(3);
  cout << "Rotation matrix:\n" << rotation_vector.matrix() << endl;
  rotation_matrix = rotation_vector.toRotationMatrix();

  Vector3d v(1, 0, 0);
  Vector3d v_rotated = rotation_vector * v;
  cout << "\n(1, 0, 0) after a rotation (by angle axis) = "
       << v_rotated.transpose() << endl;

  v_rotated = rotation_matrix * v;
  cout << "\n(1, 0, 0) after rotation (by matrix) = " << v_rotated.transpose()
       << endl;

  Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
  // ZYX order, ie yaw, pitch, roll order
  cout << "\nYaw, pitch, roll = " << euler_angles.transpose() << endl;

  Isometry3d T = Isometry3d::Identity();
  T.rotate(rotation_vector);
  T.pretranslate(Vector3d(1, 3, 4));
  cout << "\nTransform matrix:\n" << T.matrix() << endl;

  Vector3d v_transformed = T * v;
  cout << "\nv transformed =  " << v_transformed.transpose() << endl;

  Quaterniond q = Quaterniond(rotation_vector);
  cout << "\nQuaternion from rotation vector = " << q.coeffs().transpose()
       << endl;

  q = Quaterniond(rotation_matrix);
  cout << "\nQuaternion form rotation matrix = " << q.coeffs().transpose()
       << endl;

  v_rotated = q * v;
  cout << "\n(1, 0, 0) after rotation = " << v_rotated.transpose() << endl;
  cout << "Should be equal to "
       << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose()
       << endl;

  return 0;
}
