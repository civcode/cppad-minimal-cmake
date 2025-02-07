#include <cppad/cppad.hpp>
#include <vector>
#include <cmath>
#include <iostream>


using std::cout;
using std::endl;

// Define the parametric Hermite spline for both x and y
template <typename T>
std::vector<T> ParametricHermiteSpline(T t, const std::vector<T>& p0, const std::vector<T>& m0,
                                       const std::vector<T>& p1, const std::vector<T>& m1) {
    // Basis functions
    T h00 = (2 * t * t * t - 3 * t * t + 1); // basis for p0
    T h10 = (t * t * t - 2 * t * t + t);     // basis for m0
    T h01 = (-2 * t * t * t + 3 * t * t);    // basis for p1
    T h11 = (t * t * t - t * t);             // basis for m1

    std::vector<T> H(2); // 2D output (x, y)
    H[0] = h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0]; // x(t)
    H[1] = h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1]; // y(t)

    return H;
}

// Function to compute the path length of the parametric Hermite spline
double computeParametricPathLength(const std::vector<double>& p0, const std::vector<double>& m0,
                                   const std::vector<double>& p1, const std::vector<double>& m1) {
    // Number of sample points for numerical integration
    const int N = 10;
    double dt = 1.0 / N;
    double path_length = 0.0;

    // CppAD types for automatic differentiation
    typedef CppAD::AD<double> ADdouble;

    // Independent variable setup for CppAD (use a vector for independent variable)
    std::vector<ADdouble> T(1);
    T[0] = 0.0; // Initial t value

    // Convert control points and tangents to ADdouble types
    std::vector<ADdouble> p0_ad(p0.begin(), p0.end());
    std::vector<ADdouble> m0_ad(m0.begin(), m0.end());
    std::vector<ADdouble> p1_ad(p1.begin(), p1.end());
    std::vector<ADdouble> m1_ad(m1.begin(), m1.end());

    // Record the Hermite spline once outside the loop
    CppAD::Independent(T);

    // Compute the parametric Hermite spline position (x(t), y(t)) with AD types
    std::vector<ADdouble> H = ParametricHermiteSpline(T[0], p0_ad, m0_ad, p1_ad, m1_ad);

    // End recording the computation
    CppAD::ADFun<double> f;
    f.Dependent(T, H);

    // Inside the loop, only compute the derivative for different values of t
    for (int i = 0; i < N; ++i) {
        // Evaluate the derivative using CppAD by computing the Jacobian
        // First, create a vector with the current value of t
        std::vector<double> t_vec(1);
        t_vec[0] = i * dt; // Time value for the current step

        // Compute the Jacobian (which is effectively dx/dt and dy/dt)
        std::vector<double> H_dot = f.Jacobian(t_vec);
        cout << "Jacobian: [" << H_dot[0] << ", " << H_dot[1] << "]" << endl;

        // Compute the arc length increment: sqrt((dx/dt)^2 + (dy/dt)^2)
        double velocity_x = H_dot[0];
        double velocity_y = H_dot[1];
        double speed = std::sqrt(velocity_x * velocity_x + velocity_y * velocity_y);

        // Compute the Hessian for both x(t) and y(t)
        std::vector<std::vector<double>> H2(2);
        H2[0] = f.Hessian(t_vec, 0); // Second derivative for x(t)
        H2[1] = f.Hessian(t_vec, 1); // Second derivative for y(t)
        cout << "Hessian: [" << H2[0][0] << ", " << H2[1][0] << "]" << endl;
        //cout << H2.size() << endl;
        //cout << H2[0].size() << endl;
        //cout << H2[1].size() << endl;


        // Integrate path length
        path_length += speed * dt;
    }

    return path_length;
}

int main() {
    // Define vector-valued control points and tangents (for x and y)
    std::vector<double> p0 = {0.0, 0.0};  // Start point (x0, y0)
    std::vector<double> p1 = {1.0, 1.0};  // End point (x1, y1)
    std::vector<double> m0 = {1.0, 0.0};  // Tangent at p0
    std::vector<double> m1 = {0.0, 1.0};  // Tangent at p1

    // Compute the path length of the parametric Hermite spline
    double path_length = computeParametricPathLength(p0, m0, p1, m1);

    std::cout << "Path Length: " << path_length << std::endl;

    return 0;
}