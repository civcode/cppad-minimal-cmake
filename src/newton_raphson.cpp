#include <boost/math/tools/roots.hpp>
#include <iostream>
#include <cmath>
#include <utility>  // for std::pair

// Templated class for the function and its derivative
template <typename T>
class FunctionWithDerivative {
public:
    // Overload the function call operator to return a pair (f(x), f'(x))
    std::pair<T, T> operator()(T x) const {
        T f_value = x * x - 4;      // The function f(x) = x^2 - 4
        T derivative = 2 * x;       // The derivative f'(x) = 2x
        return std::make_pair(f_value, derivative);
    }
};

int main() {
    // Initial guess
    double initial_guess = 3.0;

    // Tolerance and maximum iterations
    double tol = std::numeric_limits<double>::epsilon();  // Precision tolerance
    const int digits = std::numeric_limits<double>::digits;
    int get_digits = static_cast<int>(digits * 0.6); 
    std::cout << "digits = " << digits << std::endl;
    //int max_iter = 50;
    const std::uintmax_t maxit = 20;
    std::uintmax_t max_iter = maxit;

    // Create an instance of the templated function
    FunctionWithDerivative<double> func;

    // Apply Newton-Raphson method using the class
    double root = boost::math::tools::newton_raphson_iterate(
        func,             // The function object
        initial_guess,    // Initial guess
        1.0,              // Lower bound for x
        5.0,              // Upper bound for x
        get_digits,              // Tolerance (machine epsilon)
        max_iter          // Maximum number of iterations
    );

    // Output the root and the number of iterations
    std::cout << "root = " << root << std::endl;
    std::cout << "Iterations used: " << max_iter << std::endl;

    return 0;
}