#include <boost/math/tools/roots.hpp>
#include <iostream>
#include <cmath>
#include <utility>  // for std::pair

// Function f(x) = x^2 - 4 (we want to find the root of this function)
double f(double x) {
        return x * x - 4;
}

// Derivative of f(x), f'(x) = 2*x
double f_prime(double x) {
        return 2 * x;
}

int main() {
    // Initial guess
    double initial_guess = 4.0;

    // Tolerance and maximum iterations
    double tol = std::numeric_limits<double>::epsilon();  // Precision tolerance
    const int digits = std::numeric_limits<double>::digits;
    int get_digits = static_cast<int>(digits * 0.6); 
    std::cout << "digits = " << digits << std::endl;
    //int max_iter = 50;
    const std::uintmax_t maxit = 20;
    std::uintmax_t max_iter = maxit;

    // Create an instance of the templated function

    auto func = [](double x) {return std::make_pair(f(x), f_prime(x));};

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