# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name
project(HermiteSplinePathLength)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Use FetchContent to bring in dependencies
include(FetchContent)
FetchContent_Declare(
  tictoctimer
  GIT_REPOSITORY "https://github.com/civcode/tic-toc-timer"
  GIT_TAG "v0.2.2"
)
FetchContent_MakeAvailable(tictoctimer)

# Find and link the CppAD library
find_package(PkgConfig REQUIRED)

# Append to PKG_CONFIG_PATH to find the CppAD package
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/home/chris/.local/lib/pkgconfig")

pkg_check_modules(CPPAD REQUIRED cppad)

include_directories(${CPPAD_INCLUDE_DIRS})
link_directories(${CPPAD_LIBRARY_DIRS})

# Add the executable
add_executable(HermiteSplinePathLength src/main.cpp)

# Link the CppAD library to your project
target_link_libraries(HermiteSplinePathLength ${CPPAD_LIBRARIES} tictoctimer)

# Optionally, you can enable compiler warnings for better development practices
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # target_compile_options(HermiteSplinePathLength PRIVATE -Wall -Wextra -Wpedantic)
endif()