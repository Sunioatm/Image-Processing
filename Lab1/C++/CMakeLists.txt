cmake_minimum_required(VERSION 3.0.0)
project(lab1 VERSION 0.1.0 LANGUAGES C CXX)

# Find OpenCV package
find_package(OpenCV REQUIRED)

include(CTest)
enable_testing()

add_executable(lab1 main.cpp)

# Include OpenCV directories
target_include_directories(lab1 PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries
target_link_libraries(lab1 PRIVATE ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
