cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CageGenerator CXX)

find_package(CGAL REQUIRED COMPONENTS Core Qt5)
find_package(Eigen3 REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CGAL_INCLUDE_DIRS})

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${VCG_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PUBLIC CGAL::CGAL )