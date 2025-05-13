cmake_minimum_required(VERSION 3.10)
project(MyCppProject)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
file(GLOB SOURCES "src/*.cpp")

include_directories(${OpenCV_INCLUDE_DIRS} include)
add_executable(main ${SOURCES})
target_link_libraries(main ${OpenCV_LIBS})
