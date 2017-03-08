cmake_minimum_required(VERSION 2.8.12)

project(pcasvmPoly)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
add_subdirectory (pcasvm)
include_directories(pcasvm)

add_definitions(-std=c++11)

add_executable(pcasvmPoly pcasvm pcasvmPoly.cpp)
# Indicate which libraries to include during the link process.
target_link_libraries(pcasvmPoly pcasvm ${OpenCV_LIBS})

if(VTK_LIBRARIES)
  target_link_libraries(pcasvmPoly pcasvm ${VTK_LIBRARIES})
  target_link_libraries(pcasvmPoly pcasvm ${OpenCV_LIBS})
else()
  target_link_libraries(pcasvmPoly pcasvm vtkHybrid vtkWidgets)
  target_link_libraries(pcasvmPoly pcasvm ${OpenCV_LIBS})
endif()