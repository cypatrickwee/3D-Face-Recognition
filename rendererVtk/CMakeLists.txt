cmake_minimum_required(VERSION 2.8.12)

project(rendering)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(rendering rendering.cpp)

 
if(VTK_LIBRARIES)
  target_link_libraries(rendering ${VTK_LIBRARIES})
  target_link_libraries(rendering ${OpenCV_LIBS})
else()
  target_link_libraries(rendering vtkHybrid vtkWidgets)
  target_link_libraries(rendering ${OpenCV_LIBS})
endif()