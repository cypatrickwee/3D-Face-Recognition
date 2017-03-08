cmake_minimum_required(VERSION 2.8.12)

project(pcaknnMain)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
add_subdirectory (pcaknn)
include_directories(pcaknn)

add_definitions(-std=c++0x)

add_executable(pcaknnMain pcaknn pcaknnMain.cpp)
# Indicate which libraries to include during the link process.
target_link_libraries(pcaknnMain pcaknn ${OpenCV_LIBS})

if(VTK_LIBRARIES)
  target_link_libraries(pcaknnMain pcaknn ${VTK_LIBRARIES})
  target_link_libraries(pcaknnMain pcaknn ${OpenCV_LIBS})
else()
  target_link_libraries(pcaknnMainpcaknnMain pcaknn vtkHybrid vtkWidgets)
  target_link_libraries(pcaknnMain pcaknn ${OpenCV_LIBS})
endif()