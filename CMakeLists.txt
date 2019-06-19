cmake_minimum_required (VERSION 2.6)

project (BGA2SVM)

# compiler options
set(CMAKE_CXX_FLAGS "-Wall")
if (MSVC)
	set(CMAKE_CXX_FLAGS "/EHsc /W3")
endif (MSVC)
	

add_executable(BGA2SVM src/main.cpp inc/gnuplot-iostream.h inc/ue3.h)

### includes ###

# add search path for include files
include_directories("${PROJECT_SOURCE_DIR}/inc")
include_directories("${PROJECT_BINARY_DIR}")

### libraries ###
# opencv
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect ml)
MESSAGE(STATUS "opencv include dirs: ${OpenCV_INCLUDE_DIRS}" )
MESSAGE(STATUS "opencv libs: ${OpenCV_LIBS}" )

# boost
set(Boost_USE_STATIC_LIBS ON)
# 
find_package(Boost COMPONENTS iostreams system exception filesystem REQUIRED)
MESSAGE(STATUS "boost include dirs: ${Boost_INCLUDE_DIR}" )
MESSAGE(STATUS "boost libs: ${Boost_LIBRARIES}" )

# link and include opencv
include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})

# link
TARGET_LINK_LIBRARIES(BGA2SVM ${OpenCV_LIBS} ${Boost_LIBRARIES})
