project(cvv_text_det) 
cmake_minimum_required(VERSION 2.6)
 
 

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -lpng -ljpeg -ltesseract -lavcodec -lavformat -lswscale")
#add_definitions(${CVV_COMPILE_FLAGS})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(TESSERACT REQUIRED)
if(TESSERACT_FOUND)
  include_directories(${TESSERACT_INCLUDE_DIRS})
endif()


find_package(VISP REQUIRED) 
if(VISP_FOUND) 
 include(${VISP_USE_FILE}) 
endif()

include_directories("${CMAKE_SOURCE_DIR}/lib")

#Read picture, convert to gray and save on the hd
add_executable(read_convert read_convert.cpp ${CMAKE_SOURCE_DIR}/lib/ccv.h) 
target_link_libraries(read_convert ${TESSERACT_LIBRARIES} ${CMAKE_SOURCE_DIR}/lib/libccv.a)

#Load picture from Opencv
add_executable(read_opencv read_opencv.cpp ${CMAKE_SOURCE_DIR}/lib/ccv.h) 
target_link_libraries(read_opencv ${TESSERACT_LIBRARIES} ${CMAKE_SOURCE_DIR}/lib/libccv.a)

#Detect text
add_executable(detect_text detect_text.cpp ${CMAKE_SOURCE_DIR}/lib/ccv.h)
target_link_libraries(detect_text ${TESSERACT_LIBRARIES} ${CMAKE_SOURCE_DIR}/lib/libccv.a)

#Read text
add_executable( read_text read_text.cpp ${CMAKE_SOURCE_DIR}/lib/ccv.h)
target_link_libraries(read_text ${TESSERACT_LIBRARIES} ${CMAKE_SOURCE_DIR}/lib/libccv.a)


#Detect and read text
add_executable(detect_read_text detect_read_text.cpp ${CMAKE_SOURCE_DIR}/lib/ccv.h)
target_link_libraries(detect_read_text ${TESSERACT_LIBRARIES} ${CMAKE_SOURCE_DIR}/lib/libccv.a)

#Detect and read text new version changin parameters
add_executable(detect_read_text_param detect_read_text_param.cpp ${CMAKE_SOURCE_DIR}/lib/ccv.h)
target_link_libraries(detect_read_text_param ${TESSERACT_LIBRARIES} ${CMAKE_SOURCE_DIR}/lib/libccv.a)
