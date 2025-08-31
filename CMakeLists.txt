cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(mnist_neural_net)

add_executable(run MACOSX_BUNDLE WIN32
        src/main.cpp
        src/trainer.h
)

target_include_directories(run PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/external)

add_subdirectory(external/fltk-1.4.3)
target_link_libraries(run PRIVATE fltk)


add_custom_command(TARGET run POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/dataset $<TARGET_FILE_DIR:run>/dataset
)
