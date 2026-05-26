cmake_minimum_required(VERSION 3.15...4.1)

project(
  NeuralBuilder
  VERSION 1.0
  LANGUAGES CXX)

set(IncludeDIR include)

# get files
file(GLOB_RECURSE ROOT_FILES ./src/*.cpp ./src/*.c ./src/*.h ./src/*.hpp)

# spdlog
add_subdirectory(${IncludeDIR}/spdlog)
include_directories(${spdlog_INCLUDE_DIRS})

# eigen
add_subdirectory(${IncludeDIR}/eigen)
include_directories(${eigen_INCLUDE_DIRS})

# json
add_subdirectory(${IncludeDIR}/json)
include_directories(${IncludeDIR}/json)

add_executable(neuralbuilder ${ROOT_FILES})

target_link_libraries(neuralbuilder PRIVATE spdlog eigen nlohmann_json)