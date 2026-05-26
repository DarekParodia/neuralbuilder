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
include_directories(${IncludeDIR}/eigen)

# json
add_subdirectory(${IncludeDIR}/json)
include_directories(${IncludeDIR}/json/single_include)

# slint
add_subdirectory(${IncludeDIR}/slint)

add_executable(neuralbuilder ${ROOT_FILES})
slint_target_sources(neuralbuilder ui/app-window.slint)

target_link_libraries(neuralbuilder PRIVATE spdlog nlohmann_json Slint::Slint)