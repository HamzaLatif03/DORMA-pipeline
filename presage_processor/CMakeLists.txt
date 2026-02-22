cmake_minimum_required(VERSION 3.27.0)
project(PresageProcessor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SmartSpectra REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(presage_processor main.cc)

target_link_libraries(presage_processor
    SmartSpectra::Container
    ${OpenCV_LIBS}
)

install(TARGETS presage_processor DESTINATION bin)
