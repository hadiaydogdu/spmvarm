cmake_minimum_required(VERSION 2.8.8)
project(thundercat)

message(STATUS "System is " ${CMAKE_SYSTEM_NAME})
message(STATUS "Compiler is " ${CMAKE_CXX_COMPILER_ID})
message(STATUS ${llvm_libs})
message(${CMAKE_SOURCE_DIR})
set(LLVM_DIR "/home/hadi/llvm/build_350Ninja/share/llvm/cmake/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../modules/")
message(${CMAKE_MODULE_PATH})
find_package(LLVM 3.5.0 REQUIRED CONFIG)
find_package(MKL)


include_directories(${LLVM_INCLUDE_DIRS}
                    ${LLVM_BUILD_MAIN_SRC_DIR}
                    ${LLVM_BUILD_BINARY_DIR}/lib/Target/X86/
                    ${LLVM_BUILD_BINARY_DIR}/lib/Target/ARM/
                    ${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES})
add_definitions(${LLVM_DEFINITIONS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3 -fno-rtti -std=c++11" )

message(STATUS "CXX Flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "Linker Flags: " ${CMAKE_EXE_LINKER_FLAGS})
set(dir ${CMAKE_CURRENT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}")

set(SOURCE_FILES csrByNZ.cpp
                 csrByNZAnalyzer.cpp
                 genOski.cpp
                 genOskiAnalyzer.cpp
                 main.cpp
                 matrix.cpp
                 method.cpp
                 mkl.cpp
                 plaincsr.cpp
                 profiler.cpp
                 spMVgen.cpp
                 stencil.cpp
                 stencilAnalyzer.cpp
                 svmAnalyzer.cpp
                 unfolding.cpp
                 unfoldingAnalyzer.cpp
                 unrollingWithGOTO.cpp
                 unrollingWithGOTOAnalyzer.cpp)

add_executable(spMVgen ${SOURCE_FILES})

llvm_map_components_to_libnames(llvm_libs all)

if (MKL_FOUND)
    target_link_libraries(spMVgen ${llvm_libs}
                          mkl_intel_lp64
                          mkl_sequential
                          mkl_core)
else()
    target_link_libraries(spMVgen ${llvm_libs})
endif()
