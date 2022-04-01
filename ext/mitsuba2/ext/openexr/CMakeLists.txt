cmake_minimum_required (VERSION 2.8)
project(openexr)

# Set a default build configuration (Release)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch -Wno-tautological-compare -Wno-deprecated-register")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result -Wno-deprecated")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4275 /wd4267 /wd4244 /wd4477 /wd4996 /wd4800 /wd4334 /wd4251 /wd4305 /wd4302 /wd4311 /wd4018")
endif()

add_subdirectory(IlmBase IlmBase)
include_directories(
  IlmBase/Iex
  IlmBase/Imath
  IlmBase/IlmThread
  IlmBase/Half
  ${CMAKE_CURRENT_BINARY_DIR}/IlmBase/config
)

set(CONFIGURATION "")
if (MSVC)
  set(CONFIGURATION "${OPENEXR_CONFIGURATION}")
endif()

add_custom_target(CopyIlmBaseLibs
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_BINARY_DIR}/IlmBase/Half/${OPENEXR_CONFIGURATION}
  ${CMAKE_CURRENT_BINARY_DIR}/OpenEXR/IlmImf/${OPENEXR_CONFIGURATION}
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_BINARY_DIR}/IlmBase/IlmThread/${OPENEXR_CONFIGURATION}
  ${CMAKE_CURRENT_BINARY_DIR}/OpenEXR/IlmImf/${OPENEXR_CONFIGURATION}
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_BINARY_DIR}/IlmBase/Iex/${OPENEXR_CONFIGURATION}
  ${CMAKE_CURRENT_BINARY_DIR}/OpenEXR/IlmImf/${OPENEXR_CONFIGURATION}
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_BINARY_DIR}/IlmBase/Imath/${OPENEXR_CONFIGURATION}
  ${CMAKE_CURRENT_BINARY_DIR}/OpenEXR/IlmImf/${OPENEXR_CONFIGURATION}
)
add_dependencies(CopyIlmBaseLibs Iex Imath IlmThread Half)

add_subdirectory(OpenEXR OpenEXR)
add_dependencies(b44ExpLogTable CopyIlmBaseLibs)
add_dependencies(dwaLookups CopyIlmBaseLibs)
