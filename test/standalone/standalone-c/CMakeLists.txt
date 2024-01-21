add_mlir_aggregate(
  standalone-c
  SHARED
  empty.cpp
  EMBED_LIBS
  StandaloneCAPI
)

install(TARGETS standalone-c DESTINATION lib)

# // if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
# //     target_link_options(standalone-c PRIVATE "-Wl,-exclude-libs,ALL")
# // else()
# //   if(NOT CMAKE_C_VISIBILITY_PRESET STREQUAL "hidden" OR NOT CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden")
# //       message(STATUS "brutus-c on this platform exports all symbols. Recommend building with CMAKE_(C|CXX)_VISIBILITY_PRESET=hidden or implement filtering support.")
# //   endif()
# // endif()
