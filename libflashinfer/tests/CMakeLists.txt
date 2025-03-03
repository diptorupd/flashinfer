find_package(GTest REQUIRED)
# Thread support is needed for gtest
find_package(Threads REQUIRED)

set(ALL_TEST_TARGETS)

function(setup_unit_test)
  set(options OPTIONAL IS_GTEST)
  set(oneValueArgs KERNEL_NAME ALL_TEST_TARGETS_LIST)
  set(multiValueArgs TEST_SOURCES EXTRA_LINK_LIBS EXTRA_COMPILE_FLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}")

  foreach(test_src IN TEST_SOURCES)
    message(INFO "Building ${test_src} unit test")
    add_executable(${test_src} EXCLUDE_FROM_ALL ${test_src})
    target_include_directories(${test_src} PRIVATE ${FLASHINFER_INCLUDE_DIR})
    add_dependencies(${test_src} dispatch_inc)
    foreach(extra_link_lib IN EXTRA_LINK_LIBS)
      target_link_libraries(${test_src} PRIVATE ${extra_link_lib})
    endforeach(extra_link_lib)

    foreach(copt IN EXTRA_COMPILE_FLAGS)
      target_compile_options(${test_src} PRIVATE ${copt})
    endforeach(copt)

    if(IS_GTEST)
      target_include_directories(${test_src} PRIVATE 
        ${gtest_SOURCE_DIR}/include
        ${gtest_SOURCE_DIR}
      )
      # gtest_discover_tests adds all discovered tests as ctest tests.
      gtest_discover_tests(${test_src} TEST_LIST test_list_var)
      foreach(ctest_case IN test_list_var)
        set_tests_properties(${ctest_case} PROPERTIES LABELS "${KERNEL_NAME}")
      endforeach(ctest_case)
      target_link_libraries(${test_src} PRIVATE gtest gtest_main)
    elseif(IS_GTEST)
      add_test(NAME ${test_src} COMMAND ${test_src})
      set_tests_properties(${test_src} PROPERTIES LABELS "${KERNEL_NAME}")
    endif(IS_GTEST)
  
    # Add the target to the ALL_TEST_TARGETS_LIST list so that it gets built 
    # when the unittests target is invoked.
    list(APPEND ${test_targets_list} ${test_src})
    set(${test_targets_list} ${${ALL_TEST_TARGETS_LIST}} PARENT_SCOPE)
    message(INFO "Built ${test_src} unit test")
  
  endforeach(test_src)

  # A convenience custom target to just run the unit tests in this group
  set(CMAKE_CTEST_COMMAND ctest --progress -V -L ${KERNEL_NAME} --output-on-failure -j 4)
  add_custom_target(check-${KERNEL_NAME} COMMAND ${CMAKE_COMMAND} ${CMAKE_CTEST_COMMAND})

endfunction(setup_unit_test)

# function(setup_unit_test kernel_name list_of_test_sources extra_link_libs
#          test_targets_list)
#   foreach(test_src IN list_of_test_sources)
#     message(INFO "Building ${test_src} unit test")
#     add_executable(${test_src} EXCLUDE_FROM_ALL ${test_src})
#     target_include_directories(
#       ${test_src} PRIVATE ${FLASHINFER_INCLUDE_DIR} ${gtest_SOURCE_DIR}/include
#                           ${gtest_SOURCE_DIR})
#     add_dependencies(${test_src} dispatch_inc)
#     foreach(extra_link_lib IN extra_link_libs)
#       target_link_libraries(${test_src} PRIVATE ${extra_link_lib})
#     endforeach(extra_link_lib)
#     # gtest_discover_tests adds all discovered tests as ctest tests.
#     gtest_discover_tests(${test_src} TEST_LIST test_list_var)
#     foreach(ctest_case IN test_list_var)
#       set_tests_properties(${ctest_case} PROPERTIES LABELS "${kernel_name}")
#     endforeach(ctest_case)

#     # Add the target to the ALL_TEST_TARGETS list so that it gets built when the
#     # unittests target is invoked.
#     list(APPEND ${test_targets_list} ${test_src})
#     set(${test_targets_list}
#         ${${test_targets_list}}
#         PARENT_SCOPE)

#     message(INFO "Built ${test_src} unit test")

#   endforeach(test_src)

#   # A convenience custom target to just run the unit tests in this group
#   set(CMAKE_CTEST_COMMAND ctest --progress -V -L ${kernel_name}
#                           --output-on-failure -j 4)
#   add_custom_target(check-${kernel_name} COMMAND ${CMAKE_COMMAND}
#                                                  ${CMAKE_CTEST_COMMAND})

# endfunction(setup_unit_test)

function(setup_distributed_unit_tests test_targets_list)
  find_package(MPI)
  include(FetchContent)

  if(NOT PackageMPI_FOUND)
    message(WARN
            "Cannot configure distributed unit tests as MPI package not found")
    return()
  endif(NOT PackageMPI_FOUND)

  set(DISTRIBUTED_UNIT_TESTS_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/test_sum_all_reduce.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/test_attn_all_reduce.cu)

  FetchContent_Declare(
    mscclpp
    GIT_REPOSITORY https://github.com/microsoft/mscclpp.git
    GIT_TAG 11e62024d3eb190e005b4689f8c8443d91a6c82e)
  FetchContent_MakeAvailable(mscclpp)

  setup_unit_test(distributed DISTRIBUTED_UNIT_TESTS_SRC
                  "MPI::MPI_CXX;mscclpp;" "-DENABLE_MPI")

  setup_unit_test(
    KERNEL_NAME distributed 
    TEST_SOURCES DISTRIBUTED_UNIT_TESTS_SRC 
    EXTRA_LINK_LIBS "MPI::MPI_CXX;mscclpp;"
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST ON
  )
  # foreach(test_src IN DISTRIBUTED_UNIT_TESTS_SRC)
  #   message(INFO "Building ${test_src} unit test")
  #   add_executable(${test_src} EXCLUDE_FROM_ALL ${test_src})
  #   target_include_directories(${test_src} PRIVATE ${FLASHINFER_INCLUDE_DIR})
  #   add_dependencies(${test_src} dispatch_inc)
  #   target_link_libraries(${test_src} PRIVATE "MPI::MPI_CXX;mscclpp;")
  #   target_compile_options(${test_src} PRIVATE -DENABLE_MPI)
  #   add_test(NAME ${test_src} COMMAND ${test_src})
  #   set_tests_properties(${test_src} PROPERTIES LABELS distributed)
  #   list(APPEND ${test_targets_list} ${test_src})
  #   set(${test_targets_list}
  #       ${${test_targets_list}}
  #       PARENT_SCOPE)
  #   message(INFO "Built ${test_src} unit test")
  # endforeach(test_src)

  # # A convenience custom target to just run the unit tests in this group
  # set(CMAKE_CTEST_COMMAND ctest --progress -V -L distributed
  #                         --output-on-failure -j 4)
  # add_custom_target(check-distributed COMMAND ${CMAKE_COMMAND}
  #                                             ${CMAKE_CTEST_COMMAND})

endfunction(setup_distributed_unit_tests)

set(DECODE_UNIT_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_single_decode.cu
                          ${CMAKE_CURRENT_SOURCE_DIR}/test_batch_decode.cu)
set(PREFILL_UNIT_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_single_prefill.cu
                           ${CMAKE_CURRENT_SOURCE_DIR}/test_batch_prefill.cu)
set(PAGE_UNIT_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_page.cu)
set(CASCADE_UNIT_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_cascade.cu)
set(SAMPLING_UNIT_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_sampling.cu)
set(NORM_UNIT_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_norm.cu)
set(FASTDIV_UNIT_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_fastdiv.cu)
set(FASTDEQUANT_UNIT_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_fast_dequant.cu)

set(options OPTIONAL IS_GTEST)
set(oneValueArgs KERNEL_NAME)
set(multiValueArgs TEST_SOURCES EXTRA_LINK_LIBS EXTRA_COMPILE_FLAGS)

setup_unit_test(
    KERNEL_NAME decode 
    TEST_SOURCES DECODE_UNIT_TEST_SRCS 
    EXTRA_LINK_LIBS "decode_kernels"
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST ON
)
setup_unit_test(
    KERNEL_NAME prefill 
    TEST_SOURCES PREFILL_UNIT_TEST_SRCS
    EXTRA_LINK_LIBS "prefill_kernels" 
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST ON
)
setup_unit_test(
    KERNEL_NAME page 
    TEST_SOURCES PAGE_UNIT_TEST_SRCS 
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST ON
)
setup_unit_test(
    KERNEL_NAME cascade 
    TEST_SOURCES CASCADE_UNIT_TEST_SRCS
    EXTRA_LINK_LIBS "decode_kernels;prefill_kernels"
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST ON
)
setup_unit_test(
    KERNEL_NAME sampling 
    TEST_SOURCES SAMPLING_UNIT_TEST_SRCS 
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST ON
)
setup_unit_test(
    KERNEL_NAME norm 
    TEST_SOURCES NORM_UNIT_TEST_SRCS 
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST ON
)
setup_unit_test(
    KERNEL_NAME fastdiv 
    TEST_SOURCES FASTDIV_UNIT_TEST_SRCS  
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST ON
)
setup_unit_test(
    KERNEL_NAME fast_dequant 
    TEST_SOURCES FASTDEQUANT_UNIT_TEST_SRCS     
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST ON
)

setup_distributed_unit_tests(ALL_TEST_TARGETS)

# A top-level target to build all unit tests
add_custom_target(unittests)
add_dependencies(unittests ${ALL_TEST_TARGETS})
# Emulate autotools like `make check` target to run all ctests
set(CMAKE_CTEST_COMMAND ctest --progress --output-on-failure -j 4)
add_custom_target(check-all COMMAND ${CMAKE_COMMAND} ${CMAKE_CTEST_COMMAND})
add_dependencies(check-all unittests)

enable_testing()
