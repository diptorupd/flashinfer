include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 6910c9d9165801d8827d628cb72eb7ea9dd538c5 # release-1.16.0
  FIND_PACKAGE_ARGS NAMES GTest)
FetchContent_MakeAvailable(googletest)

# GoogleTest has to be included so that gtest_discover_tests works
include(GoogleTest)
# Thread support is needed for gtest
find_package(Threads REQUIRED)

set(ALL_TEST_TARGETS "")

function(setup_unit_test)
  set(options OPTIONAL IS_GTEST)
  set(oneValueArgs KERNEL_NAME ALL_TEST_TARGETS_LIST)
  set(multiValueArgs TEST_SOURCES EXTRA_LINK_LIBS EXTRA_COMPILE_FLAGS)

  cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}")
  foreach(test_src IN LISTS arg_TEST_SOURCES)
    message(STATUS "Configure ${test_src} unit test")

    get_filename_component(TEST_EXEC_NAME "${test_src}" NAME_WLE)
    set(TEST_TARGET_NAME ${TEST_EXEC_NAME})
    add_executable(${TEST_TARGET_NAME} EXCLUDE_FROM_ALL ${test_src})
    target_include_directories(${TEST_TARGET_NAME}
                               PRIVATE ${FLASHINFER_INCLUDE_DIR})
    target_include_directories(${TEST_TARGET_NAME}
                               PRIVATE ${FLASHINFER_UTILS_INCLUDE_DIR})
    target_include_directories(${TEST_TARGET_NAME}
                               PRIVATE ${GENERATED_SOURCE_DIR_ROOT})
    add_dependencies(${TEST_TARGET_NAME} dispatch_inc)
    foreach(extra_link_lib IN LISTS arg_EXTRA_LINK_LIBS)
      target_link_libraries(${TEST_TARGET_NAME} PRIVATE ${extra_link_lib})
    endforeach(extra_link_lib)

    foreach(copt IN LISTS arg_EXTRA_COMPILE_FLAGS)
      target_compile_options(${TEST_TARGET_NAME} PRIVATE ${copt})
    endforeach(copt)

    if(${arg_IS_GTEST})
      target_include_directories(
        ${TEST_TARGET_NAME} PRIVATE ${gtest_SOURCE_DIR}/include
                                    ${gtest_SOURCE_DIR})
      target_link_libraries(${TEST_TARGET_NAME} PRIVATE gtest gtest_main)
    elseif(${arg_IS_GTEST})
      add_test(NAME ${TEST_TARGET_NAME} COMMAND ${TEST_TARGET_NAME})
    endif(${arg_IS_GTEST})

    # Add the target to the ALL_TEST_TARGETS_LIST list so that it gets built
    # when the unittests target is invoked.
    list(APPEND ${arg_ALL_TEST_TARGETS_LIST} ${TEST_TARGET_NAME})
    set(${arg_ALL_TEST_TARGETS_LIST}
        ${${arg_ALL_TEST_TARGETS_LIST}}
        PARENT_SCOPE)
    message(STATUS "Configured ${test_src} unit test")

  endforeach(test_src)

endfunction(setup_unit_test)

function(setup_distributed_unit_tests test_targets_list)
  find_package(MPI)

  if(NOT MPI_FOUND)
    message(
      WARNING "Cannot configure distributed unit tests as MPI package not found"
    )
    return()
  endif(NOT MPI_FOUND)

  set(DISTRIBUTED_UNIT_TESTS_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/test_sum_all_reduce.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/test_attn_all_reduce.cu)

  FetchContent_Declare(
    mscclpp
    GIT_REPOSITORY https://github.com/microsoft/mscclpp.git
    GIT_TAG 11e62024d3eb190e005b4689f8c8443d91a6c82e)
  FetchContent_MakeAvailable(mscclpp)

  # cmake-format: off
  setup_unit_test(
    KERNEL_NAME distributed
    TEST_SOURCES ${DISTRIBUTED_UNIT_TESTS_SRC}
    EXTRA_LINK_LIBS "MPI::MPI_CXX;mscclpp;"
    EXTRA_COMPILE_FLAGS "-DENABLE_MPI"
    ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
    IS_GTEST OFF
  )
  # cmake-format: on
endfunction(setup_distributed_unit_tests)

set(DECODE_UNIT_TEST_SRCS "test_single_decode.cu" "test_batch_decode.cu")
set(PREFILL_UNIT_TEST_SRCS "test_single_prefill.cu" "test_batch_prefill.cu")
set(PAGE_UNIT_TEST_SRCS "test_page.cu")
set(CASCADE_UNIT_TEST_SRCS "test_cascade.cu")
set(SAMPLING_UNIT_TEST_SRCS "test_sampling.cu")
set(NORM_UNIT_TEST_SRCS "test_norm.cu")
set(FASTDIV_UNIT_TEST_SRCS "test_fastdiv.cu")
set(FASTDEQUANT_UNIT_TEST_SRCS "test_fast_dequant.cu")

set(options OPTIONAL IS_GTEST)
set(oneValueArgs KERNEL_NAME)
set(multiValueArgs TEST_SOURCES EXTRA_LINK_LIBS EXTRA_COMPILE_FLAGS)

# cmake-format: off
setup_unit_test(
  KERNEL_NAME decode
  TEST_SOURCES ${DECODE_UNIT_TEST_SRCS}
  EXTRA_LINK_LIBS "decode_kernels"
  ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
  IS_GTEST ON
)
setup_unit_test(
  KERNEL_NAME prefill
  TEST_SOURCES ${PREFILL_UNIT_TEST_SRCS}
  EXTRA_LINK_LIBS "prefill_kernels"
  ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
  IS_GTEST ON
)
setup_unit_test(
  KERNEL_NAME page
  TEST_SOURCES ${PAGE_UNIT_TEST_SRCS}
  ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
  IS_GTEST ON
)
setup_unit_test(
  KERNEL_NAME cascade
  TEST_SOURCES ${CASCADE_UNIT_TEST_SRCS}
  EXTRA_LINK_LIBS "decode_kernels;prefill_kernels"
  ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
  IS_GTEST ON
)
setup_unit_test(
  KERNEL_NAME sampling
  TEST_SOURCES ${SAMPLING_UNIT_TEST_SRCS}
  ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
  IS_GTEST ON
)
setup_unit_test(
  KERNEL_NAME norm
  TEST_SOURCES ${NORM_UNIT_TEST_SRCS}
  ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
  IS_GTEST ON
)
setup_unit_test(
  KERNEL_NAME fastdiv
  TEST_SOURCES ${FASTDIV_UNIT_TEST_SRCS}
  ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
  IS_GTEST ON
)
setup_unit_test(
  KERNEL_NAME fast_dequant
  TEST_SOURCES ${FASTDEQUANT_UNIT_TEST_SRCS}
  ALL_TEST_TARGETS_LIST ALL_TEST_TARGETS
  IS_GTEST ON
)
# cmake-format: on

if(FLASHINFER_DISTRIBUTED_UNITTESTS)
  setup_distributed_unit_tests(ALL_TEST_TARGETS)
endif(FLASHINFER_DISTRIBUTED_UNITTESTS)

# A top-level target to build all unit tests
add_custom_target(unittests)

message(STATUS "All unit test targets: ${ALL_TEST_TARGETS}")

# gtest_discover_tests adds all discovered tests as ctest tests. FIXME: for some
# reason google_discover_tests does not add the ctests
foreach(test_trgt IN LISTS ${ALL_TEST_TARGETS})
  gtest_discover_tests(${test_trgt})
endforeach(test_trgt)

add_dependencies(unittests ${ALL_TEST_TARGETS})
# Emulate autotools like `make check` target to run all ctests
set(CMAKE_CTEST_COMMAND ctest --progress --output-on-failure)
add_custom_target(check COMMAND ${CMAKE_COMMAND} ${CMAKE_CTEST_COMMAND})
add_dependencies(check unittests)

enable_testing()
