find_package(GTest REQUIRED)
# Thread support is needed for gtest
find_package(Threads REQUIRED)

function(setup_unit_test 
            kernel_name
            flashinfer_option_flag 
            list_of_test_sources 
            extra_link_libs
            compile_options
)
    if(flashinfer_option_flag)
        foreach(test_src IN list_of_test_sources)
            message(INFO "Building ${test_src} unit test")
            add_executable(${test_src} EXCLUDE_FROM_ALL ${test_src})
            target_include_directories(${test_src} 
                PRIVATE 
                    ${FLASHINFER_INCLUDE_DIR}
                    ${gtest_SOURCE_DIR}/include 
                    ${gtest_SOURCE_DIR}
            )
            add_dependencies(${test_src} dispatch_inc)
            target_link_libraries(${test_src} 
                PRIVATE 
                    gtest 
                    gtest_main
            )
            foreach(extra_link_lib IN extra_link_libs)
                target_link_libraries(${test_src} PRIVATE ${extra_link_lib})
            endforeach(extra_link_lib)
            
            target_compile_options(${test_src} 
                PRIVATE 
                    ${compile_options}
            )
            gtest_discover_tests(${test_src})
            message(INFO "Built ${test_src} unit test")
        endforeach(test_src)
    elseif(flashinfer_option_flag)
        message(INFO "Not building ${kernel_name} unit tests as ${flashinfer_option_flag} option is OFF")
    endif(flashinfer_option_flag)
endfunction(setup_unit_test)


function(setup_distributed_unit_tests)
    if(NOT FLASHINFER_DISTRIBUTED)
        message(INFO "Distributed unit tests not built as FLASHINFER_DISTRIBUTED is set to OFF")
        return()
    endif(NOT FLASHINFER_DISTRIBUTED)

    find_package(MPI REQUIRED)
    include(FetchContent)

    set(DISTRIBUTED_UNIT_TESTS_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/test_sum_all_reduce.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/test_attn_all_reduce.cu
    )
    
    FetchContent_Declare(
        mscclpp
        GIT_REPOSITORY https://github.com/microsoft/mscclpp.git
        GIT_TAG 11e62024d3eb190e005b4689f8c8443d91a6c82e
    )
    FetchContent_MakeAvailable(mscclpp)

    setup_unit_test(
        distributed
        FLASHINFER_DISTRIBUTED
        DISTRIBUTED_UNIT_TESTS_SRC 
        "MPI::MPI_CXX;mscclpp;"
        "-DENABLE_MPI"
    )
    
endfunction(setup_distributed_unit_tests)

# Emulate autotools like `make check` target to build tests
set(CMAKE_CTEST_COMMAND ctest --progress --output-on-failure -j 4)
add_custom_target(check COMMAND ${CMAKE_COMMAND} ${CMAKE_CTEST_COMMAND})
enable_testing()

set(DECODE_UNIT_TEST_SRCS 
        ${CMAKE_CURRENT_SOURCE_DIR}/test_single_decode.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/test_batch_decode.cu
)

set(PREFILL_UNIT_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_single_prefill.cu        
        ${CMAKE_CURRENT_SOURCE_DIR}/test_batch_prefill.cu
)

set(PAGE_UNIT_TEST_SRCS 
        ${CMAKE_CURRENT_SOURCE_DIR}/test_page.cu
)

set(CASCADE_UNIT_TEST_SRCS 
        ${CMAKE_CURRENT_SOURCE_DIR}/test_cascade.cu
)

set(SAMPLING_UNIT_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_sampling.cu
)

set(NORM_UNIT_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_norm.cu
)

set(FASTDIV_UNIT_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fastdiv.cu
)

set(FASTDEQUANT_UNIT_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fast_dequant.cu
)

set(TRGT_COMPILE_OPTS "-Wno-switch-bool")

setup_unit_test(decode FLASHINFER_DECODE DECODE_UNIT_TEST_SRCS "decode_kernels;" TRGT_COMPILE_OPTS)
setup_unit_test(prefill FLASHINFER_PREFILL PREFILL_UNIT_TEST_SRCS "prefill_kernels;" TRGT_COMPILE_OPTS)
setup_unit_test(page FLASHINFER_PAGE PAGE_UNIT_TEST_SRCS "" TRGT_COMPILE_OPTS)
setup_unit_test(cascade FLASHINFER_CASCADE CASCADE_UNIT_TEST_SRCS "decode_kernels;prefill_kernels" TRGT_COMPILE_OPTS)
setup_unit_test(sampling FLASHINFER_SAMPLING SAMPLING_UNIT_TEST_SRCS "" TRGT_COMPILE_OPTS)
setup_unit_test(norm FLASHINFER_NORM NORM_UNIT_TEST_SRCS "" TRGT_COMPILE_OPTS)
setup_unit_test(fastdiv FLASHINFER_FASTDIV_TEST FASTDIV_UNIT_TEST_SRCS "" TRGT_COMPILE_OPTS)
setup_unit_test(fast_dequant FLASHINFER_FASTDEQUANT_TEST FASTDEQUANT_UNIT_TEST_SRCS "" TRGT_COMPILE_OPTS)

setup_distributed_unit_tests()