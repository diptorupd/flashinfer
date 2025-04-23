# Function to detect available HIP architectures
function(detect_hip_architectures output_var)
  set(detected_archs "")

  if(HIP_COMPILER_ID STREQUAL "HIP")
    execute_process(
      COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --platform
      OUTPUT_VARIABLE hip_platform
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(hip_platform STREQUAL "amd")
      # Detect AMD GPUs
      execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --devices
        OUTPUT_VARIABLE hip_devices
        OUTPUT_STRIP_TRAILING_WHITESPACE)

      # Parse architecture values
      string(REGEX MATCHALL "gfx[0-9a-z]+" arch_matches "${hip_devices}")
      foreach(match ${arch_matches})
        list(APPEND detected_archs "${match}")
      endforeach()

      if(detected_archs)
        list(REMOVE_DUPLICATES detected_archs)
        message(STATUS "Detected HIP architectures: ${detected_archs}")
      else()
        message(STATUS "No HIP architectures detected automatically")
      endif()
    endif()
  endif()

  set(${output_var}
      "${detected_archs}"
      PARENT_SCOPE)
endfunction()

# Function to generate HIP architecture flags
function(generate_hip_arch_flags arch_list output_var)
  set(hip_arch_flags "")

  foreach(arch ${arch_list})
    list(APPEND hip_arch_flags "--offload-arch=${arch}")
  endforeach()

  set(${output_var}
      "${hip_arch_flags}"
      PARENT_SCOPE)
endfunction()
