# Convert C/C++ values to Python
function(_convert_to_python_value value output_var)
  # Handle boolean values
  if(value STREQUAL "ON"
     OR value STREQUAL "TRUE"
     OR value STREQUAL "1")
    set(${output_var}
        "True"
        PARENT_SCOPE)
    return()
  elseif(
    value STREQUAL "OFF"
    OR value STREQUAL "FALSE"
    OR value STREQUAL "0")
    set(${output_var}
        "False"
        PARENT_SCOPE)
    return()
  endif()

  # Handle numeric values
  if(value MATCHES "^[0-9]+$")
    set(${output_var}
        "${value}"
        PARENT_SCOPE)
    return()
  endif()

  # Handle lists (common in CMake)
  if(value MATCHES ";")
    # Convert CMake list to Python list format
    string(REPLACE ";" "\", \"" list_items "${value}")
    set(${output_var}
        "[\"${list_items}\"]"
        PARENT_SCOPE)
    return()
  endif()

  # Handle regular strings - escape quotes and wrap in quotes
  string(REPLACE "\"" "\\\"" escaped_value "${value}")
  string(REPLACE "\n" "\\n" escaped_value "${escaped_value}")
  set(${output_var}
      "\"${escaped_value}\""
      PARENT_SCOPE)
endfunction()

function(flashinfer_generate_config_python)
  # Parse function arguments
  set(options "")
  set(oneValueArgs SOURCE_DIR BINARY_DIR INSTALL_DIR TEMPLATE_FILE OUTPUT_FILE)
  set(multiValueArgs EXCLUDE_PATTERNS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # Set defaults
  if(NOT ARG_SOURCE_DIR)
    set(ARG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  if(NOT ARG_BINARY_DIR)
    set(ARG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  if(NOT ARG_INSTALL_DIR)
    set(ARG_INSTALL_DIR "flashinfer")
  endif()
  if(NOT ARG_TEMPLATE_FILE)
    set(ARG_TEMPLATE_FILE "${CMAKE_SOURCE_DIR}/templates/__config__.py.in")
  endif()
  if(NOT ARG_OUTPUT_FILE)
    set(ARG_OUTPUT_FILE "${ARG_BINARY_DIR}/flashinfer/__config__.py")
  endif()

  # Standard exclusion patterns
  set(EXCLUDE_PATTERNS
      ".*_DIR$"
      ".*_PATH$"
      ".*_FOUND$"
      ".*_FILE$"
      "FLASHINFER_CONFIG_DEFINES"
      "FLASHINFER_ALL_OPTIONS"
      ".*GENERATED.*DIR.*"
      ".*SOURCE.*ROOT.*")

  list(APPEND EXCLUDE_PATTERNS ${ARG_EXCLUDE_PATTERNS})

  # Ensure output directory exists
  get_filename_component(OUTPUT_DIR "${ARG_OUTPUT_FILE}" DIRECTORY)
  file(MAKE_DIRECTORY "${OUTPUT_DIR}")

  # Get and filter FLASHINFER variables
  get_cmake_property(_variableNames VARIABLES)
  list(FILTER _variableNames INCLUDE REGEX "^FLASHINFER_[A-Z0-9_]+$")

  # Apply exclude patterns
  foreach(_pattern ${EXCLUDE_PATTERNS})
    list(FILTER _variableNames EXCLUDE REGEX "${_pattern}")
  endforeach()

  # Clean up variable list
  list(SORT _variableNames)
  list(REMOVE_DUPLICATES _variableNames)

  # Generate info dictionary entries
  set(INFO_DICT_ENTRIES "")

  # System info block
  set(SYS_INFO_BLOCK "import platform\n")
  string(APPEND SYS_INFO_BLOCK "system = platform.system()\n")
  string(APPEND SYS_INFO_BLOCK "python_version = platform.python_version()\n")

  # Torch info block if AOT enabled
  set(TORCH_INFO_BLOCK "# No PyTorch information available")
  if(FLASHINFER_AOT_TORCH_EXTS_CUDA)
    set(TORCH_INFO_BLOCK "import torch\n")
    string(APPEND TORCH_INFO_BLOCK "torch_version = torch.__version__\n")
    string(APPEND TORCH_INFO_BLOCK "cuda_version = torch.version.cuda\n")

    # Add to info dict
    list(APPEND INFO_DICT_ENTRIES "'torch_version': torch_version,")
    list(APPEND INFO_DICT_ENTRIES "'cuda_version': cuda_version,")
  endif()

  # Process each variable
  foreach(_var ${_variableNames})
    if(DEFINED ${_var})
      set(_value "${${_var}}")
      _convert_to_python_value("${_value}" _py_value)

      # Convert C++ variable name to Python (FLASHINFER_X_Y -> x_y)
      string(REPLACE "FLASHINFER_" "" _py_var "${_var}")
      string(TOLOWER "${_py_var}" _py_var)

      message(STATUS "Orignal Var : ${_var} : ${_value}")
      message(STATUS "Pythonized : ${_py_var} : ${_py_value}")

      set(${_py_var} ${_py_value})

      # Add to info dict if not already declared at the top
      if(NOT _py_var STREQUAL "enable_f16"
         AND NOT _py_var STREQUAL "enable_bf16"
         AND NOT _py_var STREQUAL "enable_fp8_e4m3"
         AND NOT _py_var STREQUAL "enable_fp8_e5m2")
        list(APPEND INFO_DICT_ENTRIES "'${_py_var}': ${_py_value},")
      endif()
    endif()
  endforeach()

  # Join INFO_DICT_ENTRIES with newlines
  string(JOIN "\n    " INFO_DICT_ENTRIES ${INFO_DICT_ENTRIES})

  message(STATUS "INFO_DICT_ENTRIES : ${INFO_DICT_ENTRIES}")
  # Get git hash for revision
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT GIT_HASH)
    set(GIT_HASH "unknown")
  endif()

  # Configure the template
  configure_file("${ARG_TEMPLATE_FILE}" "${ARG_OUTPUT_FILE}" @ONLY)

  # Install the generated file
  install(FILES "${ARG_OUTPUT_FILE}" DESTINATION "${ARG_INSTALL_DIR}")
  message(STATUS "Generated Python config file: ${ARG_OUTPUT_FILE}")
endfunction()
