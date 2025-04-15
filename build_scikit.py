#!/usr/bin/env python
import argparse
import os
import shutil
import site
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Build FlashInfer with scikit-build")
    parser.add_argument(
        "--dev", "-d", action="store_true", help="Install in development mode (-e)"
    )
    parser.add_argument(
        "--no-deps", "-N", action="store_true", help="Skip installing dependencies"
    )
    parser.add_argument(
        "--clean", "-c", action="store_true", help="Clean previous build artifacts"
    )
    parser.add_argument("--version-suffix", "-s", help="Add version suffix to package")
    parser.add_argument(
        "-D",
        "--define",
        action="append",
        dest="cmake_vars",
        default=[],
        help="Define a CMake variable: -DVAR=VALUE",
    )
    return parser.parse_args()


args = parse_args()

# Define build directory
BUILD_DIR = os.path.abspath("_skbuild")
WHEEL_DIR = os.path.join(BUILD_DIR, "wheels")
TEMP_DIR = os.path.join(BUILD_DIR, "temp")

# Find conda prefix for CMake
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    print(f"Using conda environment: {conda_prefix}")
else:
    print(
        "Warning: CONDA_PREFIX not found. CMAKE_PREFIX_PATH may not be set correctly."
    )

# Save original pyproject.toml
if os.path.exists("pyproject.toml.backup"):
    print("Using existing backup...")
else:
    print("Backing up original pyproject.toml...")
    shutil.copy("pyproject.toml", "pyproject.toml.backup")

try:
    # Replace with scikit-build version
    print("Installing with scikit-build configuration...")
    shutil.copy("pyproject.toml.scikit", "pyproject.toml")

    # Clean build directory if requested
    if args.clean and os.path.exists(BUILD_DIR):
        print(f"Cleaning build directory: {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)

    # Create necessary directories
    os.makedirs(BUILD_DIR, exist_ok=True)
    os.makedirs(WHEEL_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Only install basic build requirements
    print("Installing build dependencies...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "scikit-build-core>=0.4.3",
            "--no-build-isolation",
            "wheel",
            "ninja",
        ],
        check=True,
    )

    # Collect CMake variables
    config_settings = []

    # Ensure CMAKE_PREFIX_PATH includes conda prefix for finding dependencies
    if conda_prefix:
        config_settings.append(
            f"--config-settings=cmake.define.CMAKE_PREFIX_PATH={conda_prefix}"
        )
        print(f"Adding CMAKE_PREFIX_PATH={conda_prefix}")

    # Process user-defined CMake variables
    for cmake_def in args.cmake_vars:
        if "=" in cmake_def:
            key, value = cmake_def.split("=", 1)
            config_settings.append(f"--config-settings=cmake.define.{key}={value}")
            print(f"Adding cmake variable: {key}={value}")

    # Set version suffix if provided
    if args.version_suffix:
        config_settings.append(
            f"--config-settings=cmake.define.FLASHINFER_VERSION_SUFFIX={args.version_suffix}"
        )
        print(f"Setting version suffix: {args.version_suffix}")

    # Use development mode or build wheel
    if args.dev:
        print("Installing in development mode...")
        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "-v",
            f"--build={TEMP_DIR}",
            "--no-build-isolation",  # Always use system packages
        ]

        # Add --no-deps if requested
        if args.no_deps:
            install_cmd.append("--no-deps")

        # Add CMake config settings
        install_cmd.extend(config_settings)

        # Execute the install
        subprocess.run(install_cmd, check=True)
        print("Development installation completed.")
    else:
        print("Running pip wheel for build...")

        wheel_cmd = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-cache-dir",
            "-v",
            f"--wheel-dir={WHEEL_DIR}",
            f"--build={TEMP_DIR}",
            ".",
            "--no-build-isolation",  # Always use system packages
        ]

        # Add --no-deps if requested
        if args.no_deps:
            wheel_cmd.append("--no-deps")

        # Add CMake config settings
        wheel_cmd.extend(config_settings)

        # Execute the wheel build
        subprocess.run(wheel_cmd, check=True)

        # Now install from the generated wheel
        print("Installing built wheel...")
        wheel_files = [f for f in os.listdir(WHEEL_DIR) if f.endswith(".whl")]
        if wheel_files:
            print(f"Wheel created: {WHEEL_DIR}/{wheel_files[0]}")
            install_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                f"{WHEEL_DIR}/{wheel_files[0]}",
            ]

            # Add --no-deps if requested
            if args.no_deps:
                install_cmd.append("--no-deps")

            # Execute the install
            subprocess.run(install_cmd, check=True)
            print("Wheel installation completed.")
        else:
            print(f"No wheel found in {WHEEL_DIR} directory.")
            sys.exit(1)

    # Verify installation
    print("\nVerifying installation...")
    verify_cmd = [
        sys.executable,
        "-c",
        "import flashinfer; "
        "print(f'FlashInfer version: {flashinfer.__version__}'); "
        "print(f'Build metadata: {flashinfer.build_meta}')",
    ]

    try:
        subprocess.run(verify_cmd, check=True)
    except Exception as e:
        print(f"Verification failed: {e}")

except Exception as e:
    print(f"Error during build: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

finally:
    # Restore original pyproject.toml
    print("Restoring original pyproject.toml...")
    shutil.copy("pyproject.toml.backup", "pyproject.toml")
