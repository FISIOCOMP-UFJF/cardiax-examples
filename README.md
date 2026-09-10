# Cardiax Examples and Experiments

This repository contains a collection of examples, functionalities, and custom source codes designed to solve specific experiments for published academic papers using the Cardiax electromechanical simulator.

A key feature of this repository is its decoupled architecture. It allows users to import Cardiax classes and override standard logic with new custom implementations if necessary, without the need to modify the original Cardiax source code.

## Prerequisites

To execute the codes in this repository, you must have Cardiax installed on your computer using Conda. Please follow the installation guide provided in the README of the main Cardiax repository before proceeding.

## Repository Structure

The repository includes two main types of examples:

* **Shell Scripts (.sh):** Examples that utilize the standard, pre-compiled executables provided by the core Cardiax installation.
* **Custom C++ Applications (.cpp):** Advanced examples that import Cardiax data structures and internal components to program highly specific, customized problems.

## How to Run the Examples

### Running Shell Script Examples (.sh)

To run the shell script examples, you must add the directory containing the compiled Cardiax executables to your system's PATH variable. Once added, you can execute the scripts normally from your terminal.

```bash
export PATH="/path/to/your/Cardiax/build/app:$PATH"
./example_script.sh

```

### Running Custom C++ Problems (.cpp)

To compile and run the custom C++ problems, navigate to the specific example folder, create a build directory, and compile the code using CMake.

```bash
mkdir build
cd build
cmake ..
make
./your_custom_executable

```

## Creating Your Own Custom Problems

You can easily adapt this repository's framework to build your own isolated experiments. To create a custom problem:

1. Copy the provided template file `CMakeLists_example.txt` into your project folder and rename it to `CMakeLists.txt`.
2. Open the file and adapt the name of your executable, the name of your .cpp source file, and the absolute path pointing to your core Cardiax directory.
3. Repeat the build execution process (create a build folder, run cmake, and make).

## Recommended Organization for New Contributions

To maintain consistency and reproducibility across all academic experiments, we suggest that the organization of files within each example folder follows this standard pattern:

* **Source Code:** The script (.sh) or source code (.cpp) containing the experiment's logic.
* **Build System:** A properly configured `CMakeLists.txt` (if it is a C++ project).
* **Input Data:** Any mesh files, configuration files, or other required inputs.
* **Documentation (README.md):** A dedicated, local README file explaining the physiological or mathematical purpose of the experiment. It is highly recommended to clearly state the exact version of Cardiax used to execute the test to ensure reproducibility in case of future compatibility losses.
* **Visual Output:** A .png image demonstrating the expected simulation result or output data plot.