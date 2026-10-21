# ROS2 Example Package

[![Build](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/build.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/build.yaml) [![Test](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/test.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/test.yaml) [![Lint](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/lint.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/ros2_example_package/actions/workflows/lint.yaml)

This repository contains an example package for ros2 (python).

## Usage

This repository can be used as template. Simply select this repo when creating a new repository under template.

After creating a repository from the template, rename `ros2_example_package` to your package name (folder, `resource/` file, `package.xml`, `setup.py`, `setup.cfg`) and set `package_name` in `.github/workflows/build.yaml` and `.github/workflows/test.yaml` to it as well. Otherwise the CI builds and tests the wrong package. Also replace `ros2_example_package` in the badge links at the top of this README.

Alternatively, you can create python ros packages with:

```bash
# If not already created
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Create package
ros2 pkg create my_package --build-type ament_python --dependencies rclpy

# Build
cd ~/ros2_ws/src
colcon build --symlink-install --packages-select my_package
```

## Structure

- `.github/workflows/`: CI pipelines, which call the shared workflows of the [smarty_workspace](https://github.com/DHBW-Smart-Rollerz/smarty_workspace/tree/jazzy/.github/workflows):
  - `build.yaml`: Builds the package
  - `test.yaml`: Runs the tests of the package
  - `lint.yaml`: Checks the code with ruff (Python) and ament_uncrustify (C++)
- `config/`: All configurations (most of the time yaml files)
- `launch/`: Contains all launch files. Launch files can start multiple nodes with yaml-configurations
- `models/`: Contains all models (optional) and only necessary for machine learning nodes
- `resource/`: Contains the package name (required to build with colcon)
- `ros2_example_package`: Contains all nodes and sources for the ros package
- `test/`: Contains all tests
- `package.xml`: Contains metadata about the package
- `setup.py`: Used for Python package configuration
- `setup.cfg`: Additional configuration for the package
- `requirements.txt`: Python dependencies

## Contributing

Thank you for considering contributing to this repository! Here are a few guidelines to get you started:

1. Fork the repository and clone it locally.
2. Create a new branch for your contribution.
3. Make your changes and ensure they are properly tested.
4. Commit your changes and push them to your forked repository.
5. Submit a pull request with a clear description of your changes.

We appreciate your contributions and look forward to reviewing them!

## License

This repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.
