# ML models framework with Vortex compatibility

Framework for execution on Vortex ML models. CNN layers is supported now.

Directory structure
-------------------
- MainWork: framework implementation
- PythonNeuro: ml model for testing
- vortex_tests: vortex build and tests

Basic Installation Requirements
-------------------------------
Vortex requirements: https://github.com/vortexgpgpu/vortex

Install hdf5 for c++:

$ Ubuntu: sudo apt-get install libhdf5-serial-dev

$ MacOS: brew install hdf5
 
Then change h5_part in the MainWork/Makefile with your own paths ($ h5c++ -showconfig) for hdf5 library.
