# azplugins

azplugins is a component (plugin) for [HOOMD-blue](http://glotzerlab.engin.umich.edu/hoomd-blue)
which expands its functionality for tackling a variety of problems in soft matter physics.
Currently, azplugins is tested against v2.6.0 of HOOMD-blue. See [ChangeLog.md](ChangeLog.md) for
a list of recent development. If you are interested in adding new code, please refer to the
[guidelines](SourceConventions.md).

## Compiling azplugins

azplugins can be built using either of the standard [plugin build methods](http://hoomd-blue.readthedocs.io/en/stable/developer.html).
To build azplugins **internally** to HOOMD-blue, add a symlink to the code into the `hoomd-blue/hoomd`:

```bash
cd hoomd-blue/hoomd
ln -s /path/to/azplugins/azplugins azplugins
cd ../build && make install
```

azplugins is now available as a component of HOOMD-blue:

```python
import hoomd
from hoomd import azplugins
```

To build azplugins **externally** to HOOMD-blue, ensure that the `hoomd` module is on your Python path
(or hint to its location using `HOOMD_ROOT`), and install to an appropriate location:

```bash
cd /path/to/azplugins
mkdir build && cd build
cmake ..
make install
```

You must make sure that your installation location is on your `PYTHONPATH`, and then `azplugins` can
be imported as usual

```python
import hoomd
import azplugins
```

### Prerequisites

azplugins requires the same dependencies used to build HOOMD-blue. Typically, this means a modern
Python installation with numpy, a reasonably recent version of CMake, and a C++11 capable compiler.
To get good performance, you probably also want a recent CUDA toolkit and an MPI library.

### Testing

All code is unittested at the Python level. If azplugins has been built as an internal HOOMD-blue component,
it is automatically included into the testing tree. To run only the `azplugins` tests out of your build
directory, use ctest:

```bash
ctest -R azplugins-*
```

If azplugins has been built as an external Python package, all CTest options are available to you.
To run all tests out of your build directory,

```bash
make test
```

## History

azplugins began as a collection of code shared between students and postdocs at Princeton University (2016-2018).
It is named for their research advisor, Prof. Athanassios (Thanos) Z. Panagiotopoulos, whose group has made several
contributions to HOOMD-blue and frequently uses it in their work.
