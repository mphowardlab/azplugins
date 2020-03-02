Contributions are encouraged using pull requests on GitHub for feature / fix branches.

# New feature guidelines

## Create a new branch based off master

Each new feature or fix should be implemented as a git branch off master.
Name the branch after the feature or fix being done. For example, a new pair
potential could be `feature/pair_cool`, while a bug fix branch could be
`bugfix/sky_is_orange`.

## Write sphinx documentation for user-facing API

All features must be documented with examples using sphinx.
This should include a mathematical form for potentials, etc.
Also, include a hyperlink to a sensible reference if you are
implementing work from a paper. Simple working examples should
be included the documentation. Include warning blocks if there
are any significant assumptions of simplifications.

## Write doxygen documentation for developers

All C++ code must be documented using doxygen. At minimum,
a simple description is required for all objects, methods, and
member variables. Additionally, document all parameters and return
values from functions.

## Write thorough Python-level unit tests

Python unittest should be used to test syntax for object creation
and parameter setting. It must also be used to validate calculations
on simple test problems. For example, for a pair potential, you should
test that all parameters can be set, and errors are raised when parameters
are missing. Then, test for a two particle system that the correct energy
and force are reported.

## Support MPI if possible

If it is easy enough to support MPI in your code, go ahead and do it
while you are developing. If not, make sure that there is appropriate
error handling.

## Code to an interface

Write the python implementation of your new feature first, then implement
the C++ header file. Finally, implement the details of the algorithm.

## Add your feature or fix to the credits

All new features or fixes should give credit to the author in `CREDITS.md`.

# Source code guidelines

## Use current copyright notice and identify the maintainer of the code

This makes it easier to update the copyright year over year.
```c++
// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: your-bitbucket-username
```

## Add code to lists alphabetically

Try to add classes alphabetically to long lists to make them easier to find.

## Return error messages through the messenger and by exception

In C++ and python code, error messages should be processed through the Messenger
class, and also by raising an exception.


# C++ guidelines

## Namespaces

Namespaces should be used for all code. All code must reside in the `azplugins`
namespace. Implementation details, such as python export methods, should reside
in `azplugins::detail`. Do not use "using namespace" in headers.

## Classes

Classes should be named as nouns of the objects they represent.
Class names should have the first letter of each word
capitalized, e.g. ForceCompute. Class methods should be
camelcase (e.g., getNetForce), and should be verbs describing their
action. Member variables should begin with `m_`, and should be all
lower case.
```c++
//! Place people go to work
class Office
    {
    public:
        //! Constructor
        /*!
         * \param room Room the office is located in
         */
        Office(std::shared_ptr<Room> room)
            : m_room(room)
            { }

        //! Destructor
        ~Office();

        //! Get room office is located in
        std::shared_ptr<Room> getRoom() const
            {
            return m_room;
            }

    private:
        std::shared_ptr<Room> m_room;   //!< Room office is located in
    };
```

## Structs

Classes should always be preferred to structs, except for cases
where the object represents a data structure. For structs, member
variables should be all lowercase, and should not be prefaced with
`m_` since they are default publicly accessible.
```c++
//! Space in a building
struct Room
    {
    double3 dimensions; //!< Length dimensions of the room
    size_t  capacity;   //!< Number of people that the room can hold
    };
```

## Enums

Prefer scoped enums where possible. Either use the C++11 syntax for this, or
wrap the enum in a struct.
```c++
enum struct axis
    {
    x=0,
    y,
    z
    };

struct color
    {
    enum colorEnum
        {
        red=0,
        green,
        blue
        }
    };

axis d = axis::x;
color::colorEnum c = color::red;
```

## Use include guards

Include guards should be used in all headers. The guard should be named after the
directory and file, and be followed by a trailing underscore. For example, the
include guard for `azplugins/CoolUpdater.h` should be `AZPLUGINS_COOL_UPDATER_H_`.

## Include the necessary headers, and do not include extraneous headers

It is better to be more explicit including headers. Do not rely on another
class to include the header you need, since this may break in the future.
Do not leave behind headers that are unused, e.g. `<iostream>`.

## Throw errors in headers that cannot be compiled by NVCC

Headers that include STL headers, pybind11, etc. cannot be compiled by
NVCC. A preprocessor error should be raised to warn the developer of
complications arising from this.

## const correctness

Use const for variables that are intended to be constant. Also use const for
class member functions where appropriate, and for references that are passed
or returned.

## Comment style

Short, single line comments should be left using `//`, while longer comment
blocks should use `/* */`. Use asterisks to mark the sides of your long comment
blocks.
```c++
// Here is a short comment

/*
 * This is a longer comment that describes the implementation in
 * more detail than I could fit in one line.
 */
```

## Remove unused code

Code that is commented-out should not be left behind. Most likely, you should
delete this unused code, and let git take care of it. If it must be left around
"just in case", remove comment marks and place it in an `#if 0` block to prevent
compilation.

## Code line length

Keep code lines to less than 120 characters. If the code will overflow
this amount, consider wrapping it in a sensible way
```c++
// ok
void foobar(int foo, size_t bar)
    {
    }

// Needs arguments to be split on lines
void foobar_long_stuff_here(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<DomainDecomposition> decomposition)
    {
    // ...
    }
```

## Brace style

Braces should come on the following line,
```c++
void foobar()
    {
    // ...
    }

// scoped code block
    {
    int scoped_num = 2;
    }

for (size_t i=0; i < vec.size(); ++i)
    {
    // long code block
    }
```

One-line code blocks without braces are OK, but strongly discouraged since they
can decrease readability. The most common use case might be simple if statements
```c++
if (x >= L) x -= L
else if (x < 0) x += L
```
In these cases, it is sometimes better to put the statement all on one-line so
that it is clear that a second statement cannot be added without indentation.

## Assertions versus exceptions

Use assertions to test for developer / debug errors. Assertions should be used
for things that you never expect to happen. Use exceptions to process and sanitize
user input, and raise sensible exceptions. Do not throw exceptions in constructors
unless the object has been fully initialized.

# CUDA guidelines

## Namespaces

Namespaces should be used for all code. All code must reside in the `azplugins`
namespace. GPU kernel drivers must be placed into `azplugins::gpu`,
and their companion kernels should be in `azplugins::gpu::kernel`.

## Use include guards

Include guards should be used in all headers. The guard should be named after the
directory and file, and be followed by a trailing underscore. For example, the
include guard for `azplugins/CoolUpdaterGPU.cuh` should be `AZPLUGINS_COOL_UPDATER_GPU_CUH_`.

## Kernel naming conventions

Kernel drivers should be named after what they do, and their kernels should have
the same names. They must be placed properly into their namespaces.

```c++
azplugins::gpu::compute_my_force
azplugins::gpu::kernel::compute_my_force
```

## Connect GPU kernels to autotuners

Autotuning kernel launch parameters boosts performance significantly, and
ensures good performance across multiple hardware architectures. Use an
Autotuner on all kernel launches.

## Error check kernel launches

After each kernel call, detect if error checking is enabled.
```c++
if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
```

## Optimize for the latest hardware

Optimize your code to run on the latest hardware. Do not maintain workarounds
for hardware older than compute capability 3.5. (This typically means do not
include a "Fermi workaround" for emulating large 1D grids, and do not waste
effort with textures.)

## Use intrinsics for global memory reads

Take advantage of `__ldg` intrinsic for compute capability 3.5 or newer.

## Cache small parameters into shared memory

For per-type or per-pair coefficients, cache these values into shared memory
at kernel launch time.

## Use libraries where available

Take advantage of efficient routines that already exist in CUB, Thrust, or
internal to HOOMD (e.g., WarpTools).


# Python guidelines

## Python naming conventions

Python classes should be named using lower case and underscores. For example,
a wall potential named `WallPotentialLJ93` would be `wall.lj93`. Modules should
be used to logically organize similar objects.

## Unit test each object individually

Each Python object should have an analogous test file in `test-py`, which should
be named after the object. For `wall.lj93`, the test file would be `test_wall_lj93.py`.

## Prefer ducktyping

If it walks like a duck and quacks like a duck, then it's probably a duck.
Allow flexible input types from the user, and prefer `try-except` blocks.
Do not explicit test parameter types unless necessary.


# git guidelines

Configure your git to follow these guidelines with:
```bash
git config --global core.whitespace trailing-space,space-before-tab,tab-in-indent,tabwidth=4
```

and then for the azplugins repository
``` bash
cd /path/to/azplugins/.git/hooks
mv pre-commit.sample pre-commit
```

git is now configured to reject any commits with trailing spaces or tabs instead of spaces.
(Note: you **must** perform the second step for any new clone of the azplugins repository
you make. The `git config` need only be performed once.)

## Commit style

Use imperatives in your commit messages. For example, prefer,
"Add unit tests" to "Added unit tests".

## Propose a single set of related changes

Keep your changes to a minimum so that only the code relevant to the
feature / fix is modified in the branch. (Try to fix spelling mistakes,
etc. in a separate branch.) This vastly simplifies the pull request
review process.

## Merge up master to avoid conflicts

All merge conflicts must be resolved before the pull request will be accepted.

## Prefer 4 spaces to tabs

Configure your editor to put 4 spaces rather than a tab character.

## Remove trailing whitespaces

Configure your editor to remove trailing whitespaces.

## Ensure files end with a new line

Some editors (vim) automatically add a new, blank line at the end of a file, but
not all do.
