.. Copyright (c) 2018-2020, Michael P. Howard
.. Copyright (c) 2021-2024, Auburn University
.. Part of azplugins, released under the BSD 3-Clause License.

Development
===========

This page contains some helpful notes for developers.

Pull requests
-------------
All pull requests should be based off the ``main`` branch. To help organize
branches, it is helpful to give new branches a descriptive name with one
of the following labels prepended:

* ``feature/``: new features
* ``fix/``: bug fixes
* ``refactor/``: code refactoring

In general, pull requests will be squash merged to keep the git history clean.
If you don't want your pull request squash merged, make sure to explicitly note
this to a code maintainer. You will likely be asked to manually squash your commit
history.

In order to merge, a pull request must pass all required unit tests, and the
the documentation must build correctly.

Testing framework
-----------------
azplugins relies on Python ``unittest`` to test new features. New tests should be added
for any features or bug fixes. You should make sure that all unit tests pass in your
own build configuration. A selected set of different compilers and build configurations
will automatically run on GitHub Actions once you open a pull request. These tests run
inside the HOOMD-blue Docker images. Note that these configurations give pretty good
coverage of typical builds, but are not exhuastive to keep the testing requirements from
getting too burdensome. Also, GitHub Actions does not have GPU runners, so GPU code is
tested for *compilation* but not *execution*. If you add new GPU code, it is your job to
run the unit tests on your local GPU. A code maintainer will ask you to confirm that you
have performed this test in your pull request.

Documentation
-------------
All new code should be thoroughly documented. At the Python level, all features must have
Sphinx documentation that is linked into the appropriate pages and indexes. At the C++
level, doxygen comments are encouraged for all code, but this is not currently compiled.
