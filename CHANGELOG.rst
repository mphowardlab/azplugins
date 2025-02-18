.. Copyright (c) 2018-2020, Michael P. Howard
.. Copyright (c) 2021-2025, Auburn University
.. Part of azplugins, released under the BSD 3-Clause License.

Release notes
=============

v1
---

v1.1.0
^^^^^^
Released 24 January 2024

*New features*

* ``bond.Quartic`` is a new bond potential with tunable quartic form.
* ``compute.CartesianVelocityFieldCompute`` computes the 3D velocity field in
  1, 2, or 3 spatial dimensions using Cartesian coordinates.
* ``compute.CylindricalVelocityFieldCompute`` computes the 3D velocity field in
  1, 2, or 3 spatial dimensions using cylindrical coordinates.
* ``compute.VelocityCompute`` computes the mass-averaged velocity of a group of
  particles.
* ``external.PlanarHarmonicBarrier`` is a harmonic barrier in a planar geometry.
  Note that this is a reimplementation of ``evaporate.implicit`` with
  ``geometry="film"`` in v0.12.0.
* ``external.SphericalHarmonicBarrier`` is a harmonic barrier in a spherical
  geometry. Note that this is a reimplementation of ``evaporate.implicit`` with
  ``geometry="droplet"`` in v0.12.0.
* ``pair.ExpandedYukawa`` is a pair potential that extends
  ``hoomd.md.pair.Yukawa`` to account for particles with different diameters.

*Other changes*

* HOOMD 5 is now the minimum dependency.
* Update copyright year.

v1.0.0
^^^^^^
Released 26 July 2024

This major release is a complete rewrite of azplugins to support HOOMD 4. The
API and build system have been updated to conform to the HOOMD component
template. It was decided to remove some features during the porting process
if they were rarely used or had equivalent implementations in HOOMD. Some other
features will be ported but are not available in this release. Contributions
reimplementing features that have not been ported are certainly welcome!

v0
---

v0.12.0
^^^^^^^
Released 1 Nov 2022

This will be the final minor release of azplugins to support HOOMD-blue v2.x.
The next planned release of azplugins will be a major release to support HOOMD
v3.x. This will be a breaking change because of the significant rewrite of
HOOMD-blue's API. We plan to support bug fixes until this major release is
ready, and we may support bug fixes targeted toward HOOMD-blue v2.x during a
transition period.

*New features*

* Add shift parameter and allow sigma & epsilon to be zero in ``bond.fene``.
* Add ``analyze.group_velocity`` to compute the center-of-mass velocity of a
  group.

*Bug fixes*

* Apply CUDA 11.5 thrust / CUB namespace wrap fix.

v0.11.0
^^^^^^^
Released 1 Feb 2022

*New features*

* ``flow.FlowProfiler`` is generalized to support multiple velocity components,
  mass averaged velocities, and to compute temperature.
* The documentation has been reorganized, and some tutorials have been added.
  The docs are now hosted on `ReadTheDocs <https://azplugins.readthedocs.io>`_.
* Support is added for sinusoidal channels and expansion--constriction channels
  in ``integrate.sinusoidal_channel`` and
  ``integration.sinusoidal_expansion_constriction``. Similar support is also
  available for the MPCD solvent.

*Other changes*

* Add a code of conduct for contributors.
* Update copyright. azplugins is now maintained as part of our work at Auburn
  University.
* The default git branch is renamed ``main``. More `information
  <https://sfconservancy.org/news/2020/jun/23/gitbranchname>`_ is available.
* CI testing has been moved to GitHub Actions.

v0.10.1
^^^^^^^
Released 12 Jan 2021

*Bug fixes*

* Fix normalization of velocity in ``flow.FlowProfiler``.

v0.10.0
^^^^^^^
Released 27 Jul 2020

This is the first release of azplugins to include compilable documentation. The
``docs`` can be built using sphinx. Currently, only the APIs are thoroughly
documented, but we will be expanding the rest of the documentation in future
releases with examples and tutorials.

*New features*

* ``restrain.cylinder`` and ``restrain.sphere`` allow particles to be
  harmonically restrained by their distance relative to a cylinder or sphere,
  respectively. ``restrain.plane`` has been updated to share a condensed API
  using HOOMD walls.
* ``bond.double_well`` adds a new bond potential having two tunable minima.
* ``flow.FlowProfiler`` adds a python-level analyzer for averaging 1d density
  and velocity profiles for both HOOMD and MPCD systems.
* ``pair.hertz`` adds the Hertz pair potential.

*Other changes*

* The python unit tests have been updated to remove ``import *``.

v0.9.2
^^^^^^
Released 3 Mar 2020

*Bug fixes*

* Fix a compilation error on MacOS clang builds.

*Other changes*

* Update copyright to 2020.

v0.9.1
^^^^^^
Released 18 Dec 2019

*Bug fixes*

* Fix a compilation error in CUDA-enabled builds.

v0.9.0
^^^^^^
Released 15 Dec 2019

*New features*

* ``flow.constant`` implements a constant flow along a vector.
* ``variant.sphere_area`` adds a new variant that is physically motivated by a
  sphere shrinking with a constant rate of change in area. This may be useful
  with ``evaporate.implicit`` in the ``droplet`` geometry.

*Other changes*

* Support API changes in HOOMD 2.8.0. Backward compatibility is maintained
  through a new API header.
* The pair potential evaluators have been updated to support HOOMD 2.8.0. New
  pair potential evaluators should derive from one of the convenience base
  classes.
* CI testing has been added for HOOMD 2.8.0 in addition to 2.6.0.

v0.8.0
^^^^^^
Released 5 Nov 2019

*New features*

* ``evaporate.implicit`` now supports evaporation in both film and droplet
  geometries. The default geometry remains the film.
* ``restrain.plane`` allows particles to be harmonically restrained by their
  distance relative to a plane.

v0.7.1
^^^^^^
Released 20 Aug 2019

*Bug fixes*

* Silence a warning in CMake >= 3.12.
* Fix a link error in compiled unit tests.

v0.7.0
^^^^^^
Released 24 Jun 2019

This version of the plugin **requires** HOOMD-blue v2.6.0 for compatibility with
the new streaming geometries in its MPCD component. HOOMD-blue **must** be built
with the MPCD component.

*Bug fixes*

* Fix compilation errors with HOOMD-blue v2.6.0.

*Other changes*

* random123 is used as the random number generator throughout azplugins. This
  API is more robust and stable than Saru, but sequences of random numbers drawn
  for a given seed will change. New features using random numbers should add a
  unique 32-bit identifier to ``RNGIdentifiers.h``.

v0.6.2
^^^^^^
Released 25 Apr 2019

All commits and pull requests are now automatically tested against HOOMD 2.5.1
on CircleCI. Unit tests are run for CPU-only build configurations. CUDA-enabled
builds are tested for compilation, but their unit tests cannot be run on
CircleCI. The CI test environment is available on Docker Hub
(https://hub.docker.com/r/mphoward/ci), and tests for new code should be run
locally on a GPU.

*Bug fixes*

* Fix import statements in azplugins modules for python3.
* Fix HOOMD version parsing in CMake for external builds.
* Fix CMake errors in testing for certain build configurations.

v0.6.1
^^^^^^
Released 28 Mar 2019

*Bug fixes*

* Fix thrust template parameters in ``mpcd.reverse_perturbation`` for CUDA 9 &
  10.

v0.6.0
^^^^^^
Released 25 Feb 2019

This version of the plugin **requires** HOOMD-blue v2.3.0 for compatibility with
the embedded pybind11 library. Be sure to update your git submodules when
recompiling and installing HOOMD-blue.

azplugins is now maintained by Michael P. Howard and will be hosted on GitHub
(https://github.com/mphoward/azplugins).

*New features*

* ``mpcd.reverse_perturbation`` implements the reverse perturbation method in
  the optional MPCD module to simulate shear flow.

*Other changes*

* The azplugins license and packaging has been updated to reflect the project
  continuation.
* ``FindHOOMD.cmake`` has been improved to support ``find_package`` and detect
  the installed version of HOOOMD.

v0.5.0
^^^^^^
Released 11 Jun 2018

This version of the plugin **requires** HOOMD-blue v2.2.2 in order to ensure all
necessary header files are available.

*New features*

* ``flow.reverse_perturbation`` implements the reverse perturbation method for
  generating shear flow. This implementation is significantly more stable than
  the HOOMD-blue release, but does not currently support MPI.
* ``integrate.slit`` supports NVE integration with bounce-back rules in the slit
  geometry. Other bounce back geometries can also be configured.
* ``dpd.general`` implements a generalized DPD potential where the exponent of
  the dissipative weight function can be adjusted. A framework is also
  implemented for adding other DPD potentials.
* ``flow.langevin`` and ``flow.brownian`` support Langevin and Brownian dynamics
  in external flow fields. Currently, the supported fields are
  ``flow.quiescent`` and ``flow.parabolic``, but additional fields can be
  included by templating.

v0.4.0
^^^^^^
Released 16 Nov 2017

This version of the plugin **requires** HOOMD-blue v2.2.1 in order to ensure all
necessary header files are available.

*New features*

* A framework is configured for developing bond potentials.
* ``bond.fene`` implements a standard FENE potential that is cleaned up compared
  to the version found in HOOMD.
* ``bond.fene24`` implements the FENE potential with the Ashbaugh-Hatch-style
  48-24 Lennard-Jones potential repulsion.
* ``pair.ashbaugh24`` implements a Ashbaugh-Hatch 48-24 Lennard-Jones potential.
* ``pair.spline`` implements a cubic spline potential.
* ``pair.two_patch_morse`` implements the two-patch Morse anisotropic potential.
* A framework is configured for developing special pair potentials from existing
  pair potentials.
* ``special_pair.lj96`` implements the LJ 9-6 potential as a special pair.
* A framework is configured for writing and running compiled unit tests with
  upp11.
* All source code is now automatically validated for formatting.

*Bug fixes*

* Fix path to cub headers.
* Add missing status line prints.
* Fix possible linker errors for MPI libraries.
* Plugins now build when ``ENABLE_CUDA=OFF``.
* CMake exits gracefully when the MD component is not available from hoomd.
* Plugins now compile with debug flags.

v0.3.0
^^^^^^
Released 22 Aug 2017

This version of the plugin is now tested against HOOMD-blue v2.1.9. Users
running older versions of HOOMD-blue are recommended to upgrade their
installations in order to ensure compatibility and the latest bug fixes on the
main code paths.

*New features*

* ``pair.lj124`` implements the 12-4 Lennard-Jones potential.
* ``pair.lj96`` implements the 9-6 Lennard-Jones potential.
* A framework is configured for developing anisotropic pair potentials.

*Bug fixes*

* Fix import hoomd.md error in ``analyze.rdf``.
* Adds restraint module to ctest list and warns user about running with
  orientation restraints in single precision.
* Fix examples in contribution guidelines so that formatting of pull request
  checklist is OK.
* Remove unused include from particle evaporator which caused compilation errors
  with newer versions of hoomd where the header was removed.

v0.2.0
^^^^^^
Released 28 Feb 2017

*New features*

* ``analyze.rdf`` implements a radial distribution function calculator between
  particle groups for small problem sizes.
* ``restrain.position`` implements position restraints for particles.
* ``restrain.orientation`` implements orientation restraints for particles.
* ``pair.slj`` implements a core-shifted Lennard-Jones potential that does not
  read from the particle diameters.

*Other updates*

* Source code guidelines and a pull request checklist are discussed in a
  new ``CONTRIBUTING.md``.

v0.1.0
^^^^^^
Released 9 Feb 2017

*New features*

* A framework is configured for developing pair potentials.
* ``pair.ashbaugh`` implements the Ashbaugh-Hatch (Lennard-Jones perturbation)
  potential.
* ``pair.colloid`` implements the colloid (integrated Lennard-Jones) potential
  for colloidal suspensions.
* A framework is configured for developing wall potentials.
* ``wall.colloid`` implements the integrated Lennard-Jones potential between a
  colloid and a half-plane wall.
* ``wall.lj93`` implements the Lennard-Jones 9-3 potential between a point and a
  half-plane wall.
* ``update.types`` allows for swapping of particle types based on a slab region
  of the simulation box.
* ``evaporate.particles`` supports evaporation of single-particle fluids from a
  slab region of the simulation box.
* ``evaporate.implicit`` provides an implicit model for an evaporating solvent.
