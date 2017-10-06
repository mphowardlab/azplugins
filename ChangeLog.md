# AZPlugins Change Log

[TOC]

---------
## v0.4.0

Not yet released.

*Release notes*

This version of the plugin **requires** HOOMD-blue v2.2.1 in order
to ensure all necessary header files are available.

*New features*

* `bond.fene` implements a standard FENE potential that is cleaned up compared
   to the version found in HOOMD.
* `pair.ashbaugh24` implements a Ashbaugh-Hatch 48-24 Lennard-Jones potential.
* `pair.spline` implements a cubic spline potential.
* `pair.two_patch_morse` implements the two-patch Morse anisotropic potential.
* A framework is configured for developing bond potentials.
* A framework is configured for developing special pair potentials.

*Bug fixes*

* Fix path to cub headers.

## v0.3.0

22 August 2017

*Release notes*

This version of the plugin is now tested against HOOMD-blue v2.1.9.
Users running older versions of HOOMD-blue are recommended to upgrade
their installations in order to ensure compatibility and the latest
bug fixes on the main code paths.

*New features*

* `pair.lj124` implements the 12-4 Lennard-Jones potential.
* `pair.lj96` implements the 9-6 Lennard-Jones potential.
* A framework is configured for developing anisotropic pair potentials.

*Bug fixes*

* Fix import hoomd.md error in `analyze.rdf`.
* Adds restraint module to ctest list and warns user about running
  with orientation restraints in single precision.
* Fix examples in contribution guidelines so that formatting of
  pull request checklist is OK.
* Remove unused include from particle evaporator which caused
  compilation errors with newer versions of hoomd where the header
  was removed.

## v0.2.0

28 February 2017

*New features*

* `analyze.rdf` implements a radial distribution function calculator
  between particle groups for small problem sizes.
* `restrain.position` implements position restraints for particles.
* `restrain.orientation` implements orientation restraints for particles.
* `pair.slj` implements a core-shifted Lennard-Jones potential that does
  not read from the particle diameters.

*Other updates*

* Source code guidelines and a pull request checklist are discussed in a
  new `CONTRIBUTING.md`.

## v0.1.0

9 February 2017

*New features*

* A framework is configured for developing pair potentials.
* `pair.ashbaugh` implements the Ashbaugh-Hatch (Lennard-Jones perturbation)
   potential.
* `pair.colloid` implements the colloid (integrated Lennard-Jones) potential
  for colloidal suspensions.
* A framework is configured for developing wall potentials.
* `wall.colloid` implements the integrated Lennard-Jones potential between
  a colloid and a half-plane wall.
* `wall.lj93` implements the Lennard-Jones 9-3 potential between a point
  and a half-plane wall.
* `update.types` allows for swapping of particle types based on a slab region
  of the simulation box.
* `evaporate.particles` supports evaporation of single-particle fluids from
  a slab region of the simulation box.
* `evaporate.implicit` provides an implicit model for an evaporating solvent.
