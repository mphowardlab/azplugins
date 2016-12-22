# AZPlugins Change Log

[TOC]

---------
## v0.1.0

Not yet released

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
