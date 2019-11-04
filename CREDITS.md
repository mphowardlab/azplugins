# Credits

## Contributors

**Michael P. Howard**, _Lead developer_

- Original software design and plugin frameworks.
- `analyze.rdf`: Radial distribution function calculator.
- `dpd.general`: Generalized DPD pair potential.
- `evaporate.implicit`: Moving harmonic potential to simulate drying.
- `evaporate.particles`: Particle type changer to simulate explicit drying.
- `flow.brownian` / `flow.langevin` and imposed flow framework.
- `integrate.slit` and bounce-back integrator framework.
- `pair.ashbaugh`: Ashbaugh-Hatch pair potential.
- `pair.colloid`: Integrated Lennard-Jones pair potential.
- `pair.slj`: Modified core-shifted Lennard-Jones pair potential.
- `restrain.plane`: Harmonic plane restraints.
- `update.types`: Type changer by region.
- `wall.colloid`: Integrated Lennard-Jones wall potential.
- `wall.lj93`: 9-3 Lennard-Jones wall potential.

**Sally Jiao**

- `pair.lj124`: 12-4 Lennard-Jones pair potential.
- `pair.lj96`: 9-6 Lennard-Jones pair potential.
- `SpecialPairEvaluator`: Template class for special pair evaluators.

**Wes Reinhart**

- `pair.two_patch_morse`: Anisotropic two-patch Morse potential.
- `restrain.position`: Harmonic position restraints.
- `restrain.orientation`: Harmonic orientation restraints.

**Antonia Statt**

- `bond.fene`: modified FENE bond evaluator.
- `bond.fene24`: FENE bond evaluator with Ashbaugh-Hatch 48-24 pair potential.
- `flow.reverse_perturbation`: reverse perturbation shear flow.
- `mpcd.reverse_perturbation`: reverse perturbation shear flow for MPCD.
- `pair.ashbaugh24`: 48-24 Ashbaugh-Hatch pair potential.
- `pair.spline`: Spline polynomial pair potential.

## Source code

This is a continuation of the `azplugins` project begun by members of the
Panagiotopoulos Group at Princeton University. Code from that project is
used under a Modified BSD license:
```
Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
