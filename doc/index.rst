.. Copyright (c) 2018-2020, Michael P. Howard
.. Copyright (c) 2021-2025, Auburn University
.. Part of azplugins, released under the BSD 3-Clause License.

=========
azplugins
=========

azplugins is a component for `HOOMD-blue`_ which expands its functionality for
tackling a variety of problems in soft matter physics. Currently, azplugins is
tested against v5.0.0 of HOOMD-blue.

Compiling
=========

azplugins follows the `component template`_. It has the same dependencies used
to build HOOMD-blue. With HOOMD-blue installed already, adding azplugins can be
as easy as:

1. Grab a copy of the code:

   .. code-block:: bash

     git clone https://github.com/mphowardlab/azplugins

2. Configure azplugins with CMake:

   .. code-block:: bash

     cmake -B build/azplugins -S azplugins

3. Build azplugins:

   .. code-block:: bash

     cmake --build build/azplugins

4. Install azplugins alongside HOOMD-blue:

   .. code-block:: bash

     cmake --install build/azplugins

Please refer to the directions in the HOOMD-blue documentation on building an
external component for more information.

Contents
========

.. toctree::
    :maxdepth: 1
    :caption: API

    module-azplugins-bond
    module-azplugins-compute
    module-azplugins-external
    module-azplugins-flow
    module-azplugins-pair

.. toctree::
    :maxdepth: 1
    :caption: Reference

    credits
    license
    release

Index
=====

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _HOOMD-blue: http://glotzerlab.engin.umich.edu/hoomd-blue
.. _component template: https://hoomd-blue.readthedocs.io/en/latest/components.html
