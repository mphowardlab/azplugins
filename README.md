# azplugins

azplugins is a component for [HOOMD-blue][1] which expands its functionality for
tackling a variety of problems in soft matter physics. Currently, azplugins is
tested against v6.0.0 of HOOMD-blue. See [CHANGELOG.rst](CHANGELOG.rst) for a
list of recent development.

## Compiling azplugins

azplugins follows the [standard component template][2]. It has the same
dependencies used to build HOOMD-blue. With HOOMD-blue installed already, adding
azplugins can be as easy as:

```
git clone https://github.com/mphowardlab/azplugins
cmake -B build/azplugins -S azplugins
cmake --build build/azplugins
cmake --install build/azplugins
```

Please refer to the directions in the HOOMD-blue documentation on building an
external component for more information.

### Testing

After building and installing azplugins, you can run our tests with pytest:

```
python -m pytest --pyargs hoomd.azplugins
```

## Contributing

Contributions are welcomed and appreciated! Fork and create a pull request on
[GitHub][3]. Be sure to follow the [HOOMD-blue guidelines for developers][4]! We
value the input and experiences all users and contributors bring to `azplugins`.

## History

azplugins began as a collection of code shared between students and postdocs at
Princeton University (2016-2018). It is named for their research advisor, Prof.
Athanassios (Thanos) Z. Panagiotopoulos.

[1]: http://glotzerlab.engin.umich.edu/hoomd-blue
[2]: https://hoomd-blue.readthedocs.io/en/latest/components.html
[3]: https://github.com/mphowardlab/azplugins
[4]: https://hoomd-blue.readthedocs.io/en/latest/developers.html
