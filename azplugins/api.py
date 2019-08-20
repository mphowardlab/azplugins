r"""Utility methods for interfacing with the HOOMD API."""
import hoomd

def parse_version(version):
    """Parse a version string into a tuple.

    Args:
        version (str): A version string separated by '.', like 'x.y.z'.

    Returns:
        The version as a tuple.

    Examples::

        >>> parse_version('1.2.3')
        (1,2,3)

    """
    return tuple([int(x) for x in version.split('.')])

def str_version(version):
    """Stringify a version tuple.

    Args:
        version (tuple): A version tuple.

    Returns:
        The version as a string separated by '.'.

    Examples::

        >>> str_version((1,2,3))
        '1.2.3'
    """
    return '.'.join([str(x) for x in version])

available = parse_version(hoomd.__version__)

def require(version):
    r"""Decorator to require a minimum HOOMD API.

    Args:
        version (str): The minimum required version, encoded as 'x.y.z'.

    If `hoomd.__version__` expands to a tuple that is less than *version*,
    an error message is generated and an `ImportError` is raised.

    """
    # wrap the decorator
    def wrapper(function):
        # wrap the variable
        def _require(*args, **kwargs):
            _version = parse_version(version)
            if available < _version:
                hoomd.context.msg.error('HOOMD {} or newer is required, but {} found.\n'.format(str_version(_version), str_version(available)))
                raise ImportError('Minimum HOOMD version not available')
            function(*args,**kwargs)
        return _require
    return wrapper
