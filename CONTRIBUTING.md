Contributions are encouraged using pull requests on Bitbucket for
feature / fix branches.

# Guidelines for features

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
error handling at the python level.

## Connect GPU kernels to autotuners

Autotuning kernel launch parameters boosts performance significantly, and
ensures good performance across multiple hardware architectures. Use an
Autotuner on all kernel launches.

## Propose a single set of related changes

Keep your changes to a minimum so that only the code relevant to the
feature / fix is modified in the branch. (Try to fix spelling mistakes,
etc. in a separate branch.) This vastly simplifies the pull request
review process.

## Use current copyright notice

This makes it easier to update the copyright year over year.

## Correctly identify the maintainer of the code

Typically, this would be you as the submitter.

# Source code conventions

## Code line length

Keep code lines to less than 120 characters. If the code will overflow
this amount, consider wrapping it in a sensible way

```c++
// ok
void foobar(int foo, size_t bar)
    {
    }

// Needs arguments to be split on lines
void foobar_long_stuff(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<DomainDecomposition> decomposition)
    {
    // ...
    }
```


## Prefer 4 spaces to tabs

This is an age-old debate, but configure your editor to put 4-spaces
rather than a tab character.

## Brace style

Braces should come on the following line,

```c++
void foobar()
    {
    // ...
    }
```

## Classes

Classes should be named as nouns of the objects they represent.
Class names should have the first letter of each word
capitalized, e.g. ForceCompute. Class methods should be
camelcase (e.g., getNetForce), and should be verbs describing their
action. Member variables should begin with `m_`, and should be all
lower case.

Classes should always be preferred to structs, except for cases
where the object represents a data structure. For structs, member
variables should be all lowercase, and should not be prefaced with
`m_` since they are default publicly accessible.

## Commit style

Use imperatives in your commit messages. For example, prefer,
"Add unit tests" to "Added unit tests".
