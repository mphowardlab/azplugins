#!/bin/bash

# Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

if [ "$#" -ne 1 ]; then
    echo "Usage: check_format.sh /dir/to/check"
    exit 1
fi

## Prints 1 if a list $1 contains an entry $2, and 0 otherwise.
#
# https://stackoverflow.com/questions/8063228/how-do-i-check-if-a-variable-exists-in-a-list-in-bash
#
contains()
    {
    [[ $1 =~ (^|[[:space:]])$2($|[[:space:]]) ]] && echo 1 || echo 0
    }

RESULT=0
for f in $(find $1 -name '*.cc' -or -name '*.cu' -or -name '*.h' -or -name '*.cuh' -or -name '*.py')
do
    # check for trailing whitespace
    N=$(grep -c "[[:blank:]]$" $f)
    if [ "$N" -gt "0" ]
    then
        echo "Trailing whitespace in $f:"
        grep -n "[[:blank:]]$" $f
        RESULT=1
    fi

    # check for tabs
    N=$(grep -P -c "\t" $f)
    if [ "$N" -gt "0" ]
    then
        echo "Tabs in $f:"
        grep -n -P "\t" $f
        RESULT=1
    fi

    # check for semicolons in python files
    if [ "${f: -3}" == ".py" ]
    then
        N=$(grep -c ";" $f)
        if [ "$N" -gt "0" ]
        then
            echo "Semicolons in $f:"
            grep -n ";$" $f
            RESULT=1
        fi
    fi

    # check for copyright notice and year
    filename=$(basename "$f")
    skip="BoundaryCondition.h SlitGeometry.h"
    skip_file=$(contains "$skip" "$filename")
    if [ "$skip_file" -eq "0" ]
    then
        N=$(grep -c "Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University" $f)
        if [ "$N" -ne "1" ]
        then
            echo "Copyright notice incorrect in $f"
            RESULT=1
        fi
    fi

    # check for maintainer line
    N=$(grep -c "Maintainer:" $f)
    if [ "$N" -ne "1" ]
    then
        echo "Missing maintainer in $f"
        RESULT=1
    fi

done

exit $RESULT
