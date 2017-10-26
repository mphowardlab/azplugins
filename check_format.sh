#!/bin/bash

# Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

if [ "$#" -ne 1 ]; then
    echo "Usage: lint_detector.sh /dir/to/check"
    exit 1
fi

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
done

exit $RESULT
