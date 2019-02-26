from __future__ import print_function
import sys
import re

pattern = re.compile('#define HOOMD_VERSION "([0-9\.]+)"')
try:
    with open(sys.argv[1]) as f:
        for line in f:
            match = pattern.match(line)
            if match:
                print(match.group(1), end="")
                exit()
except:
    pass

print("HOOMD_VERSION-NOTFOUND", end="")
