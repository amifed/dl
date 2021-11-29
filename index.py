import getopt
import sys

print(sys.argv)
opts, _ = getopt.getopt(sys.argv[1:], "a:d:S",)
opt_dict = {k: v for [k, v] in opts}
print(opt_dict)
