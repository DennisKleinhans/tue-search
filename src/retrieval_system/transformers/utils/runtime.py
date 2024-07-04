import cProfile
import pstats
from pstats import SortKey


def measure_effective_runtime(string, file="restats", n=10):
    cProfile.run(string, file)
    p = pstats.Stats(file)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(n)