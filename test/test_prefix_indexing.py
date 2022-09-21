import sys

from dyci2.equiv import BasicEquiv
from dyci2.label import ListLabel
from dyci2.prefix_indexing import PrefixIndexing

if __name__ == '__main__':
    print(ListLabel([1]) == ListLabel([None]))
    print(BasicEquiv.eq(ListLabel([None]), ListLabel([1])))
    sys.exit(1)

    pattern = [ListLabel([None]), ListLabel([6]), ListLabel([None]), ListLabel([7])]
    sequence = [ListLabel([i % 10]) for i in range(50)]
    # sequence[1] = ListLabel([None])

    print(pattern)
    print(sequence)

    f = PrefixIndexing.prefix_indexing(sequence, pattern, equiv=BasicEquiv.eq, print_info=False)
    print(f)
    # ({3: [0, 1, 2, 3, 4, 5, 6, 7], 2: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]}, 3)



    print(BasicEquiv.eq(ListLabel([0]), ListLabel([None])))

