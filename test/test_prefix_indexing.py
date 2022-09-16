from dyci2.equiv import BasicEquiv
from dyci2.label import ListLabel
from dyci2.prefix_indexing import PrefixIndexing

if __name__ == '__main__':
    pattern = [ListLabel([0]), ListLabel([1]), ListLabel([None])]
    sequence = [ListLabel([i]) for i in range(10)]

    f = PrefixIndexing.prefix_indexing(sequence, pattern, equiv=BasicEquiv.eq, print_info=False)
    print(f)
    # ({3: [0, 1, 2, 3, 4, 5, 6, 7], 2: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]}, 3)

    print(BasicEquiv.eq(ListLabel([0]), ListLabel([None])))