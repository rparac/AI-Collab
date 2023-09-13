from typing import Iterator, Tuple, Union


def adjacent_cells_iterator(curr_pos: Tuple[int, int], grid_size: Union[Tuple[int, int], int],
                            eight_dir: bool = False, include_current: bool = True) -> \
        Iterator[Tuple[int, int]]:
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    pos_i, pos_j = curr_pos
    
    dirs_i = [ 0, 1, 0, -1]
    dirs_j = [-1, 0, 1,  0]

    if include_current:
        dirs_i.append(0)
        dirs_j.append(0)

    if eight_dir:
        dirs_i.extend([-1, 1,  1, -1])
        dirs_j.extend([ 1, 1, -1, -1])
        
    for dir_i, dir_j in zip(dirs_i, dirs_j):
        next_i, next_j = pos_i + dir_i, pos_j + dir_j
        if inbounds(grid_size, next_i, next_j):
            yield next_i, next_j


def inbounds(size: (int, int), i: int, j: int) -> bool:
    size_i, size_j = size
    return 1 <= i < size_i and 1 <= j < size_j
