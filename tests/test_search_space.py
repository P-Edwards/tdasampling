import numpy as np
from tdasampling.search_space import Search_Space

def test_Search_Space():
    space = Search_Space(1, 1e-7, [-1, 1, -1, 1])
    assert space.dimension == 2
    assert space.epsilon == 1
    assert space.delta == 1e-7
    assert space.id_counter == 0
    assert space.tree.interleaved == False
    assert len(space.bad_boxes) == 1
    assert (space.bad_boxes[0].box == np.array([-1, 1, -1, 1])).all()
    assert space.global_bounds == [-1, 1, -1, 1]
    assert space.problem_bounds == [-1, 1, -1, 1]
    assert space.current_max_length == 4

def test_shuffle():
    space = Search_Space(1, 1e-7, [-1, 1, -1, 1])
    space.bad_boxes = [1,2,3,4,5]
    space.shuffle()
    for i in [1,2,3,4,5]:
        assert i in space.bad_boxes

def test_addPoint():
    space = Search_Space(0.01, 1e-7, [-1, 1, -1, 1])
    assert list(space.tree.intersection([0.1,0.11,0.1,0.11])) == []
    assert list(space.tree.intersection([0.1,0.11,-1,-0.9])) == []

    space.addPoint([0.15,0.15], radius = 0.05)
    assert list(space.tree.intersection([0.1,0.11,0.1,0.11])) == [0]
    assert list(space.tree.intersection([0.1,0.11,-1,-0.9])) == []

    space.addPoint([0.15,0.15], is_sample_point=True)
    assert list(space.tree.intersection([0.1,0.11,0.1,0.11])) == [0]
    assert list(space.tree.intersection([0.1,0.11,-1,-0.9])) == []
    assert list(space.tree.intersection([0.15,0.16,0.15,0.16])) == [0,1]

    space.addPoint([-2,1], is_sample_point=True)
    assert list(space.tree.intersection([-2.1,-2,1,1.1])) == []

    space.addPoint([0.15,0.15], is_sample_point=True, skip_on_covered=True)
    assert list(space.tree.intersection([0.15,0.16,0.15,0.16])) == [0,1]
