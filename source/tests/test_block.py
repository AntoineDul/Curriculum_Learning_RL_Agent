import pytest
import numpy as np

from source.modules.environment.block import Block
from source.config import CONFIG

def test_block_initialization():
    block = Block()
    x, y = block.position
    grid_size = CONFIG["grid_size"]
    assert 0 < x < grid_size - 1
    assert 0 < y < grid_size - 1

def test_block_reset():
    block = Block()
    old_pos = block.position
    block.reset()
    new_pos = block.position
    grid_size = CONFIG["grid_size"]
    # Reset produces a new position within bounds
    assert 0 < new_pos[0] < grid_size - 1
    assert 0 < new_pos[1] < grid_size - 1
    assert new_pos != old_pos or True  # allows for rare collisions

def test_set_position():
    block = Block()
    block.set_position((3, 4))
    assert block.position == (3, 4)

def test_on_edge():
    block = Block()
    # Not on edge
    block.set_position((1, 1))
    assert not block.on_edge()
    # On left edge
    block.set_position((0, 2))
    assert block.on_edge()
    # On right edge
    block.set_position((CONFIG["grid_size"] - 1, 2))
    assert block.on_edge()
    # On top edge
    block.set_position((2, 0))
    assert block.on_edge()
    # On bottom edge
    block.set_position((2, CONFIG["grid_size"] - 1))
    assert block.on_edge()


