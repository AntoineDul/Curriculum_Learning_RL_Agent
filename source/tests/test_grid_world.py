import pytest
import numpy as np
from source.modules.environment.grid_world import GridWorld
from source.config import CONFIG

@pytest.fixture
def env():
    CONFIG["mode"] = "block"
    CONFIG["grid_size"] = 5
    CONFIG["max_steps"] = 100
    CONFIG["lambda"] = 1.0  # deterministic actions
    CONFIG["debug_mode_enabled"] = False
    CONFIG["goal_is_moving"] = False
    CONFIG["goal_move_frequency"] = 1
    return GridWorld()

def test_initial_positions(env):
    # Agent and block within bounds
    ax, ay = env.agent_pos
    bx, by = env.block.position
    n = CONFIG["grid_size"]
    assert 0 <= ax < n and 0 <= ay < n
    assert 0 < bx < n-1 and 0 < by < n-1

def test_agent_push_block(env):
    # Place agent next to block
    env.agent_pos = (1, 1)
    env.block.set_position((1, 2))  # block below agent
    env.goal.set_position((1, 4))   # goal below block
    
    # Move down (action=2)
    obs, reward = env.step(2)
    
    # Block should have moved down by 1
    assert env.block.position == (1, 3)
    # Agent should have moved
    assert env.agent_pos == (1, 2)
    # Reward should be 0 (goal not reached yet)
    assert reward == 0

def test_block_on_edge_reset(env):
    # Put block next to bottom edge
    n = CONFIG["grid_size"]
    env.agent_pos = (2, n-2)
    env.block.set_position((2, n-1))
    
    # Move down (action=2) -> block hits edge -> reset
    old_position = env.block.position
    obs, reward = env.step(2)
    
    # Block should no longer be at edge
    bx, by = env.block.position
    assert 0 < bx < n-1 and 0 < by < n-1
    # Agent should have moved only if block was allowed to move
    assert env.agent_pos == (2, n-2) or env.agent_pos == (2, n-1)

def test_goal_reached_reward(env):
    # Place block on goal
    env.block.set_position(env.goal.position)
    obs, reward = env.step(0)  # any action
    assert reward == 1
    assert env.done

def test_max_steps(env):
    env.steps = CONFIG["max_steps"] - 1
    obs, reward = env.step(0)  # take any action
    assert env.done

def test_illegal_move(env):
    # Place agent at top-left corner and move up/left
    env.agent_pos = (0, 0)
    obs, reward = env.step(0)  # up
    assert env.agent_pos == (0, 0)
    obs, reward = env.step(3)  # left
    assert env.agent_pos == (0, 0)

