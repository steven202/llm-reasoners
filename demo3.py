# LLM-Reasoners Demo (Updated for vllm)

import os
import json
import copy
from typing import NamedTuple
from vllm import LLM, SamplingParams
from reasoners import WorldModel, SearchConfig, Reasoner
from reasoners.benchmark import BWEvaluator
from reasoners.algorithm import BeamSearch, MCTS
import reasoners.benchmark.bw_utils as utils
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VAL'] = "/sciclone/home/cwang33/llm-reasoners/LLMs-Planning/planner_tools/VAL"
# Initialize vllm model
# llm = LLM(model="Skywork/Skywork-o1-Open-Llama-3.1-8B", tensor_parallel_size=1)
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,  # Adjust GPU memory usage
    max_model_len=35808,        # Limit max sequence length to fit KV cache
    enforce_eager=True,          # Enable eager mode for stability
    speculative_model="meta-llama/Llama-3.2-1B",
    num_speculative_tokens=5,
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=200  # Ensure output fits within the token limit
)

from reasoners.benchmark import BWEvaluator
import json
# Load prompts
with open('examples/CoT/blocksworld/prompts/pool_prompt_v1.json') as f:
    prompt = json.load(f)

evaluator = BWEvaluator(
    config_file='examples/CoT/blocksworld/data/bw_config.yaml',
    domain_file='examples/CoT/blocksworld/data/generated_domain.pddl',
    data_path='examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json',
    init_prompt=prompt
)

# Prepare prompts for reasoning
prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=4)
example = evaluator.full_dataset[1]
cot_inputs = (
    prompt['icl']
    .replace('<init_state>', example["init"])
    .replace('<goals>', example["goal"])
    .replace('<action>', '')
)

print("Initial State:", example['init'])
print("Goals:", example['goal'])
# ## Chain-of-Thought
# We first experiment with the Chain-of-Thought method.
# Since we are having the simplest generation algorithm, we directly ask the model to generate all the steps.
# We look at the 4-shot prompt and the generated answer.
print("CoT Inputs:", cot_inputs)

# Generate output using vllm
outputs = llm.generate([cot_inputs], sampling_params)
output_text = outputs[0].outputs[0].text.strip()

print("Generated Output:", output_text)
# Clearly that's not a valid solution :( 
# The orange block is on the red block, so we cannot pick up the red block as the first step.
# ## Tree-of-Thought
# Then let's turn to a tree search algorithm, [Tree-of-Thought]((https://arxiv.org/abs/2305.10601)).
# We will need to define a simple world model, and a search algorithm, for the Blocksworld task.
# Define classes for Tree-of-Thought (ToT)

from reasoners import WorldModel, LanguageModel, SearchConfig, State, Reasoner
from reasoners.algorithm import BeamSearch, MCTS
import reasoners.benchmark.bw_utils as utils
from typing import NamedTuple
import copy
import numpy as np


# We use NamedTuple for clearer presentation, you may just use normal tuple if you want a quick experiment.
class BWStateToT(NamedTuple):
    step_idx: int
    action_history: list[str]
    end: bool
# We just use the description str as the action, we use a type alias for better presentation.
# You may directly use str of you want a quick experiment.
BWAction = str

class BlocksWorldModelToT(WorldModel):
    def __init__(self, base_model: LLM, prompt: dict, max_steps: int = 4, batch_size: int = 1) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> BWStateToT:
        return BWStateToT(step_idx=0, action_history=[], end=False)

    def step(self, state: BWStateToT, action: BWAction) -> tuple[BWStateToT, dict]:
        state = copy.deepcopy(state)
        if action != "[PLAN END]":
            state = BWStateToT(
                step_idx=state.step_idx + 1,
                action_history=state.action_history + [action],
                end=False
            )
        else:
            state = BWStateToT(
                step_idx=state.step_idx + 1,
                action_history=state.action_history,
                end=True
            )
        return state, {}

    def is_terminal(self, state: State) -> bool:
        return state.end or state.step_idx >= self.max_steps

class BWConfigToT(SearchConfig):
    def __init__(self, base_model: LLM, prompt: dict, temperature: float = 0.8, n_candidate: int = 4) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature

    def get_actions(self, state: BWStateToT) -> list[BWAction]:
        prompts = (
            self.prompt["icl"]
            .replace("<action>", "\n".join(state.action_history + [""]))
            .replace("<init_state>", utils.extract_init_state(self.example))
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
        )
        outputs = self.base_model.generate(
            [prompts],
            SamplingParams(temperature=self.temperature, max_tokens=20, top_p=0.95)
        )
        actions = [output.outputs[0].text.strip().split("\n")[0] for output in outputs]
        return list(dict.fromkeys(actions))  # Deduplicate actions

    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        intuition_reward = len(action)  # Example heuristic: action length as proxy for reward
        return intuition_reward, {"intuition": intuition_reward}

# Run Tree-of-Thought
world_model = BlocksWorldModelToT(base_model=llm, prompt=prompt)
config = BWConfigToT(base_model=llm, prompt=prompt)
algorithm = BeamSearch(beam_size=4, max_depth=7)
reasoner_tot = Reasoner(world_model, config, algorithm)
result_tot = reasoner_tot(example)

print("Tree-of-Thought Result:", result_tot)

# Define classes for RAP
class BWStateRAP(NamedTuple):
    step_idx: int
    last_blocks_state: str
    blocks_state: str
    buffered_action: BWAction

class BlocksWorldModelRAP(WorldModel):
    def __init__(self, base_model: LLM, prompt: dict, max_steps: int = 4) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt

    def init_state(self) -> BWStateRAP:
        return BWStateRAP(step_idx=0, last_blocks_state="", blocks_state=utils.extract_init_state(self.example), buffered_action="")

    def step(self, state: BWStateRAP, action: BWAction) -> tuple[BWStateRAP, dict]:
        state = copy.deepcopy(state)
        blocks_state = self.update_blocks(state.blocks_state, action)
        new_buffered_action = action if state.buffered_action == "" else ""
        state = BWStateRAP(
            step_idx=state.step_idx + 1,
            last_blocks_state=state.blocks_state,
            blocks_state=blocks_state,
            buffered_action=new_buffered_action
        )
        return state, {"goal_reached": utils.goal_check(utils.extract_goals(self.example), blocks_state)}

    def update_blocks(self, block_states: str, action: BWAction) -> str:
        if "pick" in action:
            key = "world_update_pickup"
        elif "unstack" in action:
            key = "world_update_unstack"
        elif "put" in action:
            key = "world_update_putdown"
        elif "stack" in action:
            key = "world_update_stack"
        else:
            raise ValueError("Invalid action")
        world_update_prompt = self.prompt[key].format(block_states, action.capitalize() + ".")
        world_output = self.base_model.generate(
            [world_update_prompt],
            SamplingParams(max_tokens=50, top_p=0.95, temperature=0)
        )
        return utils.apply_change(world_output[0].outputs[0].text.strip(), block_states)

    def is_terminal(self, state: BWStateRAP) -> bool:
        if utils.goal_check(utils.extract_goals(self.example), state.blocks_state)[0]:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False

class BWConfigRAP(SearchConfig):
    def __init__(self, base_model: LLM, prompt: dict) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt

    def get_actions(self, state: BWStateRAP) -> list[BWAction]:
        return utils.generate_all_actions(state.blocks_state)

    def reward(self, state: BWStateRAP, action: BWAction, **kwargs) -> tuple[float, dict]:
        reward_value = len(action)  # Example heuristic
        return reward_value, {"reward": reward_value}

# Run RAP
world_model = BlocksWorldModelRAP(base_model=llm, prompt=prompt, max_steps=4)
config = BWConfigRAP(base_model=llm, prompt=prompt)
algorithm = MCTS(depth_limit=4, disable_tqdm=False, output_trace_in_each_iter=True, n_iters=10)
reasoner_rap = Reasoner(world_model, config, algorithm)
result_rap = reasoner_rap(example)

print("RAP Result:", result_rap)
result_rap.trace
# Finally, we get a valid solution!
# ## Visualization
# Visualization is as simple as calling `visualize(log)`

from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData, EdgeData
from reasoners.algorithm.mcts import MCTSNode


# (Optional) You can write node_data_factory and edge_data_factory to show customized information.
# def blocksworld_node_data_factory(n: MCTSNode) -> NodeData:
#     return NodeData({"block state": n.state.blocks_state if n.state else "Not expanded",
#                      "# goals satisfied": n.reward_details["goal_reached"][1] if hasattr(n, "reward_details") else "N/A",
#                      "# visited": len(n.cum_rewards)})

# def blocksworld_edge_data_factory(n: MCTSNode) -> EdgeData:
#     return EdgeData({"Q": n.Q,
#                      "intuition": n.fast_reward_details["intuition"],
#                      "self_eval": n.fast_reward_details["self_eval"],
#                      "action": n.action})

# visualize(result_rap,
#           node_data_factory=blocksworld_node_data_factory,
#           edge_data_factory=blocksworld_edge_data_factory)
# This evaluator module provides standard APIs and easy implementation of multiple popular reasoning datasets.

# a helper function to extract the action history from the output of the algorithm

# Helper function to extract the action history from the output of the algorithm
def bfs_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # To ensure the plan is saved before evaluation in a multi-process setting
    try:
        return "\n".join(algo_output.terminal_node.state.action_history)
    except Exception as e:
        print("Error in output extraction:", e)
        return ""

# Reload prompts for evaluation
with open('examples/CoT/blocksworld/prompts/pool_prompt_v1.json') as f:
    prompt = json.load(f)

# Initialize evaluator
evaluator = BWEvaluator(
    config_file='examples/CoT/blocksworld/data/bw_config.yaml',
    domain_file='examples/CoT/blocksworld/data/generated_domain.pddl',
    data_path='examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json',
    init_prompt=prompt,
    output_extractor=bfs_bw_extractor
)

# Perform evaluation
evaluator.evaluate(reasoner_tot, shuffle_prompt=True, num_shot=4, resume=0)
