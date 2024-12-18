# LLM-Reasoners Demo
# 
# This notebook is accompanied with our tutorial at SIGIR VF:
# [[slides](https://www.llm-reasoners.net/2024-02-Reasoners-SIGIR.pdf)]
# [[video](https://www.youtube.com/watch?v=d_x2pzEHGQY&pp=ygUJc2hpYm8gaGFv) (starting at 37:20)]
# 
# ## Setup
# 
# The following code assumes you have cloned our library with `git clone https://github.com/maitrix-org/llm-reasoners.git --recursive`
# Set cuda device and initialize an ExllamaModel use our unified LLM interface.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VAL'] = "/home/guoguo/llm-reasoners/LLMs-Planning/planner_tools/VAL"
from reasoners.lm import ExLlamaModel
import torch

# https://huggingface.co/TheBloke/Llama-2-70B-GPTQ
# It may take a few minutes to download the model

# model = ExLlamaModel(model_dir='TheBloke/LLaMA-Pro-8B-Instruct-GPTQ',
model = ExLlamaModel(model_dir='meta-llama/Llama-3.1-8B-Instruct',
                     lora_dir=None,
                     device = torch.device("cuda:0"),
                     max_batch_size=1,
                     max_new_tokens=200,
                     mem_map=None, # For 2 * 24GB GPUs. If you have > 40GB you can set it to None
                     max_seq_length=2048)

# Or use any other model providers:

# HFModel(llama_path, llama_path, device=device, max_batch_size=1, max_new_tokens=512, quantized=quantized, peft_pth=peft_path, load_awq_pth=load_awq_pth)
# Llama3Model(llama2_ckpts, llama_size, max_batch_size=1)
# OpenAIModel(openai_mode)
# ClaudeModel('claude-3-opus-20240229')
# We gather one example from the Blocksworld dataset, and the proper prompt for in-context learning examples.
# We will talk more about Evaluators later.

from reasoners.benchmark import BWEvaluator
import json

with open('examples/CoT/blocksworld/prompts/pool_prompt_v1.json') as f:
    prompt = json.load(f)
evaluator = BWEvaluator(config_file='examples/CoT/blocksworld/data/bw_config.yaml',
                        domain_file='examples/CoT/blocksworld/data/generated_domain.pddl',
                        data_path='examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json',
                        init_prompt=prompt)
prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=4)
example = evaluator.full_dataset[1]
cot_inputs = (prompt['icl'].replace('<init_state>', example["init"])
                           .replace('<goals>', example["goal"])
                           .replace('<action>', ''))
# Here is the example.

print(example['init'])
print(example['goal'])
# ## Chain-of-Thought
# We first experiment with the Chain-of-Thought method.
# Since we are having the simplest generation algorithm, we directly ask the model to generate all the steps.
# We look at the 4-shot prompt and the generated answer.

print(cot_inputs)

output = model.generate([cot_inputs],
                        hide_input=True,
                        eos_token_id='\n[').text[0][:-1].strip()

print(output)
# Clearly that's not a valid solution :( 
# The orange block is on the red block, so we cannot pick up the red block as the first step.
# ## Tree-of-Thought
# Then let's turn to a tree search algorithm, [Tree-of-Thought]((https://arxiv.org/abs/2305.10601)).
# We will need to define a simple world model, and a search algorithm, for the Blocksworld task.

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
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 4,
                 batch_size: int = 1) -> None:
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
            state = BWStateToT(step_idx=state.step_idx + 1, action_history=state.action_history + [action], end=False)
        else:
            state = BWStateToT(step_idx=state.step_idx + 1, action_history=state.action_history, end=True)
        return state, {}  # the dict is auxiliary information for SearchConfig, we don't need it here.
    
    def is_terminal(self, state: State) -> bool:
        return state.end or state.step_idx >= self.max_steps


class BWConfigToT(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 temperature: float = 0.8,
                 n_candidate: int = 4) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature

    def get_actions(self, state: BWStateToT) -> list[BWAction]:
        prompts = (self.prompt["icl"]
                       .replace("<action>", "\n".join(state.action_history + [""]))
                       .replace("<init_state>", utils.extract_init_state(self.example))
                       .replace("<goals>", utils.extract_goals(self.example, return_raw=True)))
        outputs = self.base_model.generate([prompts],
                                           num_return_sequences=self.n_candidate,
                                           max_new_tokens=20,
                                           eos_token_id="\n",
                                           temperature=self.temperature,
                                           do_sample=True,
                                           hide_input=True).text
        outputs = [output.split("\n")[0] for output in outputs]
        outputs = list(dict.fromkeys(outputs))  # deduplicate
        return outputs

    # Some reward functions are fast to calculate.
    # We calculate the reward before executing the action, which can be used to better guide the search.
    def fast_reward(self, state: BWStateToT, action: BWAction) -> tuple[float, dict]:
        # We use two rewards here:
        # 1. Intuition: The loglikelihood of the action given the prompt.
        # 2. Self-eval: Ask the language model whether this step is "Good".
        inputs = self.prompt["icl"].replace("<action>", "\n".join(state.action_history + [""])) \
            .replace("<init_state>", utils.extract_init_state(self.example)) \
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True))[:-1]
        
        intuition = self.base_model.get_loglikelihood(inputs, [inputs + "\n" + action])[0]

        self_eval_prompt = (self.prompt["self-eval"].replace("<init_state>", utils.extract_init_state(self.example))
                                                    .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
                                                    .replace("<action>", action))
        self_eval = self.base_model.get_loglikelihood(self_eval_prompt, [self_eval_prompt + "good"])[0]

        return intuition + self_eval, {'intuition': intuition, "self_eval": self_eval}
    
    # kwargs is the auxiliary information returned by SearchConfig.fast_reward and WorldModel.step,
    # so that we do not need duplicated calculations.
    # In this case, we just use the fast_reward result as the reward.
    # Generally, if a reward function depends on the new state, or is slow to calculate,
    # we will calculate it here.
    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        return kwargs['intuition'] + kwargs['self_eval'], kwargs
# Note: The following command may take to 2 minutes to run

world_model = BlocksWorldModelToT(base_model=model, prompt=prompt)
config = BWConfigToT(base_model=model, prompt=prompt)
algorithm = BeamSearch(beam_size=4, max_depth=7)
reasoner_tot = Reasoner(world_model=world_model, search_config=config, search_algo=algorithm)
result_tot = reasoner_tot(example)
print(result_tot)

print('Action, Reward')
for action, _, reward in result_tot.trace:
    print(action, reward)
# Still the same error :(
# ## RAP
# With [RAP](https://arxiv.org/abs/2305.14992), we are truly using the latest block configuration as the state, instead of a history of actions.
# Thus, we define a new world model to transit between states, which is just a little complex than the previous one.

BWAction = str


class BWStateRAP(NamedTuple):
    step_idx: int
    last_blocks_state: str
    blocks_state: str
    buffered_action: BWAction


class BlocksWorldModelRAP(WorldModel):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 4,
                 batch_size: int = 1) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> BWStateRAP:
        return BWStateRAP(step_idx=0, last_blocks_state="", blocks_state=utils.
                       extract_init_state(self.example), buffered_action="")

    def step(self, state: BWStateRAP, action: BWAction) -> tuple[BWStateRAP, dict]:
        state = copy.deepcopy(state)
        blocks_state = state.blocks_state
        step_idx = state.step_idx
        blocks_state = self.update_blocks(blocks_state, action)
        new_buffered_action = action if state.buffered_action == "" else ""

        state = BWStateRAP(step_idx=step_idx + 1,
                        last_blocks_state=state.blocks_state,
                        blocks_state=blocks_state,
                        buffered_action=new_buffered_action)
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
        world_output = self.base_model.generate([world_update_prompt],
                                                eos_token_id="\n",
                                                hide_input=True,
                                                temperature=0).text[0].strip()
        new_state = utils.apply_change(world_output, block_states)
        return new_state

    def is_terminal(self, state: BWStateRAP) -> bool:
        if utils.goal_check(utils.extract_goals(self.example), state.blocks_state)[0]:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False

class BWConfigRAP(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 batch_size: int = 1,
                 reward_alpha: float = 0.5,
                 goal_reward_default: float = 0.,
                 goal_reached_reward: float = 100.) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward

    def get_actions(self, state: BWStateRAP) -> list[BWAction]:
        blocks_state = state.blocks_state
        return utils.generate_all_actions(blocks_state)

    def fast_reward(self, state: BWStateRAP, action: BWAction) -> tuple[float, dict]:
        if state.buffered_action == "":
            current_blocks_state = state.blocks_state
        else:
            current_blocks_state = state.last_blocks_state
        previous_action = state.buffered_action + "\n" if state.buffered_action != "" else ""
        
        # every two steps, we will also reduce the icl examples by 2 steps
        # so that the distribution of step length in examples is more reasonable
        icl_template = self.prompt["icl_list"][state.step_idx // 2]
        
        inputs = (icl_template.replace("<init_state>", current_blocks_state)
                              .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
                              .replace("<action>", previous_action))
        intuition = self.base_model.get_loglikelihood(inputs, [inputs + action])[0]

        self_eval_prompt = (self.prompt["self-eval"]
                                .replace("<init_state>", current_blocks_state)
                                .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
                                .replace("<action>", action))
        self_eval = self.base_model.get_loglikelihood(self_eval_prompt, [self_eval_prompt + "good"])[0]

        return (self.calculate_reward(intuition, self_eval),
                {'intuition': intuition, "self_eval": self_eval})

    def calculate_reward(self, intuition, self_eval, goal_reached=None) -> float:
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = goal_reached[1]
        return (intuition + self_eval) * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self, state: BWStateRAP, action: BWAction,
               intuition: float = None,
               self_eval: float = None,
               goal_reached: tuple[bool, float] = None) -> tuple[float, dict]:
        return (self.calculate_reward(intuition, self_eval, goal_reached),
                {'intuition': intuition, 'goal_reached': goal_reached})
# We just use the MCTS algorithm embedded in Reasoners, and build up the pipeline again.
# Note: the following command may take 2 minutes to run

world_model = BlocksWorldModelRAP(base_model=model, prompt=prompt, max_steps=4)
config = BWConfigRAP(base_model=model, prompt=prompt)
algorithm = MCTS(depth_limit=4, disable_tqdm=False, output_trace_in_each_iter=True, n_iters=10)
reasoner_rap = Reasoner(world_model=world_model, search_config=config, search_algo=algorithm)
result_rap = reasoner_rap(example)
print(result_rap)

result_rap.trace
# Finally, we get a valid solution!
# ## Visualization
# Visualization is as simple as calling `visualize(log)`

from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData, EdgeData
from reasoners.algorithm.mcts import MCTSNode


# (Optional) You can write node_data_factory and edge_data_factory to show customized information.
def blocksworld_node_data_factory(n: MCTSNode) -> NodeData:
    return NodeData({"block state": n.state.blocks_state if n.state else "Not expanded",
                     "# goals satisfied": n.reward_details["goal_reached"][1] if hasattr(n, "reward_details") else "N/A",
                     "# visited": len(n.cum_rewards)})

def blocksworld_edge_data_factory(n: MCTSNode) -> EdgeData:
    return EdgeData({"Q": n.Q,
                     "intuition": n.fast_reward_details["intuition"],
                     "self_eval": n.fast_reward_details["self_eval"],
                     "action": n.action})

visualize(result_rap,
          node_data_factory=blocksworld_node_data_factory,
          edge_data_factory=blocksworld_edge_data_factory)
# This evaluator module provides standard APIs and easy implementation of multiple popular reasoning datasets.

# a helper function to extract the action history from the output of the algorithm

def bfs_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # to make sure the plan is saved before evaluation in multi-process setting
    try:
        return "\n".join(algo_output.terminal_node.state.action_history)
    except Exception as e:
        print("Error in output extraction,", e)
        return ""

with open('examples/CoT/blocksworld/prompts/pool_prompt_v1.json') as f:
    prompt = json.load(f)

evaluator = BWEvaluator(config_file='examples/CoT/blocksworld/data/bw_config.yaml',
                        domain_file='examples/CoT/blocksworld/data/generated_domain.pddl',
                        data_path='examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json',
                        init_prompt=prompt,
                        output_extractor=bfs_bw_extractor)

evaluator.evaluate(reasoner_tot, shuffle_prompt=True, num_shot=4, resume=0)