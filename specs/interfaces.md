Environment

reset()
input: none
output: state

step(action)
input: action:int
output: state, reward:float, terminated:bool, truncated:bool

get_actions(state)
input: state
output: list[int]


MCTS

search(state, env)
input: state, env
output: policy:list[float], value:float

update(action, next_state)
input: action:int, next_state
output: none


Agent

select_action(state)
input: state
output: action:int

get_policy(state)
input: state
output: policy:list[float]


SelfPlay

run_episode(env, agent)
input: env, agent
output: trajectory:list[(state, policy, reward)]

run_parallel(env_fn, agent_fn, episodes:int, workers:int)
input: env_fn, agent_fn, episodes, workers
output: list[trajectory]


Training

build_dataset(trajectories)
input: list[trajectory]
output: dataset

save_dataset(dataset, path)
input: dataset, path:str
output: none
