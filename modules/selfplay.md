Generar episodios Sokoban usando PUCTAgent.

Proceso:

reset env

loop:

policy = agent.get_policy(state)

action = sample(policy)

next_state, reward, terminated, truncated = env.step(action)

guardar:

(state, policy, reward)

state = next_state

al terminar episodio:

calcular retorno acumulado
generar:

(state, policy, value)

Soportar ejecución paralela de episodios.
