Implementar árbol MCTS con PUCT.

Estructura nodo:

state
parent
children
N
W
Q
P

Algoritmo:

selection
expansion
simulation
backpropagation

search(state, env):

ejecutar iteraciones de MCTS

salida:

policy
visitas normalizadas por acción

value
valor estimado del estado

permitir múltiples instancias para ejecución paralela.
