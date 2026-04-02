Implementar PUCT-MCTS para Sokoban usando Gymnasium.

Usar MCTS para generar decisiones y construir dataset mediante self-play.

Dataset:

(state, policy, value)

policy:
visitas normalizadas del árbol.

value:
retorno final del episodio.

Proceso:

1 ejecutar búsqueda MCTS en cada estado  
2 obtener policy desde visitas  
3 seleccionar acción desde policy  
4 guardar (state, policy)  
5 terminar episodio  
6 calcular retorno  
7 asignar value a todos los estados  

Ejecutar múltiples episodios en paralelo.

Entorno:

gymnasium_sokoban  
SmallRoom
