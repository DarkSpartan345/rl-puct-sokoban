Construir dataset desde trayectorias.

Entrada:

(state, policy, value)

Procesos:

agregar trayectorias
construir dataset

Formato:

dataset = {
states,
policies,
values
}

Guardar dataset en numpy.

Entrada puede provenir de múltiples workers.
