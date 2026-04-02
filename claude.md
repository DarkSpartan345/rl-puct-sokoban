# Project

Implementar el framework definido en:

docs/goal.md
specs/architecture.md
specs/interfaces.md
specs/coding_rules.md

Los módulos están especificados en:

modules/

Implementar código en:

src/


# Workflow

Implementar módulos en este orden:

1 mcts
2 agents
3 selfplay
4 training
5 envs


# File Mapping

modules/mcts.md → src/mcts/
modules/agents.md → src/agents/
modules/selfplay.md → src/selfplay/
modules/training.md → src/training/

env wrappers → src/envs/


# Model Strategy

Arquitectura / planificación
→ Claude Sonnet

Implementación de código
→ Claude Opus

Refactorización
→ Claude Sonnet

Cambios pequeños
→ Claude Haiku


# Prompt Strategy

Trabajar por módulo.

Formato de solicitud:

Read:
modules/<module>.md
specs/interfaces.md

Implement:
src/<module>/