# Isaac Gym R1StandingEnv Curriculum Progresivo - Plan

## TL;DR
- Implement quadratic penalization for leg opening, bilateral co-activation, and left-right action synchronization.
- Introduce a progressive curriculum with dynamic joint indexing, controlled by explicit thresholds and safe progression rules.
- Keep network architecture, observation space, and action space unchanged; only environment and reward logic are modified.

## Wave 1 (already defined)
### Penalizaciones centrales
- [A1] Penalización de apertura de piernas (cuadrática)
- [A2] Penalización de co-activación bilateral (caderas)
- [A3] Penalización de sincronización de acciones izquierda/derecha

## Wave 2: Curriculum Progresivo e integración
### B1: Config y estado iniciales para currículo
- What to do: Añadir en cfg los indicadores de currículo y en init el estado inicial (curriculum_level, max_curriculum_level, buffers).
- Acceptance Criteria:
  - Variables de configuración disponibles (use_curriculum, curriculum_max_level, curriculum_success_threshold, curriculum_min_episodes, curriculum_min_reward).
  - Curriculum inicia en nivel 0 para todos los envs.
- QA Scenarios:
  - Verificar lectura de cfg y estado inicial tras reset completo
- Evidence: evidence-task-B1-curriculum-init.json
- Dependencies: A1, A2, A3

### B2: Métodos _update_curriculum_parameters y _update_curriculum
- What to do: Implementar la lógica de mapeo de niveles a push_mode y current_push_force, y la lógica de actualización basada en ventanas de 200 episodios y umbrales 0.85 y 5.0.
- Acceptance Criteria:
  - Up/Down de nivel correcto sin descensos por caídas aisladas.
  - current_push_force y push_mode actualizados por nivel.
- QA Scenarios:
  - Simular acumulación de suficientes episodios con mean_success > 0.85 y mean_reward > 5.0 para subir de nivel.
- Evidence: evidence-task-B2-curriculum-logic.json
- Dependencies: B1

### B3: Integrar _apply_random_pushes con current_push_force
- What to do: Usar current_push_force para generar force_mag y aplicar direcciones según push_mode.
- Acceptance Criteria:
  - Magnitud de empuje crece con nivel según push_scales; push_mode aplicado correctamente.
- QA Scenarios:
  - Nivel 2 (lateral): verificar magnitud y dirección.
- Evidence: evidence-task-B3-push.json
- Dependencies: B2

### B4: Actualizar _reset_idx para currículum solo en resets completos de batch
- What to do: Llamar _update_curriculum(env_ids) solo si len(env_ids) == num_envs.
- Acceptance Criteria:
  - Reset parciales no provocan actualización de currículo.
- QA Scenarios:
  - Reset parcial simulado para subconjunto y verificación de no cambio de nivel.
- Evidence: evidence-task-B4-reset.json
- Dependencies: B3

## Wave 3: Logging, QA y verificación
### C1: Registro de evidencias de QA
- What to do: Registrar evidencias en .sisyphus/evidence para cada tarea (JSON con level, mean_success, mean_reward, timestamp).
- Evidence: evidence-task-C1-evidence.json
- Dependencies: B4

### C2: Pruebas mínimas de regresión (unit tests)
- What to do: Crear tests para penalización cuadrática, co-activación y sincronización.
- Evidence: tests/results-C2.json
- Dependencies: C1

### C3: QA de currículo (escenarios) y evidencia
- What to do: Escenarios de QA para curriculum: subida de nivel sostenida, no descenso por caída aislada, ventana de 200 episodios.
- Evidence: evidence-task-C3-curriculum.json
- Dependencies: C2

## Final Verification Wave
- F1: Plan compliance audit (oracle)
- F2: Code quality review (lint/tsc/tests)
- F3: Real manual QA (scenarios) + evidence
- F4: Scope fidelity check (diffs vs plan)

## Commit Strategy
- Un único plan: este archivo .md; los resultados de QA y evidence se generan fuera del plan en .sisyphus/evidence.

## Success Criteria
- Penalizaciones implementadas y validadas mediante QA y pruebas unitarias.
- Curriculum progresivo funcionando y sin creep de alcance.
- Sin cambios en red, observation space o action space.

## Notas
- Este archivo está diseñado para ser la fuente única de verdad del plan de implementación; se irán llenando las Edits con cada batch de tareas.
