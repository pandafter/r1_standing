# Retrioalimentaacion Entrenamineyto 1
- El robot se mantiene estable pero en una posicino antinatural
- Haré super estricto el hecho que tiene que mantener esa postura siempre, intentar corregir siempre para estar en esa posicion, si no, es un penalty
- En la iteracion 1000 el robot ya mantenia su iteración sin ningun tipo de problema
- podemos decir qu ela estapa de estabilizacion 
- necesitamos los 2 tipos de politicas, la politica generalista y la especialista para afinar las demas acciones que tenemos


## despues de la invstigacion con NotebookML

✅ Se apliquen empujones aleatorios
✅ Se permita stepping
✅ Se recompense recuperación
✅ Se deje de forzar rigidez
✅ Emerja estrategia tobillo → cadera → paso

Tambien cambio el numero de iteraciones para este y el save interval porque el push-recovery necesita un entrenamiento mas pesado:

- vamos a utilizar entonces 15.000 iteraciones y el save interval cambia a 1000



## Despues de las investigaciones
Luego de hacer las investigaciones en las diferentes fuentes de articulos, se vió un cambio significativo, donde el robot va aprendeiendo de manera dinamica a traves del tiempo camina para no dejarse caer, aun entonces estaremos corrigiendo el cambio de esto

C:\space_r1\IsaacLab\logs\rsl_rl\r1_standing\2026-03-03_15-35-21

esta es la direccion de la politica que funcionó para una estabilizacion aunque extraña pero sin caidas

Heirarchical Policy
High-level Policys
Para orquestar las politicas de estbilidad, locomocion y YOLO para reonomiento de los espacio y la interaccion con los mismos

-que sea capas de detenerse y con el espacio evitar cualquier tipo de contacto no deseado o golpes no deseados

Arquitectura Recomendada Para Tu Caso

Dado que ya tienes políticas entrenadas por separado:

🥇 Mejor enfoque práctico:

Estabilidad → siempre activa (low-level controller)

Locomoción → recibe comandos de dirección

YOLO → genera objetivo

Capa de decisión → FSM o Behavior Tree

Arquitectura real:

YOLO → Target Vector
        ↓
High-Level Planner
        ↓
Locomotion Policy
        ↓
Stability Controller
        ↓
Actuadores



# NOTAS IMPORTANTES
-Integrar el push and learn para aplicar los pequeños empujones para perfeccionar la posicion
- premiar el mantener la posicion inicial del robot



