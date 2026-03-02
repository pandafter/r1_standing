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