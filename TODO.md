# TODO

- [ ] Llegir el paper sobre [DDPG](https://arxiv.org/abs/1509.02971) i entendre'l. 
    - [ ] focalitzar a l'algoritme i els detalls de l'experiment
- [ ] Estudiar l'exercici de la lliçó on hem aplicat el DDPG a una tasca de l'OpenAI Gym. 
    - [ ] entendre el detall
    - [ ] retocar els diferents hiperparàmetres i ajustos per veure que pot funcionar (i què no)
- [ ] Adaptar el codi de la lliçó al projecte
    - [ ] copiar el codi al directory de treball
    - [ ] Adaptar el codi, intentant fer les mínimes modificacions
    - [ ] Assegurar-se que el codi funciona (no importa la velocitat)
    - [ ] no cal preocupar-se de modificar els hiperparàmetres, optimitzadors, ... encara
    - [ ] No cal executar el codi a la GPU, això millor al següent pas
- [ ] Optimitzar els hiperparàmetres
    - [ ] Després de verificar que el codi DDPG funciona, intentar unes sessions d'entrenament a CPU llargues.
    - [ ] Si l'agent no apren, provar algunes posibles solucions modificant el codi.
    - [ ] Una vegada fet i estant segur, executar el codi amb GPU
- [ ] Continuar explorant.
    - [ ] Aquest [paper](https://arxiv.org/abs/1604.06778) evalua diversos algoritmes RL en tasques de control continu.  
    >Introdueix REINFORCE, TNPG, RWR, REPS, TRPO, CEM, CMA-ES i DDPG i fa suggerències sobre quin podria ser el més adequat pel projecte
    
### Requeriments
- [x] Creem un nou entorn conda: drlnd_cc (continuous control)
    - [x] Unity ml-agents v0.4 
    - [x] numpy
    - [x] pytorch v0.4.0

### L'entorn
Treballarem sobre l'entorn [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

- *recompensa*: +0.1 per cada pas que la ma de l'agent està a l'objectiu. Per tant, l'objectiu de l'agent es mantenir la seva posició a la localització de l'objectiu el màxim temps possible.
- *observation space*: 33 variables que es corresponen amb la posició, rotació, velocitat i velocitat angular del braç. 
- *action* (acció): cada acció es un vector de 4 números, corresponent a la torsió (torque) aplicable a dues juntes. Cada entrada en el vector ha de ser un número entre -1 i 1

### Entrenament distribuit
Pel projecte, hi ha 2 versions separades de l'entorn:
- la primera conté un agent
- la segona versió conté 20 agents idèntics, cadascú d'ells amb la seva còpia de l'entorn

La segona versió es útil per algoritmes com [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf) i [D4PG](https://openreview.net/pdf?id=SyZipzbCb) que usen múltiples (sense interacció, paral·lels) còpies del mateix agent per distribuir la tasca d'agafar experiència.

### Resoldre l'entorn
Només necessitem resoldre una de les dues versions
**Opció 1: Resoldre la primera versió**
La tasca es episòdica, i per resoldre l'entorn, l'agent necessita una mitjade de **+30** en **100** episodis consecutius
**Opció 2: Resoldre la segona versió**
L'objectiu es una mica diferent, per tenir en compte la presència de molts agents. En particular, els agents han d'aconseguir una mitja de **+30** en **100** episodis consecutius, en tots els agents. Especificament:
- Després de cada episodi, sumem les recompenses de cada agent (sense descompte), per tenir el resultat de cada agent. Aleshores, fem la mitja d'aquests 20 resultats
- Això produeix la mitja de resultats per cada episodi.


### L'entorn
1. **Activar l'entorn**
