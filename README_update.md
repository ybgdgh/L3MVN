# Frontier Semantic Policy for Visual Target Navigation

change version between gibson and hm3d:
1. agents/sem_exp.py: 

```
class Sem_Exp_Env_Agent(ObjectGoal_Env):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

```

if gibson: ObjectGoal_Env
else: ObjectGoal_Env21

2. envs/habitat/utils/vector_env.py:

```
if hm3d: from gym.spaces.dict import Dict as SpaceDict
else: from gym.spaces.dict_space import Dict as SpaceDict
```

3. envs/_init_.py:

```
def make_vec_envs(args):
    envs = construct_envs(args)
    envs = VecPyTorch(envs, args.device)
    return envs
```

if gibson: construct_envs;
else: construct_envs21


