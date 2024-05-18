## Training

```bash
python train.py --exp=addition_experiment --train --num_steps=10 --dataset=addition --train --cuda --infinite --pretrain --batch_size=10000 --step_lr 10 --lmbda 4 --logdir test
```
Note: larger batch sizes seem essential for propper running (6000+?)
Note: hyper paramter of 10 seemed to work
Note: it seems like if the energy is allowed to drop significantly, the model gets bad

step_lr (without mean):
- 0.9 divergaes
- 0.09 converges at step 2
- 0.009 convergest too slow 

Test 2:

```bash
python train.py --exp=addition_experiment --train --num_steps=10 --dataset=addition --train --cuda --infinite --pretrain --batch_size=10000 --step_lr 5 --
lmbda 2 --logdir test_2
```

Test 4: Finally Worked!!!! 

```bash
python train.py --exp=addition_experiment --train --num_steps=10 --dataset=addition --train --cuda --infinite --pretrain --batch_size=10000 --step_lr 5 --lmbda 1 --logdir test_4
```

Also looked like it converged super slowly but in reality that's not the case if we set `--step_lr 10000` to some decent number 

Now we try using the new learning rate:

Test 5:

```bash
python train.py --exp=addition_experiment --train --num_steps=10 --dataset=addition --train --cuda --infinite --pretrain --batch_size=10000 --step_lr 10000 --lmbda 1 --logdir test_5
```

## Testing

```bash
python train.py --exp=addition_experiment --num_steps=10 --dataset=addition --cuda --infinite --pretrain --batch_size=10000 --step_lr 10 --lmbda 4 --logdir random_checkpoint_works_well --decent_steps 80
```