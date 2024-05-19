# Training

## Playground
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

We now try to modify the trainning learning rate since training is taking very long, loss curve looks almost linear. 

Test 6: 

```bash
python train.py --exp=addition_experiment --train --num_steps=10 --dataset=addition --train --cuda --infinite --pretrain --batch_size=10000 --step_lr 10000 --lmbda 1 --logdir test_6 --lr 0.001
```
Yes, we get faster convergence. After 6-8 mins, we plateu at a test error of 0.439, which for some reason is a lot worse than the test error which is continuall going down past 0.205 in 5. The test plateu in this experiment might be due to the new learning rate (10x larger) not allowing it to converge properly. Massive jumps are seen in the energy loss plot for lr = 0.001.

TODO: best to look at a learning rate between 1e-3 and 1e-4. Also looking and some learning rate decay shedule. 

Test 7: 

Same as before but we used AdamW to regularise our weights. we set the value to 1e-4, hopefully not constraining the weights too much. 
```bash
python train.py --exp=addition_experiment --train --num_steps=10 --dataset=addition --train --cuda --infinite --pretrain --batch_size=10000 --step_lr 10000 --lmbda 1 --logdir test_7 --lr 0.001
```

Note we can remake commands

## lr shedule

From the playground, I know that we need a leardning rate around 1e-3 and 1e-4. No we implement a learning reate scheduler for this.

for example below, --exp is the name of the sub folder
```python train.py --train --num_steps=10 --dataset=addition --train --cuda --infinite --pretrain --batch_size=10000 --step_lr 10000 --lmbda 1 --logdir results/lr --lr 0.0005 --exp lr_0005```

Best so far all in lr folder. 

```bash
python train.py --train --num_steps=10 --dataset=addition --train --cuda --infinite --pretrain --batch_size=10000 --step_lr 10000 --lmbda 1 --logdir results/lr --lr 0.001 --exp exp_001
```

1. lr=0.0001: Clear best; errors around 0.14
2. Exp schedule, lr=0.001: error around 0.57 `python train.py --train --num_steps=10 --dataset=addition --train --cuda --infinite --pretrain --batch_size=10000 --step_lr 10000 --lmbda 1 --logdir results/lr --lr 0.001 --exp exp_001` converges a lot faster. However, might be over fitting? loss plateus at same level as lr=0.0001, however, test error is a lot worse. or 

### Future

- Try a cosine restard schedule and see if it can find better and better minima; and
- Check if overfitting is happending, in which case we might need to regularise the model; also
- Hyper paramter search might be useful;
- Why is it that we're able to minimise the energy landscape but it leads to a worse test error; again overfitting; add way of visualising the energy landscape decent through graphs;
- Evaluate errors using MSE instead of MAE default



# Testing

The same command except that we remove the `--train` flag
```bash
python train.py --exp=addition_experiment --num_steps=10 --dataset=addition --cuda --infinite --pretrain --batch_size=10000 --step_lr 10 --lmbda 4 --logdir random_checkpoint_works_well --decent_steps 80
```