
### 

## Policy based methods 
# Continuous control 


Environment 

Reacher with one agent 

- started from the code of the ddpg-	pendulum environment form the official udacity repo
- modified the envoronment setup from unity ot udacity 
- run the training first bad resutls 
- added batch normalization to the model NN (both actor and critic class) got improvement but score still <3 (has to be >30 for 100 epochs) 

- this piece of code is needed after the implementation of the batch normalization in oder to overcome array dimensionality error.
```python
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
```

- added epsilon nose multiplier with a decaying factor 
