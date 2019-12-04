### Policy based methods
# Continuous control


### 1. Approach

I started out from the code for the ddpg-pendulum environment solutoin from the [udacity deep-reinforcement-learning repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).
In the first steps I modified the environment specific veriables.

DDPG


### 2. Environments solved

#### 1. Reacher with __one agent__

	- added batch normalization to the model NN (both actor and critic class) got improvement but score still <3 (has to be >30 for 100 epochs)

	- this piece of code is needed after the implementation of the batch normalization in order to overcome array dimensionality error.
	```python
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
	```

	- added epsilon nose multiplier with a decaying factor

	- ![Reacher One arm](assets/reacher_one_arm.gif)


  - another solution is to use a smaller number of nodes in the network layers. I used values of 256 and 256 for both the actor and critic methods. The parameter of `max_t` should be more 1000 or more.

#### 2. Reacher with __twenty agents__

The implementation of the agent and the model are the same as in the case of the one arm agent. The only difference is in the ddpg method.

  - ![Reacher one arm](assets/reacher_20_arm.gif)



__Note__

In the beginning of the project I started working inside the Udacity Workspace environment. However, I noticed that restarting of the notebook takes too much time and toggling the GPU does not restart the environment of the workspace. I ended up spending a few days of not knowing where the problem lies.

After I installed the environment on local computer with Nvidia 1050, the code implementation worked perfectly. I tested the code on Nvidia RTX 2080Ti as well. __In the case you  want to run the code on RTX cards, you should remove the pytorch 0.4 version that comes with the install of the [deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning) repo, and simply install the latest pytorch version.__ Prior to the reinstall of pytorch I encountered a problem, where the environment just did not want to run.
