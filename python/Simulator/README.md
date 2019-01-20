## use skip-gram method to generate word2vec result

>This project is part of RL-for-radio project, which aims to promote the
>recommend success ratio by utilizing RL.

To model the state in MDP, we are trying to make use of the name of albums
(maybe together with the anchor name and category) being recommended to user.
An easy way is to generate the VSM presentation for the names.

Then the DQN will try to learn the relationship between a specific name and the corresponding reward.

Here is the TSNE result for one training.

![image](https://github.com/ice871117/RL-for-radio/blob/master/python/Simulator/images/tsne.png)