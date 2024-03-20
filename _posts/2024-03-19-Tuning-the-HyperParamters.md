---
title: '2024-03-19 Tuning the HyperParamters'
date: 2024-03-19
permalink: /posts/2024/03/Tuning-the-HyperParamters/
tags:
  - rf reg tuning
  - rf clf result
  - hyperparameter turning 
---
# Random Forest Hyperparameters we’ll be Looking at:
- max_depth
- min_sample_split
- max_leaf_nodes
- min_samples_leaf
- n_estimators
- max_sample (bootstrap sample)
- max_features
## Random Forest Hyperparameter #1: max_depth
Let’s discuss the critical max_depth hyperparameter first. The max_depth of a tree in Random Forest is defined as the longest path between the root node and the leaf node:

![depth](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/depth.png)
Using the max_depth parameter, I can limit up to what depth I want every tree in my random forest to grow.
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/MaximumDepth.png)
in this graph, we can clearly see that as the max depth of the decision tree increases, the performance of the model over the training set increases continuously. upto 30 level, there is no improvement. The is because the feature of 2^30 is enough to hold 200 feature to split. 
Can you think of a reason for this? The tree starts to overfit the training set and therefore is not able to improve even more depth is given. 
Among the parameters of a decision tree, max_depth works on the macro level by greatly reducing the growth of the Decision Tree.

## Random Forest Hyperparameter #2: min_sample_split
min_sample_split – a parameter that tells the decision tree in a random forest the minimum required number of observations in any given node in order to split it.
The default value of the minimum_sample_split is assigned to 2. This means that if any terminal node has more than two observations and is not a pure node, we can split it further into subnodes.
Having a default value as 2 poses the issue that a tree often keeps on splitting until the nodes are completely pure. As a result, the tree grows in size and therefore overfits the data.
![sampleslipt](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/split.webp)
By increasing the value of the min_sample_split, we can reduce the number of splits that happen in the decision tree and therefore prevent the model from overfitting. In the above example, if we increase the min_sample_split value from 2 to 6, the tree on the left would then look like the tree on the right.
Now, let’s look at the effect of min_samples_split on the performance of the model. The graph below is plotted considering that all the other parameters remain the same and only the value of min_samples_split is changed:
On increasing the value of the min_sample_split hyperparameter, we can clearly see that for the small value of parameters, there is a significant difference between the training score and the test scores. But as the value of the parameter increases, the difference between the train score and the test score decreases.
But there’s one thing you should keep in mind. When the parameter value increases too much, there is an overall dip in both the training score and test scores. This is due to the fact that the minimum requirement of splitting a node is so high that there are no significant splits observed. As a result, the random forest starts to underfit.
You can read more about the concept of overfitting and underfitting here:
[Underfitting vs. Overfitting in Machine Learning](https://www.analyticsvidhya.com/blog/2020/02/underfitting-overfitting-best-fitting-machine-learning/?utm_source=blog&utm_medium=beginners-guide-random-forest-hyperparameter-tuning)
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/MinimumSampleSplit.png)

## Random Forest Hyperparameter #3: max_terminal_nodes
Next, let’s move on to another Random Forest hyperparameter called max_leaf_nodes. This hyperparameter sets a condition on the splitting of the nodes in the tree and hence restricts the growth of the tree. If after splitting we have more terminal nodes than the specified number of terminal nodes, it will stop the splitting and the tree will not grow further.
Let’s say we set the maximum terminal nodes as 2 in this case. As there is only one node, it will allow the tree to grow further:
![leafnote](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/leafnode.webp)
Now, after the first split, you can see that there are 2 nodes here and we have set the maximum terminal nodes as 2. Hence, the tree will terminate here and will not grow further. This is how setting the maximum terminal nodes or max_leaf_nodes can help us in preventing overfitting.
Note that if the value of the max_leaf_nodes is very small, the random forest is likely to underfit. Let’s see how this parameter affects the random forest model’s performance:
We can see that when the parameter value is very small, the tree is underfitting and as the parameter value increases, the performance of the tree over both test and train increases. According to this plot, the tree starts to overfit as the parameter value goes beyond 25.
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/max_leaf_nodes.png)



![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/MinimumSampleLeaf.png)
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/maximumsamples.png)
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/Minimum/TickerOwnedbyETFRanking.png)


### read this reference [Tuning the parameters of your Random Forest model](https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/)
