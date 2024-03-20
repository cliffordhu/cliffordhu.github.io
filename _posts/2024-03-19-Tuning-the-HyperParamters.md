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
![depth](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/MaximumDepth.png)
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/MaximumDepth.png)
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/MinimumSampleLeaf.png)
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/MinimumSampleSplit.png)
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/max_leaf_nodes.png)
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/maximumsamples.png)
![Maximum](https://raw.githubusercontent.com/cliffordhu/cliffordhu.github.io/master/_posts/Image-30-19-24/Minimum/TickerOwnedbyETFRanking.png)


### read this reference [Tuning the parameters of your Random Forest model](https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/)
