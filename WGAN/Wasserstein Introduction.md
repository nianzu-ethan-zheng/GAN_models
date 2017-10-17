# Wasserstein Distance Introduction
-----------------------------------------------------------------------
## Before Reading
Wasserstein Distance have been proved very efficient. And it gives us deeply insight into Generative adversarial nets.And there are some reading material for who need to read deeply:
* [Read-through: Wasserstein GAN](http://www.alexirpan.com/2017/02/22/wasserstein-gan.html)
* [Wasserstein GAN and the Kantorovich-Rubinstein Duality](https://vincentherrmann.github.io/blog/wasserstein/)
* [Goodfollow's assessment](https://www.quora.com/What-is-new-with-Wasserstein-GAN)

## Different Distance
* The Total Variation distance(TV) 
* The kullback-leibler distance(KL)
* The Jenson-shannon distance(JS)
* The Earth Mover Distance(EMD)

## A Simple Example
Consider Probability distribution difined over R<sup>2</sup>,Let the true data distribution be (0,y),with y sample uniformly from U[0,1].Coinsider the family of distribution P<sub>r</sub> = (x,y). It can be very confused that what the true data disctribution like.So I draw the following distribution to facilate understanding:

![Data_distribtuion](https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/datadistribution.jpg)


##  The Earth Mover Defination
Unfortunately,computing EMD exactly is intractable. It's a optimization problem,which need a lot of computation and time comsumption.but Wasserstein GAN propose a method,showing how we can compute an approximation of this.

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f1.png" height="60">

subject to :

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f21.png" height="40">

It can be rewritten as following:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f2.png" height="50">
<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/EMD.png" height="500">

and let:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f3.png" height="70"><img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f4.png" height="70">

We can get the objectives and constrains:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f6.png" height="120">

The A come from here:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f5.png" height="400">

and the dual form can be simply proved as:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f7.png" height="60">

and the formula is weaker Duality.Yeah! we have strong distance.We first discuss the  Farkas Theorem

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f8.png" height="400">

Exactly one of the following statement is true:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f9.png" height="90">

Back to our question,we can assume:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f10.png" height="70">

we can get strong duality:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f11.png" height="60">

In our case, where we use Euclidian distances, these slope limits are -1 and 1. We call this constraint Lipschitz continuity (with Lipschitz constant 1) and write:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f12.png" height="70">

ANd This is true because every K-Lipschitz function is a 1-Lipschitz function if you divide it by K, and the Wasserstein objective is linear . Further suppose these functions are all K-Lipschitz for some K, Then we have:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f13.png" height="100">

The full algorithm is below;

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/algorithm.png" height="400">

And the gradient is like that:

<img src="https://github.com/DreamPurchaseZnz/GAN_models/blob/master/WGAN/Picture/f15.png" height="400">







