# Gaussian Process Regression, Explained Backwards
*Scott Hawley, July 19, 2018*

### Preface
In July 2015, I very much enjoyed seeing [Yarin Gal](https://twitter.com/yaringal)'s post ["What My Deep Network Doeesn't Know,"](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) but, busy guy that I am, I read it rather quickly, played with the demos a bit, and then moved on to things like grading student papers and working on my existing machine learning project(s).

Three years later, while visiting London, I met up with [Will Wilkerson](https://twitter.com/wil_j_wil) of QMUL in London, who's using Gaussian Processes (GPs) to synthesize audio.  And I remembered reading something about those once...

Well, I'm in Oxford all July, and guess who else is in Oxford now, and is Associate Professor of Machine Learning too -- Yarin Gal!  And since he graciously agreed to chat with me, I figured I'd best 'study up' on GPs for real...

...but although there are many excellent tutorials -- including Gal's -- that strive to (or claim to) be [gentle introductions](http://dfm.io/george/dev/tutorials/first/), and not having time to read the [seminal textbook on the subject](http://www.gaussianprocess.org/gpml/chapters/RW1.pdf), what I *apparently* need is some sort of 'Gaussian Processes for Complete Morons'...  

...Because I don't want to start with matrix equations, or the definition of what is a multivariate Gaussian distribution, rather I want to start with 'what are we trying to do?'

### Starting from the Goal
Given some data points, can we make some mathematical model to fit a curve to them, and can we obtain some measure of the uncertainty in the curve fit? Or in Gal's words, *"What is a function that is likely to have generated our data?"*

In the following image from Gal's post \[spoiler: nothing in my post is original!\], we see what we might want in the end:

![GPgraph](http://mlg.eng.cam.ac.uk/yarin/blog_images/plot_uncertainty.jpg)

In the above image, the red points are the *given* data. The blue bits  show the predictions of our magical model, in two parts:

1. The solid line goes through all the given data, and ends up being the mean output of the GP system (which, sure, we haven't defined yet but we'll get there)  
2. The shaded blue regions denote the *range* of possible outputs from the model, in the form of the uncertainty.  The key thing to note is **that the uncertainty (or variance) is small near the given data, and in between the given data the uncertainty grows** (in fact, it's helpful to observe that the uncertainty reaches a maximum at the midpoint between any two given red points).  This property is sufficient to motivate what follows.  (For those wondering, "Why doesn't the uncertainty go to zero precisely *at* the red points?" You can make the model do that.  The above picture is just an example.)

Now, when we talk about uncertainties, we're talking about some kind of probability distribution, and often the simplest and most useful distribution is -- you guessed it -- gaussian. So, the model is going to combine gaussian functions using some kindad of measure of how far away a given (interpolation) point is from nearby given data points.

Conceptually, *that's it.*  But as with many things in the life of the scientist (e.g., General Relatvity), quantifying that simple, elegant concept involves  a \*\*\*\* ton of careful math. To wit...

### Recasting the Problem
First, some clarification and notation.  We'll regard the given $$$N$$$ domain values $$$x_i (i=1..N)$$$ as inputs and their corresponding range values $$$y_i$$$ as outputs.

Now, instead of regarding the x's as spaced out along the number line and having the model map individual values  $$$x\rightarrow y$$$ (i.e., $$$\mathbb{R}^1\rightarrow\mathbb{R}^1$$$), we're going to put all the $$$x_i$$$ values *together*, as specifying a single point (or 'vector' if you're a computer scientist rather than a physicist or mathematician) $$$\mathbf X$$$ in $$$N$$$-dimenstional space, with corresponding output point $$$\mathbf Y$$$,  and our model as mapping $$$\mathbf X\rightarrow \mathbf Y$$$
(i.e. $$$\mathbb{R}^N\rightarrow\mathbb{R}^N$$$).  Or, to be more careful, using the given data $$$\mathbf X$$$ and $$$\mathbf Y$$$, and given some *new* input vector $$$\mathbf x^*$$$, the model will predict the probability of an output of $$$\mathbf y^*$$$, i.e. it will predict

$$p(\mathbf{y^*| x^*, X, Y})$$

Gal again: "The expectation of $$$\mathbf y^*$$$ is called the *predictive mean* of the model, and its variance is called the *predictive uncertainty*."

**TODO:** more later
