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
First, some clarification and notation.  We'll regard the given <img src="/_drafts/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> domain values <img src="/_drafts/tex/dcf4a0404fbbe407a047fd99abc1d688.svg?invert_in_darkmode&sanitize=true" align=middle width=87.58570094999997pt height=24.65753399999998pt/> as inputs and their corresponding range values <img src="/_drafts/tex/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode&sanitize=true" align=middle width=12.710331149999991pt height=14.15524440000002pt/> as outputs.

Now, instead of regarding the x's as spaced out along the number line and having the model map individual values  <img src="/_drafts/tex/debeda0b97a39ec02d821c7004c2b0f6.svg?invert_in_darkmode&sanitize=true" align=middle width=43.61481299999999pt height=14.15524440000002pt/> (i.e., <img src="/_drafts/tex/0bb9016e94a207ba9c26a6901320e253.svg?invert_in_darkmode&sanitize=true" align=middle width=63.24196559999999pt height=26.76175259999998pt/>), we're going to put all the <img src="/_drafts/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> values *together*, as specifying a single point (or 'vector' if you're a computer scientist rather than a physicist or mathematician) <img src="/_drafts/tex/bff29ce102734463345939ae3e729eac.svg?invert_in_darkmode&sanitize=true" align=middle width=14.29216634999999pt height=22.55708729999998pt/> in <img src="/_drafts/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>-dimenstional space, with corresponding output point <img src="/_drafts/tex/249a487a7a503f1b1484d6e57a9ad8ec.svg?invert_in_darkmode&sanitize=true" align=middle width=14.764759349999988pt height=22.55708729999998pt/>,  and our model as mapping <img src="/_drafts/tex/44a216c04bfcfd783f6139df04b30371.svg?invert_in_darkmode&sanitize=true" align=middle width=54.62752514999998pt height=22.55708729999998pt/>
(i.e. <img src="/_drafts/tex/377204d07b0f94c8390eabb56a4239a3.svg?invert_in_darkmode&sanitize=true" align=middle width=73.42917284999999pt height=27.6567522pt/>).  Or, to be more careful, using the given data <img src="/_drafts/tex/bff29ce102734463345939ae3e729eac.svg?invert_in_darkmode&sanitize=true" align=middle width=14.29216634999999pt height=22.55708729999998pt/> and <img src="/_drafts/tex/249a487a7a503f1b1484d6e57a9ad8ec.svg?invert_in_darkmode&sanitize=true" align=middle width=14.764759349999988pt height=22.55708729999998pt/>, and given some *new* input vector <img src="/_drafts/tex/4c146ae4c2ecb1f0ddf21274ed6184c3.svg?invert_in_darkmode&sanitize=true" align=middle width=16.71231044999999pt height=22.63846199999998pt/>, the model will predict the probability of an output of <img src="/_drafts/tex/8499fb07f912d0f33837140fd42c5b20.svg?invert_in_darkmode&sanitize=true" align=middle width=16.974878249999993pt height=22.63846199999998pt/>, i.e. it will predict

<p align="center"><img src="/_drafts/tex/f80b33e0e6e91cc9bb1ba0e4e3b1a218.svg?invert_in_darkmode&sanitize=true" align=middle width=104.62187504999999pt height=16.438356pt/></p>

Gal again: "The expectation of <img src="/_drafts/tex/8499fb07f912d0f33837140fd42c5b20.svg?invert_in_darkmode&sanitize=true" align=middle width=16.974878249999993pt height=22.63846199999998pt/> is called the *predictive mean* of the model, and its variance is called the *predictive uncertainty*."

**TODO:** more later
