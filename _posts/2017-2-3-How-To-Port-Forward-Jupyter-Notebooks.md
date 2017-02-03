---
layout: post
title: How to Port-Forward Jupyter Notebooks
---



1. For my setup, my machine-learning computer called "edges" sits inside the firewall.  
On edges, I run the Jupyter notebook...  
        `shawley@edges$ jupyter notebook --no-browser --port=8889`  
or for torch, similarly,  
        `shawley@edges:~$ itorch notebook --no-browser --port=8889`  
This generates a bunch of text, including a URL with a token...  
 It'll say, "Copy/paste this URL into your browser when you connect for the first time, to login with a token:  
       http://localhost:8889/?token=96c92fc27f102995044da89ae111914c28e51757d57bebfc"  

2. The computer "hedges"(that's right, hedges and edges)  is my server which is visible from the outside world:  
        `shawley@hedges:~$ ssh -Y -N -n -L 127.0.0.1:8889:127.0.0.1:8889 edges`


3. Then on my laptop, I run a similar port-forward...  
        `shawley@laptop:~$ ssh -N -n -L 127.0.0.1:8889:127.0.0.1:8889 hedges`  


4. And then on my laptop, I paste the URL from the jupyter (or itorch) notebook into my web browser...  
    http://localhost:8889/?token=96c92fc27f102995044da89ae111914c28e51757d57bebfc
...and it works!

Wohoo!


Credit: These instructions were obtained by following [this guide](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh), (with an additional "layer" of forwarding.)

