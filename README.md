This is an informal fork of the repository that my group of four created for our CS senior project in 2019-2020. It is a reproduction, using new code (uninformed by the original), of the paper "Text-Adaptive Generative Adversarial Networks" by Nam, Kim & Kim (NIPS 2018). This particular zip includes a subset of the original Caltech-UCSD Birds dataset for demonstrational purposes. 

Since the project was largely pair-programmed, and debugged by the entire group, I cannot claim sole authorship over any one section; however, in addition to contributing to the design, theory and prototyping of every section of this code, I also examined and (in some cases) edited nearly every section of it while doing my share of the debugging. I therefore hold myself responsible both for some of its strengths and all of its weaknesses, and, aside from some minor stylistic concerns that I refrained from editing, view it as representative of my coding ability.

That said, one caveat: I am aware that there are a handful of places, particularly in train.py, where we have some duplicated code or constants that should have been args. These problems largely just arose due to last-minute fixes before our presentation; were we preparing the code for publication, we of course would have cleaned them up first.

HOW TO RUN:
The code should work with a standard 3.7.9 Conda env: just run train.py with no parameters. (However, --epochs and --bsize will be your friends if you want to see results sooner or run into memory issues, respectively; also, if available, --cuda is highly recommended!)
