# Neurons learn by predicting future activity
This is code to reproduce our results from manuscript: "Neurons learn by predicting future activity":<br/>
https://www.biorxiv.org/content/10.1101/2020.09.25.314211v2

To run CHL_clamped.py, go:

```
python CHL_clamped.py 
```
*for this code, please install pytoch.


*This python code will create a directory "results" to save the results (log.txt) and parameters.

Currently, number of epochs is set up to 3 (execution time ~7min). You can change number of epochs to 601 (line #312) for full training. <br/>
*if you run it for full training, it will take 21 hours to finish.<br/>

*You might get an error in line #235 if you are using sklearn version different than # 0.23.2. If you get an error, you can  comment out the line #235, but training data will not be shuffled between epochs.







