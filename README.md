# Neurons learn by predicting future activity
This is the code to reproduce our results in our manuscripts, Neurons learn by predicting future activity: <br/>
https://www.biorxiv.org/content/10.1101/2020.09.25.314211v2 <br/>

To run CHL_clamped.py, go:

```
python CHL_clamped.py 
```
*for this code, please install pytoch.

*You might get an error because of line #235 due to sklearn version (works on 0.23.2). So if you get an error, please comment out the line #235 <br/>
*Additionally, this python code will create a directory, "results" to save the results (log.txt) and parameters. <br/>

*Currently, number of epochs is set up to 3 (executin time ~7min). You can change number of epochs to 601 (line #312) for full training. <br/>
*if you run it for full training, it will take 21 hours to finish.

