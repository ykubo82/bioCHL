# Neurons learn by predicting future activity
This is the code to reproduce our results in our manuscripts, Neurons learn by predicting future activity: <br/>
https://www.biorxiv.org/content/10.1101/2020.09.25.314211v2 <br/>

To run CHL_clamped.py, go:

```
python CHL_clamped.py 
```
*for this code, please install pytoch.

*You might get an error because of line #234 due to sklearn version (works on 0.23.2). So if you get an error, please comment out the line #234 <br/>
*Additionally, this python code will create a directory, "with_delay13_inp12_clamped_f120_c120_ada" to save the results (log.txt) and parameters.



