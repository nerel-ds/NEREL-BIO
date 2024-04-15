The script to evaluate predictions for the BioNNE competition. 
### Usage
Firstly, create input and output directories
```
mkdir -p input/ref input/res output
```
Then, add your predictions file to the input/res folder. Add the golden results file to the input/ref directory. 
Finally, run the script by simply using
```
python score.py input output
```
You will see the macro-f1 score in the output/scores.txt
