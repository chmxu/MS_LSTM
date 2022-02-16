# Learning to Score Figure Skating Videos

This is the code for TCSVT paper 'Learning to Score Figure Skating Videos'. To run this code, you need to do as follows,

## Prepare the environment
This code needs pytorch 0.4.0.


## Download Fis-V Dataset
We provide the c3d feature of Fis-V videos here [Fis-V](https://drive.google.com/open?id=1UfJGT6cGwZ6Xps0B6N9_Ov5JapTFLF3Y), you can download it and unzip the file into any directory. The corresponding anotations are in ```./data```.

**The raw videos have been uploaded. You can download them from here [Fis-V](https://drive.google.com/file/d/1FQ0-H3gkdlcoNiCe8RtAoZ3n7H1psVCI/view?usp=sharing).**

## Train and test PCS Score
```{python}
python train.py --root /path/to/data -n {name of model} --pcs
```

## Train and test TES Score
```{python}
python train.py --root /path/to/data -n {name of model}
```



The codes in **rnn_cells** are simply modified version based on [skip-rnn](https://github.com/gitabcworld/skiprnn_pytorch.git).
