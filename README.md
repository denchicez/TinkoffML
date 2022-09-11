# TinkoffML
Introductory task to the tinkoff school for ml. Complete in Python 3.8.



# Fast run
## Train model
```
python3 train.py --model model.pkl --input-dir data 
```
## Generate text
python3 generate.py --model model.pkl --length 10

# Tips
## For train you can add text in data folder! It will be train on this .txt!
## train.py has this arguments
#### --input-dir (path to directory with .txt files)
#### --model (path to save model)
## generate.py
#### --model (path to get model)
#### --prefix (begin text from generate text, not required)
#### --length (length (word counter) of output)
