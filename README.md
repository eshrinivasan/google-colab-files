# google-colab-files


1. Run the python file for training bert with domain specific data

```python 1_run_bert_2ndlevel_trainer.py --train_file "./output_lt_10.txt" --model_name_or_path bert-large-uncased --do_train --output_dir ./train_data --line_by_line True```

2. Run the python file for prediction of domain specific masked word


```python 2_run_mlm_pred.py True "I absolutely love the [MASK]"```


3. Expected output("design")

```
pen
screen
tablet
design
game 
games
camera
display
surface
case
```

### Reference:

run_mlm.py from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py

https://stackoverflow.com/questions/64712375/fine-tune-bert-for-specific-domain-unsupervised


