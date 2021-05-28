# Instructions for running the code in Azure ML

## Step 1: Second level pretraining on Bert large uncased with youtube subtitles collected

### Install pip dependencies
```
!pip install datasets
#install 4.7.0.dev0 version of transformers
!pip install git+git://github.com/huggingface/transformers/
```

### Clone the repo
!npx degit https://github.wdf.sap.corp/sentient-commerce/secondlevel-pretraining-mlm -f

### Execute the following commands 
1. Run the python file for training bert with domain specific data

```python 1_run_bert_2ndlevel_trainer.py --train_file "./output_lt_10.txt" --model_name_or_path bert-large-uncased --do_train --output_dir ./train_data --line_by_line True```

2. Run the python file for prediction of domain specific masked word

```python 2_run_mlm_pred.py True "the sound from the [MASK] is very good for this price range" ```

3. Expected output("design")

```
speakers
speaker
tablet
screen
pen
phone
device
box
camera
driver
```
4. A folder "train_data_28k" is generated with the 2nd level pre-trained model.

## Step 2: Using the 2nd level pretrained bert model, fine tune it for sentiment analysis downstream task

1. Run the python file to do sentiment analysis on Azure ML with "reviews" on product from the github repo from curiousily.com

```3_run_sentiment_analysis_2ndlevelbert.py```
 
A fine tuned model "best_model_state.bin" is generated in the binary format at the end of the train/test exercise

## Step 3: Use the fine tuned model for further analysis




