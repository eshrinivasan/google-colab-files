# Instructions for running the code in Azure ML

## Step 1: Second level pretraining on Bert large uncased with youtube subtitles collected

### 1.  Install pip dependencies
		```
		!pip install datasets
		#install 4.7.0.dev0 version of transformers
		!pip install git+git://github.com/huggingface/transformers/
		```
### 2. Clone the repo
		```
		!npx degit https://github.wdf.sap.corp/sentient-commerce/secondlevel-pretraining-mlm -f
		```

### 3. Execute the following commands

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
4. A folder "train\_data\_28k" is generated with the 2nd level pre-trained model.

## Step 2: Using the 2nd level pretrained bert model, fine tune it for sentiment analysis downstream task

### 1. Install pip dependencies

		```
		!pip install transformers==2.8.0
		!pip install torch==1.4.0
		!pip install seaborn
		```

### 2. Execute the following commands
	1. Run the python file to do sentiment analysis on Azure ML with "reviews" on product from the github repo from curiousily.com with the pretained model name set as PRE_TRAINED_MODEL_NAME = './train_data_28k/checkpoint-2500'

	```python 3_run_sentiment_analysis_2ndlevelbert.py```
 
A fine tuned model "best\_model\_state.bin" is generated in the binary format at the end of the train/test exercise

## Step 3: Use the fine tuned model for further analysis



## Frequently occurring issues while doing the pretraining or fine tuning:

1. Use the same version of transformers and torch modules as used in the reference artcles to avoid new errors
2. For CUDA memory errors, reduce the batch\_size in half ie. if the batch\_size is 16 try with batch\_size as 8
3. Try to use the latest version of Python ie. Python 3.8 while using Azure ML notebook
4. Sometimes errors occurs because the previous errors are cached, in that case restart the kernel and run again
5. Sometimes the older version of modules are not removed and hence will be used by the code which may cause some issues like "Expecting a tensor and not a string", in that case just verify if the version you are looking for is actually installed with the command 
```pip show transformers```







