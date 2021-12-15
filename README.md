# PawNet

PawNet is a multiview model for predicting the popularity of pet photos. 

# Test

To test our model, you must first download the dataset from Kaggle. The dataset we used is provided by PetFinder.my. You can download the Pawpularity [here](https://www.kaggle.com/c/petfinder-pawpularity-score/data). 

If you are not sure how to download a dataset from Kaggle, you can download it directly through [this link](https://drive.google.com/file/d/17DfTZF5B9jLiH9zE76m_xhJIQhtzE87f/view?usp=sharing).

After you downloaded the dataset, please unzip it to the project's root folder. The file structure should look like this:

```
\PawNet
|- \petfinder-pawpularity-score
   |- \test
   |- \train
   |- ...
|- \model
   |- ...
|- main_test.py
|- ...
```

Now you are ready to test our model. Please run `pip install -r requirements.txt `  to install the requirments.

Then, you can run one of the following to test

```shell
python main_test.py  // Test PawNet (our model)
python densenet_test.py  // Test Densenet (baseline)
python resnet_test.py    // Test Resnet (baseline)
```

Enjoy!
