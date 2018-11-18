# AI

## Get Started
1. Install Python 3.6
2. Create a virtual environment named env using [venv](https://docs.python.org/3/tutorial/venv.html)
3. Activate the virtual environment
4. Install all dependencies (`pip install -r requirements.txt`)
5. Start the server (`flask run`)

## Train Models
1. Prepare a train models data JSON based on train_models_sample.json.
2. `python train_models.py <train_models_data_path>`

## Save Predictions in Local
1. `python save_predictions.py local <stock_code>`

## Save Predictions onto Firebase Cloud Storage
1. Download Firebase service account key from [Firebase Console](https://console.firebase.google.com) and save it as firebase-adminsdk.json in credentials.
2. `python save_predictions.py cloud <stock_code>`

## Docs
- [Tensorflow](https://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [Keras](https://keras.io/)
- [Scikit Learn](http://scikit-learn.org/stable/index.html)
- [Pandas](https://pandas.pydata.org/)
- [Flask](http://flask.pocoo.org/)
