import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# Breaking down large csv into columns of Genre and common words used in both Horror and Comedy Movies
CSV_COLUMN_NAMES = ["Genre", "laugh", "marri", "dead", "heart", "cop"]
df = pd.read_csv('movies.csv')
df_new = df.drop(columns=['Title', 'Year', 'Rating', '# Words'])

# Slicing table into training and test sets
df_new['Genre'] = pd.Categorical(df['Genre'])
df_new['Genre'] = df_new.Genre.astype('category').cat.codes
df_new = df_new.get(CSV_COLUMN_NAMES)
train_model, test_model = train_test_split(df_new, test_size=0.2)

# removing the Genre column from our training and test sets
train_y = train_model.pop('Genre')
test_y = test_model.pop('Genre')


def input_evaluation_set():
    """Takes set of features and creates a dictionary using the training set"""
    features = {'laugh': np.array(train_y.loc[:, 'laugh'].copy()),
                'marri': np.array(train_y.loc[:, 'marri'].copy()),
                'dead': np.array(train_y.loc[:, 'dead'].copy()),
                'heart': np.array(train_y.loc[:, 'heart'].copy()),
                'cop': np.array(train_y.loc[:, 'cop'].copy())}
    labels = np.array([2, 1])
    return features, labels


def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating and randomizing"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


# Feature columns describe how to use the input.
my_feature_columns = []
for key in train_model.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers with 35 and 15 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 35 and 15 nodes respectively.
    hidden_units=[35, 15],
    # The model must choose between 2 classes.
    n_classes=2)

# Train the Model.
classifier.train(input_fn=lambda: input_fn(train_model, train_y, training=True), steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test_model, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
print(eval_result)

# Creating expected values and prediction dictionary
expected = ['comedy', 'thriller']
predict_x = {'laugh': np.array([0.000000, 0.000000]),
             'marri':  np.array([0.002413, 0.001164]),
             'dead': np.array([0.000000, 0.000333]),
             'heart':  np.array([0.000000, 0.000000]),
             'cop' : np.array([0.000000, 0.000832]) }


def input_fn1(features, batch_size=256):
    """An input function for prediction."""
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


predictions = classifier.predict(
    input_fn=lambda: input_fn1(predict_x))

for pred_dict, expect in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    variable = ''
    if class_id == 0:
        variable = 'comedy'
    else:
        variable = 'thriller'
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        variable, 100 * probability, expect))

