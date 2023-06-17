# A.I-ASSIGMENT-3

Code Description
This code performs various tasks on a dataset containing questions and their corresponding correctness labels. Here are the steps performed by the code:

Mounting Google Drive: The code mounts the Google Drive to access the dataset file.

Loading the dataset: The code reads a CSV file named "output.csv" from Google Drive using pandas. The dataset contains two columns: "question" and "is_correct".

Data preprocessing:

Dropping duplicates: The code drops duplicate questions from the dataset, keeping only the first occurrence.
Encoding labels: The code uses LabelEncoder from scikit-learn to encode the "is_correct" column, converting True/False labels to numerical values (1/0).
Adding additional features: The code adds three additional columns to the dataset: "num_characters" (number of characters in a question), "num_words" (number of words in a question), and "num_sentences" (number of sentences in a question). It uses the NLTK library to perform tokenization and sentence splitting.
Data analysis and visualization:

Label distribution: The code displays a pie chart to visualize the distribution of true and false labels in the dataset.
Descriptive statistics: The code calculates descriptive statistics (mean, standard deviation, minimum, maximum) for the numerical features: "num_characters", "num_words", and "num_sentences".
Descriptive statistics by label: The code calculates separate descriptive statistics for the numerical features based on the label. It provides statistics for questions marked as true and questions marked as false.
Histograms: The code plots histograms to visualize the distribution of the numerical features ("num_characters" and "num_words") based on the label.
Pair plot: The code creates a pair plot to visualize the relationships between the numerical features.
Natural Language Processing (NLP) preprocessing:

Downloading NLTK resources: The code downloads the necessary NLTK resources, specifically the "punkt" and "stopwords" corpora.
Text transformation function: The code defines a text transformation function that performs several preprocessing steps on a given text. It converts text to lowercase, tokenizes it, removes non-alphanumeric characters and stopwords, and applies stemming using the Porter stemming algorithm.
Text transformation: The code applies the text transformation function to the "question" column, creating a new column named "transformed_text" in the dataset. The transformed_text column contains preprocessed versions of the original questions.
Text analysis and visualization:

Spam corpus analysis: The code analyzes the transformed text of questions marked as true. It creates a list of words (spam_corpus) and visualizes the most common 30 words using a bar plot.
Ham corpus analysis: The code analyzes the transformed text of questions marked as false. It creates a list of words (ham_corpus) and visualizes the most common 30 words using a bar plot.
Model training and evaluation:

Splitting the dataset: The code splits the dataset into training and testing sets using a 80:20 ratio.
Tokenizing the text: The code tokenizes the training and testing sets using the Tokenizer class from Keras.
Padding sequences: The code pads the tokenized sequences to ensure they have the same length.
Model architecture: The code defines a sequential model using Keras. It includes an embedding layer, bidirectional LSTM layers, dense layers, dropout layer, and output layer.
Learning rate schedule and optimizer: The code defines a learning rate schedule and uses the Adam optimizer for model compilation.
Model training: The code trains the model on the training data for 10
Building a simple language model using GPT-3.5 architecture to create a GitHub description for a given code snippet.
Importing necessary libraries and mounting Google Drive.
Reading a CSV file named "output.csv(2)" from Google Drive into a pandas DataFrame.
Dropping duplicate rows based on the "question" column and keeping the first occurrence.
Encoding the "is_correct" column using LabelEncoder.
Visualizing the distribution of the encoded "is_correct" column using a pie chart.
Installing the NLTK library and downloading the required tokenizer.
Preprocessing the text data by removing stopwords, punctuation, and applying stemming.
Adding new columns to the DataFrame to store the number of characters, words, and sentences in each question.
Analyzing the descriptive statistics of the newly added columns.
Visualizing the distribution of the number of characters and words based on the "is_correct" column using histograms.
Creating a pair plot and correlation heatmap to explore the relationships between the numerical columns.
Importing the NLTK library, downloading the stopwords corpus, and defining a function to transform text data by removing stopwords, punctuation, and applying stemming.
Applying the transformation function to the "question" column and storing the results in a new column named "transformed_text".
Analyzing the most common words in the "transformed_text" column for both the "is_correct" values using bar plots.
Preparing the data for training a machine learning model by splitting it into training and testing sets.
Installing the Streamlit library for building interactive web applications.
Importing necessary libraries and setting up the TensorFlow environment for training a deep learning model.
Loading the dataset from a CSV file named "my_dataframe.csv".
Splitting the dataset into training and testing sets using the train_test_split function.
Tokenizing the text data and converting it to sequences for training the LSTM-based model.
Defining the model architecture using the Sequential API from Keras, which includes an embedding layer, bidirectional LSTM layers, and dense layers.
Compiling the model with binary cross-entropy loss and Adam optimizer with a learning rate schedule.
Training the model on the training set and evaluating its performance on the testing set.
Monitoring the training progress with accuracy and loss metrics.
Saving the trained model for future use.
This code snippet demonstrates various steps involved in preprocessing and analyzing text data, as well as building and training a deep learning model for binary classification. The model leverages LSTM layers for sequence modeling and a combination of other layers for feature extraction and prediction. The code also includes data visualization using plots and integrates the Streamlit library to create interactive web applications.
