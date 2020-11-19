<img align="right" src="../logo.png">


*Lab 5*: Recurrent Neural Networks and Sentiment Analysis
==================================================================================


This chapter covers the following topics:

-   Building RNNs
-   Working with LSTMs
-   Building a sentiment analyzer using LSTM
-   Deploying the application on Heroku


Technical requirements
======================


Heroku can be installed from [www.heroku.com](http://www.heroku.com/).
The data was taken from
<https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences>.


Building RNNs
=============


RNNs consist of recurrent layers. While they are
similar in many ways to the fully connected layers within a standard
feed forward neural network, these recurrent layers consist of a hidden
state that is updated at each step of the sequential input. This means
that for any given sequence, the model is initialized with a hidden
state, often represented as a one-dimensional vector. The first step of
our sequence is then fed into our model and the hidden state is updated
depending on some learned parameters. The second word is then fed into
the network and the hidden state is updated again depending on some
other learned parameters. These steps are repeated until the whole
sequence has been processed and we are left with the final hidden state.
This computation *loop*, with the hidden state carried over from the
previous computation and updated, is why we refer to these networks as
recurrent. This final hidden state is then connected to a further fully
connected layer and a final classification is predicted.

Our recurrent layer looks something like the following, where *h* is the
hidden state and *x* is our input at various time steps in our sequence.
For each iteration, we update our hidden state at each time step, *x*:


![](./images/B12365_05_1.jpg)

Figure 5.1 -- Recurrent layer

Alternatively, we can expand this out to the whole
sequence of time steps, which looks like this:


![](./images/B12365_05_2.jpg)

Figure 5.2 -- Sequence of time steps

This layer is for an input that is *n* time steps long. Our hidden state
is initialized in state *h*[0]{.subscript}, and then uses our first
input, *x*[1]{.subscript}, to compute the next hidden state,
*h*[1]{.subscript}. There are two sets of weight matrices that are also
learned---matrix *U*, which learns how the hidden state changes
between time steps, and matrix *W*, which learns
how each input step affects the hidden state.

We also apply a *tanh* activation function to the resulting product,
keeping the values of the hidden state between -1 and 1. The equation
for calculating any hidden state, *h*[t]{.subscript}, becomes the
following:

![](./images/Formula_05_001.png)

This is then repeated for each time step within our input sequence, and
the final output for this layer is our last hidden state,
*h*[n]{.subscript}. When our network learns, we perform a forward pass
through the network, as before, to compute our final classification. We
then calculate a loss against this prediction and backpropagate through
the network, as before, calculating gradients as we go. This
backpropagation process occurs through all the steps within the
recurrent layer, with the parameters between each input step and the
hidden state being learned.

We will see later that we can actually take the hidden state at each
time step, rather than using the final hidden state, which is useful for
sequence-to-sequence translation tasks in NLP. However, for the time
being, we will just take the hidden layer as output to the rest of the
network.



Using RNNs for sentiment analysis
---------------------------------

In the context of sentiment analysis, our model is
trained on a sentiment analysis dataset of reviews that consists of a
number of reviews in text and a label of 0 or 1,
depending on whether the review is negative or positive. This means that
our model becomes a classification task (where the two classes are
negative/positive). Our sentence is passed through a layer of learned
word embeddings to form a representation of the sentence comprising
several vectors (one for each word). These vectors are then fed
sequentially into our RNN layer and the final hidden state is passed
through another fully connected layer. Our model\'s output is a single
value between 0 and 1, depending on whether our model predicts a
negative or positive sentiment from the sentence.
This means our complete classification model looks like this:


![](./images/B12365_05_3.jpg)

Figure 5.3 -- Classification model

Now, we will highlight one of the issues with
RNNs---exploding and shrinking gradients---and how we can remedy this
using gradient clipping.



Exploding and shrinking gradients
---------------------------------

One issue that we are often faced with within RNNs is that of
**exploding or shrinking gradients**. We can think of
the recursive layer as a very deep network. When
calculating the gradients, we do so at every iteration of the hidden
state. If the gradient of the loss relative to the weights at any given
position becomes very big, this will have a multiplicative effect as it
feeds forward through all the iterations of the recurrent layer. This
can cause gradients to explode as they get very large very quickly. If
we have large gradients, this can cause instability in our network. On
the other hand, if the gradients within our hidden state are very small,
this will again have a multiplicative effect and the gradients will be
close to 0. This means that the gradients can become too small to
accurately update our parameters via gradient descent, meaning our model
fails to learn.

One technique we can use to prevent our gradients
from exploding is to use **gradient clipping**. This technique limits
our gradients to prevent them from becoming too
large. We simply choose a hyperparameter, *C*, and can calculate our
clipped gradient, as follows:

![](./images/Formula_05_002.png)

The following graph shows the relationship between the two variables:


![](./images/B12365_05_4.jpg)

Figure 5.4 -- Comparison of gradient clipping

Another technique we can use to prevent exploding or disappearing
gradients is to shorten our input sequence length. The effective depth
of our recurrent layer depends on the length of our input sequence as
the sequence length determines how many iterative updates we
need to perform on our hidden state. The fewer
number of steps in this process, the smaller the multiplicative effects
of the gradient accumulation between hidden states will be. By
intelligently picking the maximum sequence length as a hyperparameter in
our model, we can help prevent exploding and vanishing gradients.


Introducing LSTMs
=================


While RNNs allow us to use sequences of words as input to our models,
they are far from perfect. RNNs suffer from two
main flaws, which can be partially remedied by using a more
sophisticated version of the RNN, known as **LSTM**.

The basic structure of RNNs means that it is very difficult for them to
retain information long term. Consider a sentence that\'s 20 words long.
From our first word in the sentence affecting the initial hidden state
to the last word in the sentence, our hidden state is updated 20 times.
From the beginning of our sentence to our final hidden state, it is very
difficult for an RNN to retain information about words at the beginning
of the sentence. This means that RNNs aren\'t very good at capturing
long-term dependencies within sequences. This also ties in with the
vanishing gradient problem mentioned earlier, where it is very
inefficient to backpropagate through long, sparse sequences of vectors.

Consider a long paragraph where we are trying to predict the next word.
The sentence begins with `I study math…` and ends with
`my final exam is in…`. Intuitively, we would expect the next
word to be `math` or some math-related field. However, in an
RNN model on a long sequence, our hidden state may struggle to retain
the information for the beginning of the sentence by the time it reaches
the end of the sentence as it takes multiple update steps.

We should also note that RNNs are poor at capturing the context of words
within a sentence as a whole. We saw earlier, when looking at n-gram
models, that the meaning of a word in a sentence is dependent on its
context within the sentence, which is determined by the words that occur
before it and the words that occur after it. Within an RNN, our hidden
state updates in one direction only. In a single forward pass, our
hidden state is initialized and the first word in the sequence is passed
into it. This process is then repeated with all the subsequent words in
the sentence sequentially until we are left with
our final hidden state. This means that for any given word in a
sentence, we have only considered the cumulative effect of the words
that have occurred before it in the sentence up to that point. We do not
consider any words that follow it, meaning we do not capture the full
context of each word in the sentence.

In another example, we again want to predict the missing word in a
sentence, but it now occurs toward the beginning as opposed to at the
end. We have the sentence
`I grew up in…so I can speak fluent Dutch`. Here, we can
intuitively guess that the person grew up in the Netherlands from the
fact that they speak Dutch. However, because an RNN parses this
information sequentially, it would only use `I grew up in…` to
make a prediction, missing the other key context within the sentence.

Both of these issues can be partially addressed using LSTMs.



Working with LSTMs
------------------

LSTMs are more advanced versions of RNNs and
contain two extra properties---an **update gate** and a **forget gate**.
These two 
easier for the network to learn long-term dependencies. Consider the
following film review:

*The film was amazing. I went to see it with my wife and my daughters on
Tuesday afternoon. Although I didn\'t expect it to be very entertaining,
it turned out to be loads of fun. We would definitely go back and see it
again given the chance.*

In sentiment analysis, it is clear that not all of the words in the
sentence are relevant in determining whether it is a positive or
negative review. We will repeat this sentence, but this time
highlighting the words that are relevant to gauging the sentiment of the
review:

*The film was amazing. I went to see it with my wife and my daughters on
Tuesday afternoon. Although I didn\'t expect it to be very entertaining,
it turned out to be loads of fun. We would definitely go back and see it
again given the chance.*

LSTMs attempt to do exactly this---remember the relevant words within a
sentence while forgetting all the irrelevant information. By doing this,
it stops the irrelevant information from diluting
the relevant information, meaning long-term
dependencies can be better learned across long sequences.

LSTMs are very similar in structure to RNNs. While there is a hidden
state that is carried over between steps within the LSTM, the inner
workings of the LSTM cell itself are different from that of the RNN:

![](./images/54.PNG)

 Figure 5.5 -- LSTM cell



LSTM cells
----------

While an RNN cell just takes the previous hidden
state and the new input step and calculates the next hidden state using
some learned parameters, the inner workings of an LSTM cell are
significantly more complicated:


![](./images/B12365_05_6.jpg)

Figure 5.6 -- Inner workings of an LSTM cell

While this looks significantly more daunting than the RNN, we will
explain each component of the LSTM cell in turn.
We will first look at the **forget gate** (indicated by
the bold rectangle):


![](./images/B12365_05_7.jpg)

Figure 5.7 -- The forget gate

The forget gate essentially learns which elements of the sequence to
forget. The previous hidden state, *h*[t-1]{.subscript}, and the latest
input step, *x*[1]{.subscript}, are concatenated together and passed
through a matrix of learned weights on the forget gate and a sigmoid
function that squashes the values between 0 and 1.
This resulting matrix, *ft*, is multiplied pointwise by the cell state
from the previous step, *c*[t-1]{.subscript}. This effectively applies a
mask to the previous cell state so that only the relevant information
from the previous cell state is brought forward.

Next, we will look at the **input gate**:


![](./images/B12365_05_8.jpg)

Figure 5.8 -- The input gate

The input gate again takes the concatenated previous hidden state,
*h*[t-1]{.subscript}, and the current sequence input,
*x*[t]{.subscript}, and passes this through a sigmoid function with
learned parameters, which outputs another matrix, *i*[t]{.subscript},
consisting of values between 0 and 1. The concatenated hidden
state and sequence input also pass through a tanh
function, which squashes the output between -1 and 1. This is multiplied
by the *i*[t]{.subscript} matrix. This means that the learned parameters
required to generate *i*[t]{.subscript} effectively learn which elements
should be kept from the current time step in our cell state. This is
then added to the current cell state to get our final cell state, which
will be carried over to the next time step.

Finally, we have the last element of the LSTM
cell---the **output gate**:


![](./images/B12365_05_9.jpg)

Figure 5.9 -- The output gate

The output gate calculates the final output of the LSTM cell---both the
cell state and the hidden state that is carried over to the next step.
The cell state, *c*[t]{.subscript}, is unchanged from the previous two
steps and is a product of the forget gate and the
input gate. The final hidden state, *h*[t]{.subscript}, is calculated by
taking the concatenated previous hidden state, *h*[t-1]{.subscript}, and
the current time step input, *x*[t]{.subscript}, and passing through a
sigmoid function with some learned parameters to get the output gate
output, *o*[t]{.subscript}. The final cell state, *c*[t]{.subscript}, is
passed through a tanh function and multiplied by the output gate output,
*o*[t]{.subscript}, to calculate the final hidden state,
*h*[t]{.subscript}. This means that the learned parameters on the output
gate effectively control which elements of the previous hidden state and
current output are combined with the final cell state to carry over to
the next time step as the new hidden state.

In our forward pass, we simply iterate through the model, initializing
our hidden state and cell state and updating them at each time step
using the LSTM cells until we are left with a final hidden state, which
is output to the next layer of our neural network. By backpropagating
through all the layers of our LSTM, we can calculate the gradients
relative to the loss of the network and so we know which direction to
update our parameters through gradient descent. We get several matrices
or parameters---one for the input gate, one for the output gate, and one
for the forget gate.

Because we get more parameters than for a simple
RNN and our computation graph is more complex, the process of
backpropagating through the network and updating the weights will likely
take longer than for a simple RNN. However, despite the longer training
time, we have shown that LSTM offers significant advantages over a
conventional RNN as the output gate, input gate, and forget gate all
combine to give the model the ability to determine which elements of the
input should be used to update the hidden state and which elements of
the hidden state should be forgotten going forward, which means the
model is better able to form long-term dependencies and retain
information from previous sequence steps.



Bidirectional LSTMs
-------------------

We previously mentioned that a downside of simple
RNNs is that they fail to capture the full context of a word within a
sentence as they are backward-looking only. At each time step of the
RNN, only the previously seen words are considered and the words
occurring next within the sentence are not taken into account. While
basic LSTMs are similarly backward-facing, we can use a modified version
of LSTM, known as a **bidirectional LSTM**, which considers both the
words before and after it at each time step within the sequence.

Bidirectional LSTMs process sequences in regular order and reverse order
simultaneously, maintaining two hidden states.
We\'ll call the forward hidden state *f*[t]{.subscript} and use
*r*[t]{.subscript} for the reverse hidden state:

![](./images/55.PNG)

Figure 5.10 -- The bidirectional LSTM process

Here, we can see that we maintain these two hidden states throughout the
whole process and use them to calculate a final hidden state,
*h*[t]{.subscript}. Therefore, if we wish to calculate the final hidden
state at time step *t*, we use the forward hidden state,
*f*[t]{.subscript}, which has seen all words up to and including input
*x*[t]{.subscript}, as well as the reverse hidden state,
*r*[t]{.subscript}, which has seen all the words after and including
*x*[t]{.subscript}. Therefore, our final hidden state,
*h*[t]{.subscript}, comprises hidden states that have seen all the words
in the sentence, not just the words occurring
before time step *t*. This means that the context of any given word
within the whole sentence can be better captured. Bidirectional LSTMs
have proven to offer improved performance on several NLP tasks over
conventional unidirectional LSTMs.


Building a sentiment analyzer using LSTMs
=========================================


We will now look at how to build our own simple
LSTM to categorize sentences based on their sentiment. We will train our
model on a dataset of 3,000 reviews that have been categorized as
positive or negative. These reviews come from
three different sources---film reviews, product reviews, and location
reviews---in order to ensure that our sentiment analyzer is robust. The
dataset is balanced so that it consists of 1,500 positive reviews and
1,500 negative reviews. We will start by importing our dataset and
examining it:

```
with open("sentiment labelled sentences/sentiment.txt") as f:
    reviews = f.read()
    
data = pd.DataFrame([review.split('\t') for review in                      reviews.split('\n')])
data.columns = ['Review','Sentiment']
data = data.sample(frac=1)
```


This returns the following output:

![](./images/56.PNG)

Figure 5.11 -- Output of the dataset

We read in our dataset from the file. Our dataset is tab-separated, so
we split it up with tabs and the new line
character. We rename our columns and then use the sample function to
randomly shuffle our data. Looking at our dataset, the first thing we
need to be able to do is preprocess our sentences to feed them into our
LSTM model.



Preprocessing the data
----------------------

First, we create a function to tokenize our data,
splitting each review into a list of individual preprocessed words. We
loop through our dataset and for each review, we remove any punctuation,
convert letters into lowercase, and remove any trailing whitespace. We
then use the NLTK tokenizer to create individual tokens from this
preprocessed text:

```
def split_words_reviews(data):
    text = list(data['Review'].values)
    clean_text = []
    for t in text:
        clean_text.append(t.translate(str.maketrans('', '',                   punctuation)).lower().rstrip())
    tokenized = [word_tokenize(x) for x in clean_text]
    all_text = []
    for tokens in tokenized:
        for t in tokens:
            all_text.append(t)
    return tokenized, set(all_text)
reviews, vocab = split_words_reviews(data)
reviews[0]
```


This results in the following output:

![](./images/57.PNG)


Figure 5.12 -- Output of NTLK tokenization

We return the reviews themselves, as well as a set of all words within
all the reviews (that is, the vocabulary/corpus), which we will use to
create our vocab dictionaries.

In order to fully prepare our sentences for entry into a neural network,
we must convert our words into numbers. In order to do this, we create a
couple of dictionaries, which will allow us to convert data from word
into index and from index into word. To do this, we simply loop through
our corpus and assign an index to each unique word:

```
def create_dictionaries(words):
    word_to_int_dict = {w:i+1 for i, w in enumerate(words)}
    int_to_word_dict = {i:w for w, i in word_to_int_dict.                            items()}
    return word_to_int_dict, int_to_word_dict
word_to_int_dict, int_to_word_dict = create_dictionaries(vocab)
int_to_word_dict
```


This gives the following output:

![](./images/58.PNG)

Figure 5.13 -- Assigning an index to each word

Our neural network will take input of a fixed length; however, if we
explore our reviews, we will see that our reviews
are all of different lengths. In order to ensure that all of our inputs
are of the same length, we will *pad* our input sentences. This
essentially means that we add empty tokens to shorter sentences so that
all the sentences are of the same length. We must first decide on the
length of the padding we wish to implement. We first calculate the
maximum length of a sentence in our input reviews, as well as the
average length:

```
print(np.max([len(x) for x in reviews]))
print(np.mean([len(x) for x in reviews]))
```


This gives the following:

![](./images/59.PNG)


Figure 5.14 -- Length value

We can see that the longest sentence is `70` words long and
the average sentence length has a length of `11.78`. To
capture all the information from all our sentences, we want to pad all
of our sentences so that they have a length of 70. However, using longer
sentences means longer sequences, which causes our LSTM layer to become
deeper. This means model training takes longer as we have to
backpropagate our gradients through more layers, but it also means that
a large percentage of our inputs would just be sparse
and full of empty tokens, which makes learning
from our data much less efficient. This is illustrated by the fact that
our maximum sentence length is much larger than our average sentence
length. In order to capture the majority of our sentence information
without unnecessarily padding our inputs and making them too sparse, we
opt to use an input size of `50`. You may wish to experiment
with using different input sizes between `20` and
`70` to see how this affects your model performance.

We will create a function that allows us to pad our sentences so that
they are all the same size. For reviews shorter than the sequence
length, we pad them with empty tokens. For reviews longer than the
sequence length, we simply drop any tokens over the maximum sequence
length:

```
def pad_text(tokenized_reviews, seq_length):
    
    reviews = []
    
    for review in tokenized_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append(['']*(seq_length-len(review)) +                    review)
        
    return np.array(reviews)
padded_sentences = pad_text(reviews, seq_length = 50)
padded_sentences[0]
```


Our padded sentence looks like this:


![](./images/B12365_05_15.jpg)

Figure 5.15 -- Padding the sentences

We must make one further adjustment to allow the use of empty tokens
within our model. Currently, our vocabulary
dictionaries do not know how to convert empty tokens into integers to
use within our network. Because of this, we manually add these to our
dictionaries with index `0`, which means empty tokens will be
given a value of `0` when fed into our model:

```
int_to_word_dict[0] = ''
word_to_int_dict[''] = 0
```


We are now very nearly ready to begin training our model. We perform one
final step of preprocessing and encode all of our padded sentences as
numeric sequences for feeding into our neural network. This means that
the previous padded sentence now looks like this:

```
encoded_sentences = np.array([[word_to_int_dict[word] for word in review] for review in padded_sentences])
encoded_sentences[0]
```


Our encoded sentence is represented as follows:

![](./images/60.PNG)

Figure 5.16 -- Encoding the sentence

Now that we have all our input sequences encoded as numerical vectors,
we are ready to begin designing our model architecture.



Model architecture
------------------

Our model will consist of several main parts.
Besides the input and output layers that are common to many neural
networks, we will first require an **embedding layer**. This is so that
our model learns the vector representations of
the words it is being trained on. We could opt to
use precomputed embeddings (such as GLoVe), but for demonstrative
purposes, we will train our own embedding layer. Our input sequences are
fed through the input layer and come out as sequences of vectors.

These vector sequences are then fed into our
**LSTM layer**. As explained in detail earlier in this chapter, the LSTM
layer learns sequentially from our sequence of embeddings and outputs a
single vector output representing the final hidden state of the LSTM
layer. This final hidden state is finally passed
through a further **hidden layer** before the final output node predicts
a value between 0 and 1, indicating whether the input sequence was a
positive or negative review. This means that our model architecture
looks something like this:


![](./images/B12365_05_17.jpg)

Figure 5.17 -- Model architecture

We will now demonstrate how to code this model
from scratch using PyTorch. We create a class called
`SentimentLSTM`, which inherits from the `nn.Module`
class. We define our `init` parameters as the size
of our vocab, the number of LSTM layers
our model will have, and the size of our model\'s
hidden state:

```
class SentimentLSTM(nn.Module):
    
    def __init__(self, n_vocab, n_embed, n_hidden, n_output,    n_layers, drop_p = 0.8):
        super().__init__()
        
        self.n_vocab = n_vocab  
        self.n_layers = n_layers 
        self.n_hidden = n_hidden 
```


We then define each of the layers of our network. Firstly, we define our
embedding layer, which will have the length of the
number of words in our vocabulary and the size of the embedding vectors
as a `n_embed` hyperparameter to be specified. Our LSTM layer
is defined using the output vector size from the embedding layer, the
length of the model\'s hidden state, and the
number of layers that our LSTM layer will have. We also add an argument
to specify that our LSTM can be trained on batches
of data and an argument to allow us to implement network regularization
via dropout. We define a further dropout layer with probability,
`drop_p` (a hyperparameter to be specified on model creation),
as well as our definitions of our final fully connected layer and
output/prediction node (with a sigmoid activation function):

```
       self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers,                        batch_first = True, dropout = drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()
```


Next, we need to define our forward pass within our model class. Within
this forward pass, we just chain together the output of one layer to
become the input into our next layer. Here, we can see that our
embedding layer takes `input_words` as input and outputs the
embedded words. Then, our LSTM layer takes embedded words as input and
outputs `lstm_out`. The only nuance here is that we use
`view()` to reshape our tensors from the LSTM output to be the
correct size for input into our fully connected layer. The same also
applies for reshaping the output of our hidden layer to match that of
our output node. Note that our output will return a prediction for
`class = 0` and `class = 1`, so we slice the output
to only return a prediction for `class = 1`---that is, the
probability that our sentence is positive:

```
 def forward (self, input_words):
                          
        embedded_words = self.embedding(input_words)
        lstm_out, h = self.lstm(embedded_words) 
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1,                             self.n_hidden)
        fc_out = self.fc(lstm_out)                  
        sigmoid_out = self.sigmoid(fc_out)              
        sigmoid_out = sigmoid_out.view(batch_size, -1)  
        
        sigmoid_last = sigmoid_out[:, -1]
        
        return sigmoid_last, h
```


We also define a function called `init_hidden()`, which
initializes our hidden layer with the dimensions
of our batch size. This allows our model to train
and predict on many sentences at once, rather than just training on one
sentence at a time, if we so wish. Note that we
define `device` as `"cpu"` here to run it on our
local processor. However, it is also possible to set this to a
CUDA-enabled GPU in order to train it on a GPU if you have one:

```
    def init_hidden (self, batch_size):
        
        device = "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size,\
                 self.n_hidden).zero_().to(device),\
             weights.new(self.n_layers, batch_size,\
                 self.n_hidden).zero_().to(device))
        
        return h
```


We then initialize our model by creating a new instance of the
`SentimentLSTM` class. We pass the size of our vocab, the size
of our embeddings, the size of our hidden state, as well as the output
size, and the number of layers in our LSTM:

```
n_vocab = len(word_to_int_dict)
n_embed = 50
n_hidden = 100
n_output = 1
n_layers = 2
net = SentimentLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers)
```


Now that we 
model architecture fully, it\'s time to begin training our model.



Training the model
------------------

To train our model, we must first define our
datasets. We will train our model using a training set of data, evaluate
our trained model at each step on a validation set, and then finally,
measure our model\'s final performance using an unseen test set of data.
The reason we use a test set that is separate from our validation
training is that we may wish to fine-tune our model hyperparameters
based on the loss against the validation set. If we do this, we may end
up picking the hyperparameters that are only optimal in performance for
that particular validation set of data. We evaluate a final time against
an unseen test set to make sure our model generalizes well to data it
hasn\'t seen before at any part of the training loop.

We have already defined our model inputs (*x*) as
`encoded_sentences`, but we must also define our model output
(*y*). We do this simply, as follows:

```
labels = np.array([int(x) for x in data['Sentiment'].values])
```


Next, we define our training and validation ratios. In this case, we
will train our model on 80% of the data, validate on a further 10% of
the data, and finally, test on the remaining 10% of the data:

```
train_ratio = 0.8
valid_ratio = (1 - train_ratio)/2
```


We then use these ratios to slice our data and
transform them into tensors and then tensor datasets:

```
total = len(encoded_sentences)
train_cutoff = int(total * train_ratio)
valid_cutoff = int(total * (1 - valid_ratio))
train_x, train_y = torch.Tensor(encoded_sentences[:train_cutoff]).long(), torch.Tensor(labels[:train_cutoff]).long()
valid_x, valid_y = torch.Tensor(encoded_sentences[train_cutoff : valid_cutoff]).long(), torch.Tensor(labels[train_cutoff : valid_cutoff]).long()
test_x, test_y = torch.Tensor(encoded_sentences[valid_cutoff:]).long(), torch.Tensor(labels[valid_cutoff:])
train_data = TensorDataset(train_x, train_y)
valid_data = TensorDataset(valid_x, valid_y)
test_data = TensorDataset(test_x, test_y)
```


Then, we use these datasets to create PyTorch `DataLoader`
objects. `DataLoader` allows us to batch process our datasets
with the `batch_size` parameter, allowing different batch
sizes to be easily passed to our model. In this instance, we will keep
it simple and set `batch_size = 1`, which means our model will
be trained on individual sentences, rather than using larger batches of
data. We also opt to randomly shuffle our `DataLoader` objects
so that data is passed through our neural network in random order,
rather than the same order each epoch, potentially removing any biased
results from the training order:

```
batch_size = 1
train_loader = DataLoader(train_data, batch_size = batch_size,                          shuffle = True)
valid_loader = DataLoader(valid_data, batch_size = batch_size,                          shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size,                         shuffle = True)
```


Now that we have defined our `DataLoader` object for each of
our three datasets, we define our training loop. We first define a
number of hyperparameters, which will be used
within our training loop. Most importantly, we
define our loss function as binary cross entropy (as we are dealing with
predicting a single binary class) and we define our optimizer to be
`Adam` with a learning rate of `0.001`. We also
define our model to run for a short number of epochs (to save time) and
set `clip = 5` to define our gradient clipping:

```
print_every = 2400
step = 0
n_epochs = 3
clip = 5  
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
```


The main body of our training loop looks like this:

```
for epoch in range(n_epochs):
    h = net.init_hidden(batch_size)
    
    for inputs, labels in train_loader:
        step += 1  
        net.zero_grad()
        output, h = net(inputs)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), clip)
        optimizer.step()
```


Here, we just train our model for a number of epochs, and for every
epoch, we first initialize our hidden layer using the batch size
parameter. In this instance, we set `batch_size = 1` as we are
just training our model one sentence at a time.
For each batch of input sentences and labels within our train loader, we
first zero our gradients (to stop them accumulating) and calculate our
model outputs using the forward pass of our data using the model\'s
current state. Using this output, we then calculate our loss using the
predicted output from the model and the correct labels. We then perform
a backward pass of this loss through our network to calculate the
gradients at each stage. Next, we use the `grad_clip_norm()`
function to clip our gradients as this will stop our gradients from
exploding, as mentioned earlier in this chapter. We defined
`clip = 5`, meaning the maximum gradient at any given node is
`5`. Finally, we update our weights using the gradients
calculated on our backward pass by calling `optimizer.step()`.

If we run this loop by itself, we will train our model. However, we want
to evaluate our model performance after every epoch in order to
determine its performance on a validation set of data. We do this as
follows:

```
if (step % print_every) == 0:            
            net.eval()
            valid_losses = []
            for v_inputs, v_labels in valid_loader:
                       
                v_output, v_h = net(v_inputs)
                v_loss = criterion(v_output.squeeze(),                                    v_labels.float())
                valid_losses.append(v_loss.item())
            print("Epoch: {}/{}".format((epoch+1), n_epochs),
                  "Step: {}".format(step),
                  "Training Loss: {:.4f}".format(loss.item()),
                  "Validation Loss: {:.4f}".format(np.                                     mean(valid_losses)))
            net.train()
```


This means that at the end of each epoch, our model calls
`net.eval()` to freeze the weights of our model and performs a
forward pass using our data as before. Note that dropout is also not
applied when we are in evaluation mode. However, this time, instead of
using the training data loader, we use the validation loader. By doing
this, we can calculate the total loss of the model\'s current state over
our validation set of data. Finally, we print our results and call
`net.train()` to unfreeze our model\'s weights so that we can
train again on the next epoch. Our output looks
something like this:


![](./images/B12365_05_18.jpg)

Figure 5.18 -- Training the model

Finally, we can save our model for future use:

```
torch.save(net.state_dict(), 'model.pkl')
```


After training our model for three epochs, we notice two main things.
We\'ll start with the good news first---our model is learning something!
Not only has our training loss fallen, but we can also see that our loss
on the validation set has fallen after each epoch. This means that our
model is better at predicting sentiment on an unseen set of data after
just three epochs! The bad news, however, is that our model is massively
overfitting. Our training loss is much lower than that of our validation
loss, showing that while our model has learned how to predict the
training set of data very well, this doesn\'t generalize as well to an
unseen set of data. This was expected to happen as we are
using a very small set of training data (just
2,400 training sentences). As we are training a whole embedding layer,
it is possible that many of the words occur just once in the training
set and never in the validation set and vice versa, making it
practically impossible for the model to generalize all the different
variety of words within our corpus. In practice, we would hope to train
our model on a much larger dataset to allow our model to learn how to
generalize much better. We have also trained this model over a very
short time period and have not performed hyperparameter tuning to
determine the best possible iteration of our model. Feel free to try
changing some of the parameters within the model (such as the training
time, hidden state size, embedding size, and so on) in order to improve
the performance of the model.

Although our model overfitted, it has still learned something. We now
wish to evaluate our model on a final test set of data. We perform one
final pass on the data using the test loader we defined earlier. Within
this pass, we loop through all of our test data and make predictions
using our final model:

```
net.eval()
test_losses = []
num_correct = 0
for inputs, labels in test_loader:
    test_output, test_h = net(inputs)
    loss = criterion(test_output, labels)
    test_losses.append(loss.item())
    
    preds = torch.round(test_output.squeeze())
    correct_tensor = preds.eq(labels.float().view_as(preds))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)
    
print("Test Loss: {:.4f}".format(np.mean(test_losses)))
print("Test Accuracy: {:.2f}".format(num_correct/len(test_loader.dataset)))    
```


Our performance on our test set of data is as follows:


![](./images/61.PNG)

Figure 5.19 -- Output values

We then compare our model predictions with our true labels to get
`correct_tensor`, which is a vector that evaluates whether
each of our model\'s predictions was correct. We then sum this vector
and divide it by its length to get our model\'s total accuracy. Here, we
get an accuracy of 76%. While our model is certainly far from perfect,
given our very small training set and limited training time, this is not
bad at all! This just serves to illustrate how
useful LSTMs can be when it comes to learning from NLP data. Next, we
will show how we can use our model to make predictions from new data.



Using our model to make predictions
-----------------------------------

Now that we have a trained model, it should be
possible to repeat our preprocessing steps on a new sentence, pass this
into our model, and get a prediction of it\'s sentiment. We first create
a function to preprocess our input sentence to predict:

```
def preprocess_review(review):
    review = review.translate(str.maketrans('', '',                    punctuation)).lower().rstrip()
    tokenized = word_tokenize(review)
    if len(tokenized) >= 50:
        review = tokenized[:50]
    else:
        review= ['0']*(50-len(tokenized)) + tokenized
    
    final = []
    
    for token in review:
        try:
            final.append(word_to_int_dict[token])
            
        except:
            final.append(word_to_int_dict[''])
        
    return final
```


We remove punctuation and trailing whitespace, convert letters into
lowercase, and tokenize our input sentence as before. We pad our
sentence to a sequence with a length of `50` and then convert
our tokens into numeric values using our
precomputed dictionary. Note that our input may contain new words that
our network hasn\'t seen before. In this case, our function treats these
as empty tokens.

Next, we create our actual `predict()` function. We preprocess
our input review, convert it into a tensor, and pass this into a data
loader. We then loop through this data loader (even though it only
contains one sentence) and pass our review through our network to obtain
a prediction. Finally, we evaluate our prediction and print whether it
is a positive or negative review:

```
def predict(review):
    net.eval()
    words = np.array([preprocess_review(review)])
    padded_words = torch.from_numpy(words)
    pred_loader = DataLoader(padded_words, batch_size = 1,                             shuffle = True)
    for x in pred_loader:
        output = net(x)[0].item()
    
    msg = "This is a positive review." if output >= 0.5 else           "This is a negative review."
    print(msg)
    print('Prediction = ' + str(output))
```


Finally, we just call `predict()` on our review to make a
prediction:

```
predict("The film was good")
```


This results in the following output:

![](./images/62.PNG)

Figure 5.20 -- Prediction string on a positive value

We also try using `predict()` on the negative value:

```
predict("It was not good")
```


This results in the following output:

![](./images/63.PNG)

Figure 5.21 -- Prediction string on a negative value

We have now built an LSTM model to perform sentiment analysis from the
ground up. Although our model is far from perfect,
we have demonstrated how we can take some sentiment labeled reviews and
train a model to be able to make predictions on new reviews. Next, we
will show how we can host our model on the Heroku cloud platform so that
other people can make predictions using your model


Deploying the application on Heroku
===================================


We have now trained our model on our local machine
and we can use this to make predictions. However, this isn\'t
necessarily any good if you want other people to
be able to use your model to make predictions. If we host our model on a
cloud-based platform, such as Heroku, and create a basic API, other
people will be able to make calls to the API to make predictions using
our model.



Introducing Heroku
------------------

**Heroku** is a cloud-based platform where you can
host your own basic programs. While the free tier of Heroku has a
maximum upload size of 500 MB and limited processing power, this should
be sufficient for us to host our model and create a basic API in order
to make predictions using our model.

The first step is to create a free account on Heroku and install the
Heroku app. Then, in the command line, type the following command:

```
heroku login
```


Log in using your account details. Then, create a new `heroku`
project by typing the following command:

```
heroku create sentiment-analysis-flask-api
```


Note that all the project names must be unique, so you will need to pick
a project name that isn\'t `sentiment-analysis-flask-api`.

Our first step is building a basic API using
Flask.



Creating an API using Flask -- file structure
---------------------------------------------

Creating an API is fairly simple using Flask as
Flask contains a default template required to make
an API:

First, in the command line, create a new folder
for your flask API and navigate to it:

```
mkdir flaskAPI
cd flaskAPI
```


Then, create a virtual environment within the folder. This will be the
Python environment that your API will use:

```
python3 -m venv vir_env
```


Within your environment, install all the packages that you will need
using `pip`. This includes all the packages that you use
within your model program, such as NLTK, `pandas`, NumPy, and
PyTorch, as well as the packages you will need to run the API, such as
Flask and Gunicorn:

```
pip install nltk pandas numpy torch flask gunicorn
```


We then create a list of requirements that our API will use. Note that
when we upload this to Heroku, Heroku will automatically download and
install all the packages within this list. We can do this by typing the
following:

```
pip freeze > requirements.txt
```


One adjustment we need to make is to replace the `torch` line
within the `requirements.txt` file with the following:

```
https://download.pytorch.org/whl/cpu/torch-1.3.1%2Bcpu-cp37-cp37m-linux_x86_64.whl
```


This is a link to the wheel file of the version of PyTorch that only
contains the CPU implementation. The full version of PyTorch that
includes full GPU support is over 500 MB in size, so it will not run on
the free Heroku cluster. Using this more compact version of PyTorch
means that you will still be able to run your model using PyTorch on
Heroku. Finally, we create three more files within our folder, as well
as a final directory for our models:

```
touch app.py
touch Procfile
touch wsgi.py
mkdir models
```


Now, we have created all the files we will need
for our Flash API and we are ready to start making
adjustments to our file.



Creating an API using Flask -- API file
---------------------------------------

Within our `app.py` file, we can begin
building our API:

1.  We first carry out all of our imports and
    create a `predict` route. This allows us to call our API
    with the `predict` argument in order to run a
    `predict()` method within our API:
    ```
    import flask
    from flask import Flask, jsonify, request
    import json
    import pandas as pd
    from string import punctuation
    import numpy as np
    import torch
    from nltk.tokenize import word_tokenize
    from torch.utils.data import TensorDataset, DataLoader
    from torch import nn
    from torch import optim
    app = Flask(__name__)
    @app.route('/predict', methods=['GET'])
    ```
    

2.  Next, we define our `predict()` method within our
    `app.py` file. This is largely a rehash of our model file,
    so to avoid repetition of code, it is advised that you look at the
    completed `app.py` file within the GitHub repository
    linked in the *Technical requirements* section of this chapter. You
    will see that there are a few additional
    lines. Firstly, within our `preprocess_review()` function,
    we will see the following lines:

    ```
    with open('models/word_to_int_dict.json') as handle:
    word_to_int_dict = json.load(handle)
    ```
    

    This takes the `word_to_int` dictionary we computed within
    our main model notebook and loads it into our
    model. This is so that our word indexing is consistent with our
    trained model. We then use this dictionary to convert our input text
    into an encoded sequence. Be sure to take the
    `word_to_int_dict.json` file from the original notebook
    output and place it within the `models` directory.

3.  Similarly, we must also load the weights from our trained model. We
    first define our `SentimentLSTM` class and the load our
    weights using `torch.load`. We will use the
    `.pkl` file from our original notebook, so be sure to
    place this in the `models` directory as well:
    ```
    model = SentimentLSTM(5401, 50, 100, 1, 2)
    model.load_state_dict(torch.load("models/model_nlp.pkl"))
    ```
    

4.  We must also define the input and outputs of our API. We want our
    model to take the input from our API and pass this to our
    `preprocess_review()` function. We do this using
    `request.get_json()`:
    ```
    request_json = request.get_json()
    i = request_json['input']
    words = np.array([preprocess_review(review=i)])
    ```
    

5.  To define our output, we return a JSON response consisting of the
    output from our model and a response code,
    `200`, which is what is returned by our predict function:
    ```
    output = model(x)[0].item()
    response = json.dumps({'response': output})
        return response, 200
    ```
    

6.  With the main body of our app complete, there
    are just two more additional things we must add in order to make our
    API run. We must first add the following to our `wsgi.py`
    file:
    ```
    from app import app as application
    if __name__ == "__main__":
        application.run()
    ```
    

7.  Finally, add the following to our Procfile:
    ```
    web: gunicorn app:app --preload
    ```
    

That\'s all that\'s required for the app to run. We can test that our
API runs by first starting the API locally using the following command:

```
gunicorn --bind 0.0.0.0:8080 wsgi:application -w 1
```


Once the API is running locally, we can make a request to the API by
passing it a sentence to predict the outcome:

```
curl -X GET http://0.0.0.0:8080/predict -H "Content-Type: application/json" -d '{"input":"the film was good"}'
```


If everything is working correctly, you should receive a prediction from
the API. Now that we have our API making predictions locally, it is time
to host it on Heroku so that we can make predictions in the cloud.



Creating an API using Flask -- hosting on Heroku
------------------------------------------------

We first need to commit our files to Heroku in a
similar way to how we would commit files using
GitHub. We define our working `flaskAPI` directory as a
`git` folder by simply running the following command:

```
git init
```


Within this folder, we add the following code to the
`.gitignore` file, which will stop us from adding unnecessary
files to the Heroku repo:

```
vir_env
__pycache__/
.DS_Store
```


Finally, we add our first `commit` function and push it to our
`heroku` project:

```
git add . -A 
git commit -m 'commit message here'
git push heroku master
```


This may take some time to compile as not only does the system have to
copy all the files from your local directory to Heroku, but Heroku will
automatically build your defined environment, installing all the
required packages and running your API.

Now, if everything has worked correctly, your API will automatically run
on the Heroku cloud. In order to make predictions, you can simply make a
request to the API using your project name instead of
`sentiment-analysis-flask-api`:

```
curl -X GET https://sentiment-analysis-flask-api.herokuapp.com/predict -H "Content-Type: application/json" -d '{"input":"the film was good"}'
```


Your application will now return a prediction from the model.
Congratulations, you have now learned how to train an LSTM model from
scratch, upload it to the cloud, and make predictions using it! Going
forward, this tutorial will hopefully serve as a basis for you to train
your own LSTM models and deploy them to the cloud yourself.


Deploying the application on Heroku
===================================


We have now trained our model on our local machine
and we can use this to make predictions. However, this isn\'t
necessarily any good if you want other people to
be able to use your model to make predictions. If we host our model on a
cloud-based platform, such as Heroku, and create a basic API, other
people will be able to make calls to the API to make predictions using
our model.



Introducing Heroku
------------------

**Heroku** is a cloud-based platform where you can
host your own basic programs. While the free tier of Heroku has a
maximum upload size of 500 MB and limited processing power, this should
be sufficient for us to host our model and create a basic API in order
to make predictions using our model.

The first step is to create a free account on Heroku and install the
Heroku app. Then, in the command line, type the following command:

```
heroku login
```


Log in using your account details. Then, create a new `heroku`
project by typing the following command:

```
heroku create sentiment-analysis-flask-api
```


Note that all the project names must be unique, so you will need to pick
a project name that isn\'t `sentiment-analysis-flask-api`.

Our first step is building a basic API using
Flask.



Creating an API using Flask -- file structure
---------------------------------------------

Creating an API is fairly simple using Flask as
Flask contains a default template required to make
an API:

First, in the command line, create a new folder
for your flask API and navigate to it:

```
mkdir flaskAPI
cd flaskAPI
```


Then, create a virtual environment within the folder. This will be the
Python environment that your API will use:

```
python3 -m venv vir_env
```


Within your environment, install all the packages that you will need
using `pip`. This includes all the packages that you use
within your model program, such as NLTK, `pandas`, NumPy, and
PyTorch, as well as the packages you will need to run the API, such as
Flask and Gunicorn:

```
pip install nltk pandas numpy torch flask gunicorn
```


We then create a list of requirements that our API will use. Note that
when we upload this to Heroku, Heroku will automatically download and
install all the packages within this list. We can do this by typing the
following:

```
pip freeze > requirements.txt
```


One adjustment we need to make is to replace the `torch` line
within the `requirements.txt` file with the following:

```
https://download.pytorch.org/whl/cpu/torch-1.3.1%2Bcpu-cp37-cp37m-linux_x86_64.whl
```


This is a link to the wheel file of the version of PyTorch that only
contains the CPU implementation. The full version of PyTorch that
includes full GPU support is over 500 MB in size, so it will not run on
the free Heroku cluster. Using this more compact version of PyTorch
means that you will still be able to run your model using PyTorch on
Heroku. Finally, we create three more files within our folder, as well
as a final directory for our models:

```
touch app.py
touch Procfile
touch wsgi.py
mkdir models
```


Now, we have created all the files we will need
for our Flash API and we are ready to start making
adjustments to our file.



Creating an API using Flask -- API file
---------------------------------------

Within our `app.py` file, we can begin
building our API:

1.  We first carry out all of our imports and
    create a `predict` route. This allows us to call our API
    with the `predict` argument in order to run a
    `predict()` method within our API:
    ```
    import flask
    from flask import Flask, jsonify, request
    import json
    import pandas as pd
    from string import punctuation
    import numpy as np
    import torch
    from nltk.tokenize import word_tokenize
    from torch.utils.data import TensorDataset, DataLoader
    from torch import nn
    from torch import optim
    app = Flask(__name__)
    @app.route('/predict', methods=['GET'])
    ```
    

2.  Next, we define our `predict()` method within our
    `app.py` file. This is largely a rehash of our model file,
    so to avoid repetition of code, it is advised that you look at the
    completed `app.py` file within the GitHub repository
    linked in the *Technical requirements* section of this chapter. You
    will see that there are a few additional
    lines. Firstly, within our `preprocess_review()` function,
    we will see the following lines:

    ```
    with open('models/word_to_int_dict.json') as handle:
    word_to_int_dict = json.load(handle)
    ```
    

    This takes the `word_to_int` dictionary we computed within
    our main model notebook and loads it into our
    model. This is so that our word indexing is consistent with our
    trained model. We then use this dictionary to convert our input text
    into an encoded sequence. Be sure to take the
    `word_to_int_dict.json` file from the original notebook
    output and place it within the `models` directory.

3.  Similarly, we must also load the weights from our trained model. We
    first define our `SentimentLSTM` class and the load our
    weights using `torch.load`. We will use the
    `.pkl` file from our original notebook, so be sure to
    place this in the `models` directory as well:
    ```
    model = SentimentLSTM(5401, 50, 100, 1, 2)
    model.load_state_dict(torch.load("models/model_nlp.pkl"))
    ```
    

4.  We must also define the input and outputs of our API. We want our
    model to take the input from our API and pass this to our
    `preprocess_review()` function. We do this using
    `request.get_json()`:
    ```
    request_json = request.get_json()
    i = request_json['input']
    words = np.array([preprocess_review(review=i)])
    ```
    

5.  To define our output, we return a JSON response consisting of the
    output from our model and a response code,
    `200`, which is what is returned by our predict function:
    ```
    output = model(x)[0].item()
    response = json.dumps({'response': output})
        return response, 200
    ```
    

6.  With the main body of our app complete, there
    are just two more additional things we must add in order to make our
    API run. We must first add the following to our `wsgi.py`
    file:
    ```
    from app import app as application
    if __name__ == "__main__":
        application.run()
    ```
    

7.  Finally, add the following to our Procfile:
    ```
    web: gunicorn app:app --preload
    ```
    

That\'s all that\'s required for the app to run. We can test that our
API runs by first starting the API locally using the following command:

```
gunicorn --bind 0.0.0.0:8080 wsgi:application -w 1
```


Once the API is running locally, we can make a request to the API by
passing it a sentence to predict the outcome:

```
curl -X GET http://0.0.0.0:8080/predict -H "Content-Type: application/json" -d '{"input":"the film was good"}'
```


If everything is working correctly, you should receive a prediction from
the API. Now that we have our API making predictions locally, it is time
to host it on Heroku so that we can make predictions in the cloud.



Creating an API using Flask -- hosting on Heroku
------------------------------------------------

We first need to commit our files to Heroku in a
similar way to how we would commit files using
GitHub. We define our working `flaskAPI` directory as a
`git` folder by simply running the following command:

```
git init
```


Within this folder, we add the following code to the
`.gitignore` file, which will stop us from adding unnecessary
files to the Heroku repo:

```
vir_env
__pycache__/
.DS_Store
```


Finally, we add our first `commit` function and push it to our
`heroku` project:

```
git add . -A 
git commit -m 'commit message here'
git push heroku master
```


This may take some time to compile as not only does the system have to
copy all the files from your local directory to Heroku, but Heroku will
automatically build your defined environment, installing all the
required packages and running your API.

Now, if everything has worked correctly, your API will automatically run
on the Heroku cloud. In order to make predictions, you can simply make a
request to the API using your project name instead of
`sentiment-analysis-flask-api`:

```
curl -X GET https://sentiment-analysis-flask-api.herokuapp.com/predict -H "Content-Type: application/json" -d '{"input":"the film was good"}'
```


Your application will now return a prediction from the model.
Congratulations, you have now learned how to train an LSTM model from
scratch, upload it to the cloud, and make predictions using it! Going
forward, this tutorial will hopefully serve as a basis for you to train
your own LSTM models and deploy them to the cloud yourself.


Summary
=======


In this chapter, we discussed the fundamentals of RNNs and one of their
main variations, LSTM. We then demonstrated how you can build your own
RNN from scratch and deploy it on the cloud-based platform Heroku. While
RNNs are often used for deep learning on NLP tasks, they are by no means
the only neural network architecture suitable for this task.

In the next chapter, we will look at convolutional neural networks and
show how they can be used for NLP learning tasks.