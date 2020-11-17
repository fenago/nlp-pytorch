
*[]{#_idTextAnchor124}Chapter 7*: Text Translation Using Sequence-to-Sequence Neural Networks {#_idParaDest-116}
=============================================================================================

::: {#_idContainer208}
In the previous two chapters, we used neural networks to classify text
and perform sentiment analysis. Both tasks involve taking an NLP input
and predicting some value. In the case of our sentiment analysis, this
was a number between 0 and 1 representing the sentiment of our sentence.
In the case of our sentence classification model, our output was a
multi-class prediction, of which there were several categories our
sentence belonged to. But what if we wish to make not just a single
prediction, but predict a whole sentence? In this chapter, we will build
a sequence-to-sequence model that takes a sentence in one language as
input and outputs the translation of this sentence in another language.

We have already explored several types of neural network architecture
used for NLP learning, namely recurrent neural networks in [*Chapter
5*](https://subscription.packtpub.com/book/data/9781789802740/8)*,
Recurrent Neural Networks and Sentiment Analysis*, and convolutional
neural networks in [*Chapter
6*](https://subscription.packtpub.com/book/data/9781789802740/9)*, Text
Classification Using CNNs*. In this chapter, we will again be using
these familiar RNNs, but instead of just building a simple RNN model, we
will use RNNs as part of a larger, more complex model in order to
perform sequence-to-sequence translation. By using the underpinnings of
RNNs that we learned about in the previous chapters, we can show how
these concepts can be extended in order to create a variety of models
that can be fit for purpose.

In this chapter, we will cover the following topics:

-   Theory of sequence-to-sequence models
-   Building a sequence-to-sequence neural network for text translation
-   Next steps


Technical requirements {#_idParaDest-117}
======================

::: {#_idContainer208}
All the code for this chapter can be found at
<https://github.com/PacktPublishing/Hands-On-Natural-Language-Processing-with-PyTorch-1.x>.


Theory of sequence-to-sequence models {#_idParaDest-118}
=====================================

::: {#_idContainer208}
Sequence-to-sequence models []{#_idIndexMarker362}are very similar to
the conventional neural network structures we have seen so far. The main
difference is that for a model\'s output, we expect another sequence,
rather than a binary or multi-class prediction. This is particularly
useful in tasks such as translation, where we may wish to convert a
whole sentence into another language.

In the following example, we can see that our English-to-Spanish
translation maps word to word:

<div>

::: {#_idContainer188 .IMG---Figure}
![Figure 7.1 -- English to Spanish
translation](3_files/B12365_07_01.jpg)
:::

</div>

Figure 7.1 -- English to Spanish translation

The first word in our input sentence maps nicely to the first word in
our output sentence. If this were the case for all languages, we could
simply pass each word in our sentence one by one through our trained
model to get an output sentence, and there would be no need for any
sequence-to-sequence modeling, as shown here:

<div>

::: {#_idContainer189 .IMG---Figure}
![Figure 7.2 -- English-to-Spanish translation of words
](3_files/B12365_07_02.jpg)
:::

</div>

Figure 7.2 -- English-to-Spanish translation of words

However, we know from[]{#_idIndexMarker363} our experience with NLP that
language is not as simple as this! Single words in one language may map
to multiple words in other languages, and the order in which these words
occur in a grammatically correct sentence may not be the same.
Therefore, we need a model that can capture the context of a whole
sentence and output a correct translation, not a model that aims to
directly translate individual words. This is where sequence-to-sequence
modeling becomes essential, as seen here:

<div>

::: {#_idContainer190 .IMG---Figure}
![Figure 7.3 -- Sequence-to-sequence modeling for translation
](3_files/B12365_07_03.jpg)
:::

</div>

Figure 7.3 -- Sequence-to-sequence modeling for translation

To train[]{#_idIndexMarker364} a sequence-to-sequence model that
captures the context of the input sentence and translates this into an
output sentence, we will essentially train two smaller models that allow
us to do this:

-   An **encoder** model, which[]{#_idIndexMarker365} captures the
    context of our sentence and outputs it as a single context vector
-   A **decoder**, which takes []{#_idIndexMarker366}the context vector
    representation of our original sentence and translates this into a
    different language

So, in reality, our full sequence-to-sequence translation model will
actually look something like this:

<div>

::: {#_idContainer191 .IMG---Figure}
![Figure 7.4 -- Full sequence-to-sequence model
](3_files/B12365_07_04.jpg)
:::

</div>

Figure 7.4 -- Full sequence-to-sequence model

By splitting our []{#_idIndexMarker367}models into individual encoder
and decoder elements, we are effectively modularizing our models. This
means that if we wish to train multiple models to translate from English
into different languages, we do not need to retrain the whole model each
time. We only need to train multiple different decoders to transform our
context vector into our output sentences. Then, when making predictions,
we can simply swap out the decoder that we wish to use for our
translation:

<div>

::: {#_idContainer192 .IMG---Figure}
![Figure 7.5 -- Detailed model layout ](3_files/B12365_07_05.jpg)
:::

</div>

Figure 7.5 -- Detailed model layout

Next, we will examine the encoder and decoder components of the
sequence-to-sequence model.

[]{#_idTextAnchor127}

Encoders {#_idParaDest-119}
--------

The purpose of the[]{#_idIndexMarker368} encoder element of our
sequence-to-sequence model is to be able to fully capture the context of
our input sentence and represent it as a vector. We can do this by using
RNNs or, more specifically, LSTMs. As you may recall from our previous
chapters, RNNs take a sequential input and maintain a hidden state
throughout this sequence. Each new word in the sequence updates the
hidden state. Then, at the end of the sequence, we can use the model\'s
final hidden state as our input into our next layer.

In the case of our encoder, the hidden state represents the context
vector representation of our whole sentence, meaning we can use the
hidden state output of our RNN to represent the entirety of the input
sentence:

<div>

::: {#_idContainer193 .IMG---Figure}
![Figure 7.6 -- Examining the encoder ](3_files/B12365_07_06.jpg)
:::

</div>

Figure 7.6 -- Examining the encoder

We use our final hidden state, *h*[n]{.subscript}, as our context
vector, which we will then decode using a trained decoder. It
is[]{#_idIndexMarker369} also worth observing that in the context of our
sequence-to-sequence models, we append \"start\" and \"end\" tokens to
the beginning and end of our input sentence, respectively. This is
because our inputs and outputs do not have a finite length and our model
needs to be able to learn when a sentence should end. Our input sentence
will always end with an \"end\" token, which signals to the encoder that
the hidden state, at this point, will be used as the final context
vector representation for this input sentence. Similarly, in the decoder
step, we will see that our decoder will keep generating words until it
predicts an \"end\" token. This allows our decoder to generate actual
output sentences, as opposed to a sequence of tokens of infinite length.

Next, we will look at how the decoder takes this context vector and
learns to translate it into an output sentence.

[]{#_idTextAnchor128}

Decoders {#_idParaDest-120}
--------

Our decoder takes the []{#_idIndexMarker370}final hidden state from our
encoder layer and decodes this into a sentence in another language. Our
decoder is an RNN, similar to that of our encoder, but while our encoder
updates its hidden state given its current hidden state and the current
word in the sentence, our decoder updates its hidden state and outputs a
token at each iteration, given the current hidden state and the previous
predicted word in the sentence. This can be seen in the following
diagram:

<div>

::: {#_idContainer194 .IMG---Figure}
![Figure 7.7 -- Examining the decoder ](3_files/B12365_07_07.jpg)
:::

</div>

Figure 7.7 -- Examining the decoder

First, our model takes the []{#_idIndexMarker371}context vector as the
final hidden state from our encoder step, *h0*. Our model then aims to
predict the next word in the sentence, given the current hidden state,
and then the previous word in the sentence. We know our sentence must
begin with a \"start\" token so, at our first step, our model tries to
predict the first word in the sentence given the previous hidden state,
*h0*, and the previous word in the sentence (in this instance, the
\"start\" token). Our model makes a prediction (\"pienso\") and then
updates the hidden state to reflect the new state of the model, *h1*.
Then, at the next step, our model uses the new hidden state and the last
predicted word to predict the next word in the sentence. This continues
until the model predicts the \"end\" token, at which point our model
stops generating output words.

The intuition behind this model is in line with what we have learned
about language representations thus far. Words in any given sentence are
dependent on the words that come before it. So, to predict any given
word in a sentence without considering the words that have been
predicted before it, this would not make sense as words in any given
sentence are not independent from one another.

We learn our model parameters as we have done previously: by making a
forward pass, calculating the loss of our target sentence against the
predicted sentence, and backpropagating this loss through the network,
updating the parameters as we go. However, learning using this process
can be very slow because, to begin with, our model will have very little
predictive power. Since our predictions for the words in our target
sentence are not independent[]{#_idIndexMarker372} of one another, if we
predict the first word in our target sentence incorrectly, subsequent
words in our output sentence are also unlikely to be correct. To help
with this process, we can use a technique known as **teacher forcing**.

[]{#_idTextAnchor129}

Using teacher forcing {#_idParaDest-121}
---------------------

As our model does not make good []{#_idIndexMarker373}predictions
initially, we will find that any initial errors
[]{#_idIndexMarker374}are multiplied exponentially. If our first
predicted word in the sentence is incorrect, then the rest of the
sentence will likely be incorrect as well. This is because the
predictions our model makes are dependent on the previous predictions it
makes. This means that any losses our model has can be multiplied
exponentially. Due to this, we may face the exploding gradient problem,
making it very difficult for our model to learn anything:

<div>

::: {#_idContainer195 .IMG---Figure}
![Figure 7.8 -- Using teacher forcing ](3_files/B12365_07_08.jpg)
:::

</div>

Figure 7.8 -- Using teacher forcing

However, by using **teacher forcing**, we train our model using the
correct previous target word so that one wrong prediction does not
inhibit our model\'s ability to learn from the correct predictions. This
means that if our model makes an incorrect prediction at one point in
the sentence, it can still make correct predictions using subsequent
words. While our model will still have incorrectly predicted words and
will have losses by which we can update our gradients, now, we do not
suffer from exploding[]{#_idIndexMarker375} gradients, and our model
will learn much more quickly:

<div>

::: {#_idContainer196 .IMG---Figure}
![Figure 7.9 -- Updating for losses ](3_files/B12365_07_09.jpg)
:::

</div>

Figure 7.9 -- Updating for losses

You can consider teacher []{#_idIndexMarker376}forcing as a way of
helping our model learn independently of its previous predictions at
each time step. This is so the losses that are incurred by a
mis-prediction at an early time step are not carried over to later time
steps.

By combining the encoder and decoder steps and applying teacher forcing
to help our model learn, we can build a sequence-to-sequence model that
will allow us to translate sequences of one language into another. In
the next section, we will illustrate how we can build this from scratch
using PyTorch.


Building a sequence-to-sequence model for text translation {#_idParaDest-122}
==========================================================

::: {#_idContainer208}
In order to[]{#_idIndexMarker377} build our sequence-to-sequence model
for translation, we will implement the encoder/decoder framework we
outlined previously. This will show how the two halves of our model can
be utilized together in order to capture a representation of our data
using the encoder and then translate this representation into another
language using our decoder. In order to do this, we need to obtain our
data.

[]{#_idTextAnchor131}

Preparing the data {#_idParaDest-123}
------------------

By now, we[]{#_idIndexMarker378} know enough about machine learning to
know that for a task like this, we will need a set of training data with
corresponding labels. In this case, we will need **sentences in one
language with the corresponding translations in another language**.
Fortunately, the `Torchtext`{.literal} library that we used in the
previous chapter contains a dataset that will allow us to get this.

The `Multi30k`{.literal} dataset in `Torchtext`{.literal} consists of
approximately 30,000 sentences with corresponding translations in
multiple languages. For this translation task, our input sentences will
be in English and our output sentences will be in German. Our fully
trained model will, therefore, allow us to **translate English sentences
into German**.

We will start by extracting our data and preprocessing it. We will once
again use `spacy`{.literal}, which contains a built-in dictionary of
vocabulary that we can use to tokenize our data:

1.  We start by loading our `spacy`{.literal} tokenizers into Python. We
    will need to do this once for each language we are using since we
    will be building two entirely separate vocabularies for this task:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    spacy_german = spacy.load(‘de’)
    spacy_english = spacy.load(‘en’)
    ```
    :::

    Important note

    You may have to install the German vocabulary from the command line
    by doing the following (we installed the English vocabulary in the
    previous chapter):**python3 -m spacy download de**

2.  Next, we create a function for each of our languages to tokenize our
    sentences. Note that our tokenizer for our input English sentence
    reverses the order of the tokens:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def tokenize_german(text):
        return [token.text for token in spacy_german.            tokenizer(text)]
    def tokenize_english(text):
        return [token.text for token in spacy_english.            tokenizer(text)][::-1]
    ```
    :::

    While[]{#_idIndexMarker379} reversing the order of our input
    sentence is not compulsory, it has been shown to improve the model's
    ability to learn. If our model consists of two RNNs joined together,
    we can show that the information flow within our model is improved
    when reversing the input sentence. For example, let's take a basic
    input sentence in English but not reverse it, as follows:

    ::: {#_idContainer197 .IMG---Figure}
    ![Figure 7.10 -- Reversing the input words
    ](data:application/octet-stream;base64,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irsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdiq10VxRlBHgRXFUum0TTrj+9sLd/wDWhQ/rGKpFd/l75WvgRc+W9Nlr152MDfrTFWLXn5C/ltqFTc+QtDlr/Npdsf8AmXiqTSf840flXJ/0wWkp/wAY7VY/+IUxVBSf84tflZJ/0yECf8Y57mP/AIhMMVQT/wDOJ35XP9ny/PGf+K9Y1NPwW7AxVDH/AJxN/L5f7iHVoP8AjHruo7f8FcHFWx/zir5NX7N/r6/LX77/AKq4qqj/AJxe8qr9nV/MS/LzBe/9VMVVB/zjN5cX7PmDzKvy8wXn/NeKqo/5xu0Rfs+aPNK/LzDd/wDNWKrj/wA45aQ3XzZ5rPz8xXf/ADViqmf+catBb7XmTzO3z8w3f/NWKqR/5xi8st9rXPMjfPzBef8ANeKqR/5xa8ot9rVPMDfPX73/AKqYqs/6FS8kN/eXOuSDwbXr+n4TDFV6f84l/lr1l0y/m/4ya7qh/VdDFUZH/wA4p/lbH/0zBf8A4yajfyf8TuTiqNj/AOcYPysj/wCmKsX/AOMnqSf8Tc4qjov+cb/yshII/L7Q2I6F9Ogc/wDDqcVZDafkz5BsKfVvJejRU6cNNtx+qPFWSW/kry9abQaDYRf6lnCv6kxVNYtIsIP7uygT/ViQfqGKo9UVBRQAPACmKrsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdir//2Q==)
    :::

    Figure 7.10 -- Reversing the input words

    Here, we can see that in order to predict the first output word,
    *y0*, correctly, our first English word from *x0* must travel
    through three RNN layers before the prediction is made. In terms of
    learning, this means that our gradients must be backpropagated
    through three RNN layers, while maintaining the flow of information
    through the network. Now, let's compare this to a situation
    []{#_idIndexMarker380}where we reverse our input sentence:

    ::: {#_idContainer198 .IMG---Figure}
    ![Figure 7.11 -- Reversing the input sentence
    ](4_files/B12365_07_101.jpg)
    :::

    Figure 7.11 -- Reversing the input sentence

    We can now see that the distance between the true first word in our
    input sentence and the corresponding word in the output sentence is
    just one RNN layer. This means that the gradients only need to be
    backpropagated to one layer, meaning the flow of information and the
    ability to learn is much greater for our network compared to when
    the distance between these two words was three layers.

    If we were to calculate the total distances between the input words
    and their output counterparts for the reversed and non-reversed
    variants, we would see that they are the same. However, we have seen
    previously that the most important word in our output sentence is
    the first one. This is because the words in our output sentences are
    dependent on the words that come before them. If we were to predict
    the first word in the output sentence incorrectly, then chances are
    the rest of the words in our sentences would be predicted
    incorrectly too. However, by predicting the first word correctly, we
    maximize our chances of predicting the whole sentence correctly.
    Therefore, by minimizing the distance between the first word in our
    output sentence and its input counterpart, we can increase our
    model's ability to learn this relationship. This increases the
    chances of this prediction being correct, thus maximizing the
    chances of our entire output sentence being predicted correctly.

3.  With our []{#_idIndexMarker381}tokenizers constructed, we now need
    to define the fields for our tokenization. Notice here how we append
    start and end tokens to our sequences so that our model knows when
    to begin and end the sequence's input and output. We also convert
    all our input sentences into lowercase for the sake of simplicity:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    SOURCE = Field(tokenize = tokenize_english, 
                init_token = ‘<sos>’, 
                eos_token = ‘<eos>’, 
                lower = True)
    TARGET = Field(tokenize = tokenize_german, 
                init_token = ‘<sos>’, 
                eos_token = ‘<eos>’, 
                lower = True)
    ```
    :::

4.  With our fields defined, our tokenization becomes a simple
    one-liner. The dataset containing 30,000 sentences has built-in
    training, validation, and test sets that we can use for our model:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    train_data, valid_data, test_data = Multi30k.splits(exts = (‘.en’, ‘.de’), fields = (SOURCE, TARGET))
    ```
    :::

5.  We can examine individual sentences using the `examples`{.literal}
    property of our dataset objects. Here, we can see that the source
    (`src`{.literal}) property contains our
    reversed[]{#_idIndexMarker382} input sentence in English and that
    our target (`trg`{.literal}) contains our non-reversed output
    sentence in German:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    print(train_data.examples[0].src)
    print(train_data.examples[0].trg)
    ```
    :::

    This gives us the following output:

    ::: {#_idContainer199 .IMG---Figure}
    ![Figure 7.12 -- Training data examples ](4_files/B12365_07_12.jpg)
    :::

    Figure 7.12 -- Training data examples

6.  Now, we can examine the size of each of our datasets. Here, we can
    see that our training dataset consists of 29,000 examples and that
    each of our validation and test sets consist of 1,014 and 1,000
    examples, respectively. In the past, we have used 80%/20% splits for
    the training and validation data. However, in instances like this,
    where our input and output fields are very sparse and our training
    set is of a limited size, it is often beneficial to train on as much
    data as there is available:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    print(“Training dataset size: “ + str(len(train_data.       examples)))
    print(“Validation dataset size: “ + str(len(valid_data.       examples)))
    print(“Test dataset size: “ + str(len(test_data.       examples)))
    ```
    :::

    This returns the following output:

    ::: {#_idContainer200 .IMG---Figure}
    ![](4_files/B12365_07_13.jpg)
    :::

    Figure 7.13 -- Data sample lengths

7.  Now, we []{#_idIndexMarker383}can build our vocabularies and check
    their size. Our vocabularies should consist of every unique word
    that was found within our dataset. We can see that our German
    vocabulary is considerably larger than our English vocabulary. Our
    vocabularies are significantly smaller than the true size of each
    vocabulary for each language (every word in the English dictionary).
    Therefore, since our model will only be able to accurately translate
    words it has seen before, it is unlikely that our model will be able
    to generalize well to all possible sentences in the English
    language. This is why training models like this accurately requires
    extremely large NLP datasets (such as those Google has access to):

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    SOURCE.build_vocab(train_data, min_freq = 2)
    TARGET.build_vocab(train_data, min_freq = 2)
    print(“English (Source) Vocabulary Size: “ +        str(len(SOURCE.vocab)))
    print(“German (Target) Vocabulary Size: “ +        str(len(TARGET.vocab)))
    ```
    :::

    This gives the following output:

    ::: {#_idContainer201 .IMG---Figure}
    ![Figure 7.14 -- Vocabulary size of the dataset
    ](4_files/B12365_07_14.jpg)
    :::

    Figure 7.14 -- Vocabulary size of the dataset

8.  Finally, we can []{#_idIndexMarker384}create our data iterators from
    our datasets. As we did previously, we specify the usage of a
    CUDA-enabled GPU (if it is available on our system) and specify our
    batch size:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    device = torch.device(‘cuda’ if torch.cuda.is_available()                       else ‘cpu’)
    batch_size = 32
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = batch_size, 
        device = device)
    ```
    :::

Now that our data has been preprocessed, we can start building the model
itself.

[]{#_idTextAnchor132}

Building the encoder {#_idParaDest-124}
--------------------

Now, we are ready to start []{#_idIndexMarker385}building our encoder:

1.  First, we begin by initializing our model by inheriting from our
    `nn.Module`{.literal} class, as we've done with all our previous
    models. We initialize with a couple of parameters, which we will
    define later, as well as the number of dimensions in the hidden
    layers within our LSTM layers and the number of LSTM layers:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    class Encoder(nn.Module):
        def __init__(self, input_dims, emb_dims, hid_dims,     n_layers, dropout):
            super().__init__()   
            self.hid_dims = hid_dims
            self.n_layers = n_layers
    ```
    :::
2.  Next, we define our embedding layer within our encoder, which is the
    length of the number of input dimensions and the depth of the number
    of embedding dimensions:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    self.embedding = nn.Embedding(input_dims, emb_dims)
    ```
    :::
3.  Next, we define our actual LSTM layer. This takes our embedded
    sentences from the embedding[]{#_idIndexMarker386} layer, maintains
    a hidden state of a defined length, and consists of a number of
    layers (which we will define later as 2). We also implement
    `dropout`{.literal} to apply regularization to our network:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    self.rnn = nn.LSTM(emb_dims, hid_dims, n_layers, dropout                    = dropout)
    self.dropout = nn.Dropout(dropout)
    ```
    :::
4.  Then, we define the forward pass within our encoder. We apply the
    embeddings to our input sentences and apply dropout. Then, we pass
    these embeddings through our LSTM layer, which outputs our final
    hidden state. This will be used by our decoder to form our
    translated sentence:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (h, cell) = self.rnn(embedded)
        return h, cell
    ```
    :::

Our encoders will consist of two LSTM layers, which means that our
output will output two hidden states. This also means that our full LSTM
layer, along with our encoder, will look something[]{#_idIndexMarker387}
like this, with our model outputting two hidden states:

<div>

::: {#_idContainer202 .IMG---Figure}
![Figure 7.15 -- LSTM model with an encoder](4_files/B12365_07_15.jpg)
:::

</div>

Figure 7.15 -- LSTM model with an encoder

Now that we have built our encoder, let\'s start building our decoder.

[]{#_idTextAnchor133}

Building the decoder {#_idParaDest-125}
--------------------

Our decoder will[]{#_idIndexMarker388} take the final hidden states from
our encoder\'s LSTM layer and translate them into an output sentence in
another language. We start by initializing our decoder in almost exactly
the same way as we did for the encoder. The only difference here is that
we also add a fully connected linear layer. This layer will use the
final hidden states from our LSTM in order to make predictions regarding
the correct word in the sentence:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
class Decoder(nn.Module):
    def __init__(self, output_dims, emb_dims, hid_dims,     n_layers, dropout):
        super().__init__()
        
        self.output_dims = output_dims
        self.hid_dims = hid_dims
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dims, emb_dims)
        
        self.rnn = nn.LSTM(emb_dims, hid_dims, n_layers,                           dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dims, output_dims)
        
        self.dropout = nn.Dropout(dropout)
```
:::

Our forward pass is incredibly similar to that of our encoder, except
with the addition of two key steps. We first unsqueeze our input from
the previous layer so that it\'s the correct size[]{#_idIndexMarker389}
for entry into our embedding layer. We also add a fully connected layer,
which takes the output hidden layer of our RNN layers and uses it to
make a prediction regarding the next word in the sequence:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
def forward(self, input, h, cell):
                
    input = input.unsqueeze(0)
                
    embedded = self.dropout(self.embedding(input))
                
    output, (h, cell) = self.rnn(embedded, (h, cell))
        
    pred = self.fc_out(output.squeeze(0))
        
    return pred, h, cell
```
:::

Again, similar to our encoder, we use a two-layer LSTM layer within our
decoder. We take our final hidden state from our encoders and use these
to generate the first word in our sequence, Y[1]{.subscript}. We then
[]{#_idIndexMarker390}update our hidden state and use this and
Y[1]{.subscript} to generate our next word, Y[2]{.subscript}, repeating
this process until our model generates an end token. Our decoder looks
something like this:

<div>

::: {#_idContainer203 .IMG---Figure}
![Figure 7.16 -- LSTM model with a decoder ](4_files/B12365_07_16.jpg)
:::

</div>

Figure 7.16 -- LSTM model with a decoder

Here, we can see that defining the encoders and decoders individually is
not particularly complicated. However, when we combine these steps into
one larger sequence-to-sequence model, things begin to get interesting:

[]{#_idTextAnchor134}

Constructing the full sequence-to-sequence model {#_idParaDest-126}
------------------------------------------------

We []{#_idIndexMarker391}must now stitch the two halves of our model
together to produce the full sequence-to-sequence model:

1.  We start by creating a new sequence-to-sequence class. This will
    allow us to pass our encoder and decoder to it as arguments:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, device):
            super().__init__()
            
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
    ```
    :::

2.  Next, we create the `forward`{.literal} method within our
    `Seq2Seq`{.literal} class. This is arguably the most complicated
    part of the model. We combine our encoder with our decoder and use
    teacher forcing to help our model learn. We start by creating a
    tensor in which we still store our predictions. We initialize this
    as a tensor full of zeroes, but we still update this with our
    predictions as we make them. The shape of our tensor of zeroes will
    be the length of our target sentence, the width of our batch size,
    and the depth of our target (German) vocabulary size:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def forward(self, src, trg, teacher_forcing_rate = 0.5):
        batch_size = trg.shape[1]
        target_length = trg.shape[0]
        target_vocab_size = self.decoder.output_dims
            
         outputs = torch.zeros(target_length, batch_size,                     target_vocab_size).to(self.device)
    ```
    :::

3.  Next, we feed our []{#_idIndexMarker392}input sentence into our
    encoder to get the output hidden states:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    h, cell = self.encoder(src)
    ```
    :::

4.  Then, we must loop through our decoder model to generate an output
    prediction for each step in our output sequence. The first element
    of our output sequence will always be the `<start>`{.literal} token.
    Our target sequences already contain this as the first element, so
    we just set our initial input equal to this by taking the first
    element of the list:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    input = trg[0,:]
    ```
    :::

5.  Next, we loop through and make our predictions. We pass our hidden
    states (from the output of our encoder) to our decoder, along with
    our initial input (which is just the `<start>`{.literal} token).
    This returns a prediction for all the words in our sequence.
    However, we are only interested in the word within our current step;
    that is, the next word in the sequence. Note how we start our loop
    from 1 instead of 0, so our first prediction is the second word in
    the sequence (as the first word that's predicted will always be the
    start token).

6.  This output consists of a vector of the target vocabulary's length,
    with a prediction for each word within the vocabulary. We take the
    `argmax`{.literal} function to identify the actual word that is
    predicted by the model.

    We then need to select our new input for the next step. We set our
    teacher forcing ratio to 50%, which means that 50% of the time, we
    will use the prediction we just made as our next input into our
    decoder and that the other 50% of the time, we will take the true
    target. As we discussed previously, this helps our model learn much
    more rapidly than relying on just the model's predictions.

    We then continue this loop []{#_idIndexMarker393}until we have a
    full prediction for each word in the sequence:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    for t in range(1, target_length):
    output, h, cell = self.decoder(input, h, cell)
                
    outputs[t] = output
                
    top = output.argmax(1) 
            
    input = trg[t] if (random.random() < teacher_forcing_                   rate) else top
            
    return outputs
    ```
    :::

7.  Finally, we create an instance of our Seq2Seq model that's ready to
    be trained. We initialize an encoder and a decoder with a selection
    of hyperparameters, all of which can be changed to slightly alter
    the model:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    input_dimensions = len(SOURCE.vocab)
    output_dimensions = len(TARGET.vocab)
    encoder_embedding_dimensions = 256
    decoder_embedding_dimensions = 256
    hidden_layer_dimensions = 512
    number_of_layers = 2
    encoder_dropout = 0.5
    decoder_dropout = 0.5
    ```
    :::

8.  We then pass our encoder[]{#_idIndexMarker394} and decoder to our
    `Seq2Seq`{.literal} model in order to create the complete model:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    encod = Encoder(input_dimensions,\
                    encoder_embedding_dimensions,\
                    hidden_layer_dimensions,\
                    number_of_layers, encoder_dropout)
    decod = Decoder(output_dimensions,\
                    decoder_embedding_dimensions,\
                    hidden_layer_dimensions,\
                    number_of_layers, decoder_dropout)
    model = Seq2Seq(encod, decod, device).to(device)
    ```
    :::

Try experimenting with different parameters here and see how it affects
the performance of the model. For instance, having a larger number of
dimensions in your hidden layers may cause the model to train slower,
although the overall final performance of the model may be better.
Alternatively, the model may overfit. Often, it is a matter of
experimenting to find the best-performing model.

After fully defining our Seq2Seq model, we are now ready to begin
training it.

[]{#_idTextAnchor135}

Training the model {#_idParaDest-127}
------------------

Our model will begin []{#_idIndexMarker395}initialized with weights of 0
across all parts of the model. While the model should theoretically be
able to learn with no (zero) weights, it has been shown that
initializing with random weights can help the model learn faster. Let\'s
get started:

1.  Here, we will initialize our model with the weights of random
    samples from a normal distribution, with the values being between
    -0.1 and 0.1:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def initialize_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.1, 0.1)
            
    model.apply(initialize_weights)
    ```
    :::

2.  Next, as with all our other models, we define our optimizer and loss
    functions. We're using cross-entropy loss as we are performing
    multi-class classification (as opposed to binary cross-entropy loss
    for a binary classification):
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = TARGET.               vocab.stoi[TARGET.pad_token])
    ```
    :::

3.  Next, we define the training process within a function called
    `train()`{.literal}. First, we set our model to train mode and set
    the epoch loss to `0`{.literal}:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def train(model, iterator, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
    ```
    :::

4.  We then loop through each batch within our training iterator and
    extract the sentence to be translated (`src`{.literal}) and the
    correct translation of this sentence (`trg`{.literal}). We then zero
    our gradients (to prevent gradient accumulation) and calculate the
    output of our model by passing our model function our inputs and
    outputs:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    for i, batch in enumerate(iterator):
    src = batch.src
    trg = batch.trg
    optimizer.zero_grad()
    output = model(src, trg)
    ```
    :::

5.  Next, we need to []{#_idIndexMarker396}calculate the loss of our
    model's prediction by comparing our predicted output to the true,
    correct translated sentence. We reshape our output data and our
    target data using the shape and view functions in order to create
    two tensors that can be compared to calculate the loss. We calculate
    the `loss`{.literal} criterion between our output and
    `trg`{.literal} tensors and then backpropagate this loss through the
    network:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    output_dims = output.shape[-1]
    output = output[1:].view(-1, output_dims)
    trg = trg[1:].view(-1)
            
    loss = criterion(output, trg)
            
    loss.backward()
    ```
    :::

6.  We then implement gradient clipping to prevent exploding gradients
    within our model, step our optimizer in order to perform the
    necessary parameter updates via gradient descent, and finally add
    the loss of the batch to the epoch loss. This
    whole[]{#_idIndexMarker397} process is repeated for all the batches
    within a single training epoch, whereby the final averaged loss per
    batch is returned:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
    optimizer.step()
            
    epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)
    ```
    :::

7.  After, we create a similar function called `evaluate()`{.literal}.
    This function will calculate the loss of our validation data across
    the network in order to evaluate how our model performs when
    translating data it hasn't seen before. This function is almost
    identical to our `train()`{.literal} function, with the exception of
    the fact that we switch to evaluation mode:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    model.eval()
    ```
    :::

8.  Since we don't perform any updates for our weights, we need to make
    sure to implement `no_grad`{.literal} mode:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    with torch.no_grad():
    ```
    :::

9.  The only other difference is that we need to make sure we turn off
    teacher forcing when in evaluation mode. We wish to assess our
    model's performance on unseen data, and enabling teacher forcing
    would use our correct (target) data to help our model make better
    predictions. We want our model to be able to[]{#_idIndexMarker398}
    make perfect, unaided predictions:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    output = model(src, trg, 0)
    ```
    :::

10. Finally, we need to create a training loop, within which our
    `train()`{.literal} and `evaluate()`{.literal} functions are called.
    We begin by defining how many epochs we wish to train for and our
    maximum gradient (for use with gradient clipping). We also set our
    lowest validation loss to infinity. This will be used later to
    select our best-performing model:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    epochs = 10
    grad_clip = 1
    lowest_validation_loss = float(‘inf’)
    ```
    :::

11. We then loop through each of our epochs and within each epoch,
    calculate our training and validation loss using our
    `train()`{.literal} and `evaluate()`{.literal} functions. We also
    time how long this takes by calling `time.time()`{.literal} before
    and after the training process:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    for epoch in range(epochs):
        
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer,                       criterion, grad_clip)
        valid_loss = evaluate(model, valid_iterator,                          criterion)
        
        end_time = time.time()
    ```
    :::

12. Next, for each epoch, we determine whether the model we just trained
    is the best-performing model we have seen thus far. If our model
    performs the best on our validation []{#_idIndexMarker399}data (if
    the validation loss is the lowest we have seen so far), we save our
    model:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    if valid_loss < lowest_validation_loss:
    lowest_validation_loss = valid_loss
    torch.save(model.state_dict(), ‘seq2seq.pt’) 
    ```
    :::

13. Finally, we simply print our output:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    print(f’Epoch: {epoch+1:02} | Time: {np.round(end_time-start_time,0)}s’)
    print(f’\tTrain Loss: {train_loss:.4f}’)
    print(f’\t Val. Loss: {valid_loss:.4f}’)
    ```
    :::

    If our training is working correctly, we should see the training
    loss decrease over time, like so:

<div>

::: {#_idContainer204 .IMG---Figure}
![Figure 7.17 -- Training the model ](4_files/B12365_07_17.jpg)
:::

</div>

Figure 7.17 -- Training the model

Here, we can see that[]{#_idIndexMarker400} both our training and
validation loss appear to be falling over time. We can continue to train
our model for a number of epochs, ideally until the validation loss
reaches its lowest possible value. Now, we can evaluate our
best-performing model to see how well it performs when making actual
translations.

[]{#_idTextAnchor136}

Evaluating the model {#_idParaDest-128}
--------------------

In order to evaluate []{#_idIndexMarker401}our model, we will take our
test set of data and run our English sentences through our model to
obtain a prediction of the translation in German. We will then be able
to compare this to the true prediction in order to see if our model is
making accurate predictions. Let\'s get started!

1.  We start by creating a `translate()`{.literal} function. This is
    functionally identical to the `evaluate()`{.literal} function we
    created to calculate the loss over our validation set. However, this
    time, we are not concerned with the loss of our model, but rather
    the predicted output. We pass the model our source and target
    sentences and also make sure we turn teacher forcing off so that our
    model does not use these to make predictions. We then take our
    model's predictions and use []{#_idIndexMarker402}an
    `argmax`{.literal} function to determine the index of the word that
    our model predicted for each word in our predicted output sentence:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    output = model(src, trg, 0)
    preds = torch.tensor([[torch.argmax(x).item()] for x         in output])
    ```
    :::

2.  Then, we can use this index to obtain the actual predicted word from
    our German vocabulary. Finally, we compare the English input to our
    model that contains the correct German sentence and the predicted
    German sentence. Note that here, we use `[1:-1]`{.literal} to drop
    the start and end tokens from our predictions and we reverse the
    order of our English input (since the input sentences were reversed
    before they were fed into the model):

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    print(‘English Input: ‘ + str([SOURCE.vocab.itos[x] for x        in src][1:-1][::-1]))
    print(‘Correct German Output: ‘ + str([TARGET.vocab.       itos[x] for x in trg][1:-1]))
    print(‘Predicted German Output: ‘ + str([TARGET.vocab.       itos[x] for x in preds][1:-1]))
    ```
    :::

    By doing this, we can compare our predicted output with the correct
    output to assess if our model is able to make accurate predictions.
    We can see from our model's predictions that our model is able to
    translate English sentences into German, albeit far from perfectly.
    Some of our model's predictions are exactly the same as the target
    data, showing that our model translated these sentences perfectly:

<div>

::: {#_idContainer205 .IMG---Figure}
![Figure 7.18 -- Translation output part one ](4_files/B12365_07_18.jpg)
:::

</div>

Figure 7.18 -- Translation output part one

In other instances, our model is off by a single word. In this case, our
model predicts the word `hüten`{.literal} instead of `mützen`{.literal};
however, `hüten`{.literal} is actually an
acceptable[]{#_idIndexMarker403} translation of `mützen`{.literal},
though the words may not be semantically identical:

<div>

::: {#_idContainer206 .IMG---Figure}
![Figure 7.19 -- Translation output part two ](4_files/B12365_07_19.jpg)
:::

</div>

Figure 7.19 -- Translation output part two

We can also see examples that seem to have been mistranslated. In the
following example, the English equivalent of the German sentence that we
predicted is "`A woman climbs through one`{.literal}", which is not
equivalent to "`Young woman climbing rock face`{.literal}". However, the
model has still managed to translate key elements of the English
sentence (woman and climbing):

<div>

::: {#_idContainer207 .IMG---Figure}
![Figure 7.20 -- Translation output part three
](data:application/octet-stream;base64,/9j/4AAQSkZJRgABAgEA8wDzAAD/7QAsUGhvdG9zaG9wIDMuMAA4QklNA+0AAAAAABAA8wAAAAEAAQDzAAAAAQAB/+4AE0Fkb2JlAGSAAAAAAQUAAklE/9sAhAACAgIDAgMDAwMDBQQEBAUFBQUFBQUHBgYGBgYHCAcICAgIBwgJCgoKCgoJCwwMDAwLDAwMDAwMDAwMDAwMDAwMAQMEBAoFCg8KCg8PDg4ODw8ODg4ODw8MDg4ODA8PDBEREREMDwwREREREQwRERERERERERERERERERERERERERH/wAARCAB3BMkDAREAAhEBAxEB/8QBogAAAAcBAQEBAQAAAAAAAAAABAUDAgYBAAcICQoLAQACAgMBAQEBAQAAAAAAAAABAAIDBAUGBwgJCgsQAAIBAwMCBAIGBwMEAgYCcwECAxEEAAUhEjFBUQYTYSJxgRQykaEHFbFCI8FS0eEzFmLwJHKC8SVDNFOSorJjc8I1RCeTo7M2F1RkdMPS4ggmgwkKGBmElEVGpLRW01UoGvLj88TU5PRldYWVpbXF1eX1ZnaGlqa2xtbm9jdHV2d3h5ent8fX5/c4SFhoeIiYqLjI2Oj4KTlJWWl5iZmpucnZ6fkqOkpaanqKmqq6ytrq+hEAAgIBAgMFBQQFBgQIAwNtAQACEQMEIRIxQQVRE2EiBnGBkTKhsfAUwdHhI0IVUmJy8TMkNEOCFpJTJaJjssIHc9I14kSDF1STCAkKGBkmNkUaJ2R0VTfyo7PDKCnT4/OElKS0xNTk9GV1hZWltcXV5fVGVmZ2hpamtsbW5vZHV2d3h5ent8fX5/c4SFhoeIiYqLjI2Oj4OUlZaXmJmam5ydnp+So6SlpqeoqaqrrK2ur6/9oADAMBAAIRAxEAPwD7+Yq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq8F0D/nKb8nvNWr2mg6N+Z/ly/wBTu5hb29pbavayzTTE0CRqkhLMT0ArXtir3rFXkf5w/nt5G/IPTtM1bz3ri6TaalfQ6bayG3nn9S6mVnVONvHIyjijEswCgDdtxVV65irsVdirzPyv+c/kLzvruqeV/L3nLSdU1rSxMb7TrO/gnurYQSrDKZIo3LLwkYI1R8LEKaE4q9MxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KsQ8/efNC/K/y7q/mvzNffUtI0mBrm8ufSlm9KJaAt6cKO7dRsqk+2Kq3kjzro35j6Bo/mjy9efXNK1e1ivLK49KSL1YJlDI3CVUdag9GUEdxirKcVdirsVdirsVdirsVdirsVdirsVeS69+eXknyz588uflnqWtej5o8w2093plh9VuX9eGBJpJG9ZImhSiwSGjyKTx2BJFVXrWKuxV2KuxV2KvN/wA2fzc8qfkd5ZvfOHnXVP0Zo1m8Ec9yLee4KtPIsMY9O2jkkNXYDZTTqaDfFWcaRqtrrtjZalYy+ra3kEVxBJxZecUyB0ajAEVUg0IB8RiqYYq7FXYq7FXYq7FXYq7FXYq7FXYq8ni/PHyTN+YM35WJrVfNkWnDVX076rc7WZKr6nrmL0K1YfCJOf8Ak4q9YxV2KuxV2KuxV5L+cX55+SfyC0nT9c8961+irC/v4dMt5vqtzc87udJJEj42sUrCqxOeRAUcdzUiqr1rFXYq7FXjXnv/AJyI/K/8r9SGjebvzA0PRNRMST/VL/U7e3n9JyQrmORwwVqGhpvTFXqGha7p3mfTrHV9Ivob6wvoY7m1uraRZYZ4ZVDpJG6EqyspBBBoRiq7WtZsvLunX+ralcLb2dhbzXVzMwJWOGFDJI5CgmiqpOwrirCfyl/N3yn+eXlmy84eStV/SejXjzxw3HoTW5Z4JGikBjuI45FoykbqK9RUEHFXpOKuxV5n5k/OfyF5P8w6X5T1zzlpOna7qhgFjpl1fwxXlybmUww+nC7h29SQFEoPiYFVqcVemYq7FXkfl789vI3mrz35j/LTS9cW480eXraG71KwFvOvoQyiIq3qvGIn2ljqEdivIA0NaKvXMVdirsVdirsVeS/mR+eXkn8o9T8oaP5s1r6he+bdQTS9Gi+q3M/1q8d4o1j5QRSLH8UqDlIUX4vtUBoq9axV2KuxV2KuxViHn7z5oX5X+XdX81+Zr76lpGkwNc3lz6Us3pRLQFvThR3bqNlUn2xVW8keddG/MfQNH80eXrz65pWr2sV5ZXHpSRerBMoZG4SqjrUHoygjuMVZTirsVdirsVdirsVeb/mJ+cXkX8o47GTzt5v0ry+t80q2p1O+htfXMQUyemJWUtw5rypXjyWvUYqzrTNTtNas7TULC5jurS7ijnt54XEkU0Mqh0kR1JDKykEEGhBqMVR2KvJ4vzx8kzfmDN+Via1XzZFpw1V9O+q3O1mSq+p65i9CtWHwiTn/AJOKvWMVdirsVdirsVeS/nF+efkn8gtJ0/XPPetfoqwv7+HTLeb6rc3PO7nSSRI+NrFKwqsTnkQFHHc1Iqq9axVKtb13TfLVjc6pq+oW+n2NqhknuruZIIIkHVnkkKqoHiSBirzfyN/zkB+Wf5nalLo3lH8wNB13UIo2ma107Vba6m9JCA0gSKRiUUsAWAIFRU74q9dxV2KuxV2KuxV2KuxV5L+XP55+Sfza1bzjoflTWvr9/wCUL9tM1qH6rcw/VbtXljMfKeKNZPihkHKMuvw9aEVVetYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FX5gf8/VPy9hufyf1v8xrbXNZ07WPKqWMdithqcttayC/1SytpTPCho5CSNxNVINNyNsVfKeifkf8AkP5xm8qeTfKv5v8Am7zB5y8yaDqF9bJpPnO3vLO1vrKyW4aK8MRLQF3Y8FI34OGK03VfZOj/AJlfmV/ziL/zjD5L1zXvIN55q13RLQjX7SfWoraaxtFa4lNxNcOtyZPTURpwRXb4h0CnFXl83/Pyrz7d+U7f8zNJ/wCcctYn8hQxQSahrE+sQwyISVS4a3gFuzSwwyFlWbZJONX9H4gqr9G7T86vKVx+Xdv+acupC28sy6RHrZup1KtHaPCJhyQVPqUPHgKsX+AVNMVfBNr/AM/Jtcv7CTzna/8AOPXm6b8vI+cp8yIYzKbVGo1ytl6dDEFqzOLjgtDVtiQq9/8A+cM/+cqtd/5yy0HVPNF3+XEvlbRo5Vi0y8bWItQXUGV5UnARYIJIjCUUHkpVufwt8JxV8wf85WflD5F/Lb84f+cXL3yl5M0bQbi/85yC7l0vTLaye4CiBgJWgjQvQkkVruSe+Kv1txV+T/8AzmT+el3pWuaT5f8AzN/5xq/T3kqDzPpltpXmC68yxRwyXs6ssc6WkFs8gIRpf3bvxalHpUUVfrBir87/ADJ/zmf5680eefOXkj8lPycl88/4OuRZa3qdzrtro9rFejmGt4Rcp+8ZWjdCeQoy148KOVWS/wDOK3/OYHmD/nIHzj578k+aPywufJGqeU4LGS4gutQN3Iz3LSKQQbWABfhDRspdZEPIGlMVfFHn/wA9J/zjl/zmd5luPIv5ZS+YdX8x+SEt7bRtEjgshdaneahBeT3V1KEKxqUt5HlnZHYtxL7Esqr7X/5xu/5zG1X82PPPmH8rPzA/Lu58i+c9Is11IWMl6t9b3VkTGpkjnSONagypsvNWFSHqrqqr7rxV2Kvmb8/f+cV/LX/ORN1pF3r3mDzHpj6ZHLFEuiaxJYxusrKxMiBXVmHHZqA0232oq/NL/n3V/wA4w6P+bHlKx/MfzF5y83TatpPmW8jhij8wTJaSJp00bRCWPiWYE/bAcBhtsMVfcP8AzmR/zmLqn/OIa+XdWufy2uNf8t6hOlteavDq0NqbO4cuywrbtDI0rmKN3FWjQ04lweirxW8/5+JecPJOqeVtR/Mj8hdW8o+SPM1/Bp9hrt1qcUlzE1xyaJ7qyEKtESgLtGzh1RXKeqVoVXu//OZP/OW2s/8AOJNhoevj8uJfMegXcwt9Q1KPV4rIWErsBGnomCaSVnXkRQKvw0LAnFXguof8/KddstPHnMf848eb2/L4j1R5jYokhtC21z9UMXERMvxB2uAh2+LeuKvb/wA8/wDnNBfy8/KbRfzk8g+Tm89eW75UnuZ49Uj00WNs7CJXlEkE8hcTH0mjEfJHBD0piqefnr/zk15t/K78vfL/AOYnlX8qZ/N1heaVJrOrKmtQaeNIsktEuy8jSQTNL8LNskf7BJ3Kgqp/H/zk9y/J3yt+bdv5B1/Wf05Z6dc/oPy9bfpS/hN4vI7D0eccZ2Zwo7HgK7Kvy+/5w0/5zQ1byLafmqlx+Uf5i+af0p561zU1fSdEkvxp/wBZENbK5JkHpXEZWrx/s8h44q/bH8tvOrfmL5Z0fzI+g6noLajD6p03WbX6rqFsQzLwnh5NwY0qBX7JB2rTFXxx/wA5O/8APxT8v/8AnFjzrpHknzHo+rX1zd20F7dXFjFCYbS2nkkjQ0llRpHrGSyqKBaUYt8GKvvuKVJ0SSNuSOoZT4gioOKqmKuxV+Uf/PyOK617zp/zjP5S/S+o2OmeYvNrWGpJp97NaPNBLNYREFomU1CyOFPVeRpir9Ivy1/LzS/yp8t6Z5W0aa8lsdPEohe/vJr25PqyvM3Oedmdvic0BNFWirRQBirGP+cg/wAt7r84Py089eSrCaGG813R7+xtZbksIEuJomELSFFdggfiWKqxA3Ck7YqjvyO8gT/lV+XXkXybdzRTXOgaHpem3EsHL0pJ7W2jilePmFbizqxWoBodwDir4m1v/nO/zxof552X5JTfkXOLnULx/qN/J5mtojd6SksqnUYoWtOBX0oZJfR+sepRCho4IxVP/wDnJT/nNbzl+Q/5k+Wfy70f8l5vNMnmYKNGuovMMFo97KgT10W3+qTmMQ8xV5XjUirV4qxCrO/+civ+cy7D8kNY8ueSNE8o3/nHz95giWay8tabKiPHGa1kubjjKIk+F6MEcUR3bjGOeKvI0/5z281flhrug6X+fn5L335e6brlwtrZ69FrFtq+mxzvuqXUluirCONSTzZqAt6YRXZVXrf/ADkL/wA5Ya1+Qnnv8uPLb/lxJqvl/wA46lpekL5jXWIoI7a9v7loTCLUQSySGOMCWpeNWB4hqgnFUg/5yw/5zC82f84vajYzt+UNxrnlSVrGKfzEuuwWkcNzdytH6Itvq00jMoUGpKBq0qOuKsz/AOcx/wDnJ3Xf+cU/K1p5wsPy7k816Uspj1S4TVotOTTg8kMNuWDQTvJ60kvEcEopX4iKjFXyxrn/AD8n85eW9K0z8wtV/wCce9bsfyzvpbZE1+41KBb0QXLAR3BsViJVHqPT5ShJKrxl+NQVX6IfmT+dXlL8qfI19+YvmDUxFoNraxXYnReTzrPx9BIUNCzyl1VF23IrQVIVfnzf/wDPyzzNoGhTedtb/wCccPNth5NeAzWestMjNJzVvQaeBoEEEMrcAJTI60aq8/hDKvr3/nEz8/8AXv8AnJTyTH511jyDJ5StryU/o1H1WLUVvbUCnrqyQwMg5hl4vGDtUEg4qlXn3/nHe/8ANv5+flf+bkN3aJY+VNJ1uxuYJGkF1JLexSQwGICMoVAnl5lnUjbiGqeKr60xV2KuxV2KuxV8w/8AOZP5Hap/zkd+UHm38v8AQ7u0tNS1T9HvaT3zSJbJJaX1vdH1GijlcBliZahG3bpir26SK68meVWi0nTm1S60nTCtpZRyJC13LawUihWST4EMjKFDN8K1qdhir4l/5xR/5za8zf8AOR9/5kfWfylbyh5f8vNfWuo61ceY7e6ittRsvRaS1mhe1tXQhJCxk3ReJB70VVP+cof+c0/Nn/OPvmDyppui/lA3m3SfNM1hZaPq0HmS2tEvdQvSfTtoYRbXDGo4kSMVRuQ37lVM/wA5v+c3X8i+btN/LHyN+XmoeefzAuLOK9vdEsryG3g0uORFel5elZURhzU/Z4cWUtInOMMqx/yd/wA5061oXnbQfIH54/lTe/lxqPmFxDo9+2pwappV5OSF9I3MCIkblmVQA0lC68ygZSVWLfml/wA55fmR5H/NPU/yj0b/AJx8vNZ1poJr3RZI/MMIXUNPjlkjF7Ii2jLbxMIpCA8vLkFjIVnFFXjdt/z9c81axoeo+Y9K/IDU59N8rNFF5wuH1ZETTJ5J3h9KH/Ri0tAoZiyoUYlXRVAlZV+i35rfn3d+TPy403z/AOT/ACNq/niTVk06TTtL0qJjPJHqEYljllKRytFEqkc3Eb8SRUU3Cr5Otv8AnO38x/IGueUbb86PyHuvJWg+aNQg0uz1iLXLfUVgurg0jS4hiiBStCx5MjhFZljficVe+f8AOY//ADk7rv8Azin5WtPOFh+XcnmvSllMeqXCatFpyacHkhhtywaCd5PWkl4jglFK/ERUYq+WNc/5+T+cvLelaZ+YWq/84963Y/lnfS2yJr9xqUC3oguWAjuDYrESqPUenylCSVXjL8agqv1Y0zUrbWbO01CymE1tdwxzwSLWjxSqHRhXehBBxV8s2v8AzjzqNv8A85IXv50G8tDpU/ktPLwteUn1wX4v1nMpHp+n6XooFB9TlyNOAAqVX1pirsVdirsVdir5M/5zF/5x4v8A/nJTyn5b8vadd2ltLpnmfRtaka8aRUa3tGkSdV9OOQ+oYpW4AgKTszKDUKvrPFX5red/+c/Na1jzprvkT8j/AMpdQ/Me88vu0Or6jFfJYaZbToxUxJO8UiyGoZfiaPkVb0xIqlsVV/8AnHr/AJz91L85PzVb8o/Mf5U33k7XLXTru7vlvdQEzQTwGNliWP6rEXjkicOsvJewVWUh8Vek/wDOcn5Q+RfMv5T/AJpea9W8maNfa9YeVtWa01W50y2lv4DBbSvF6dy8ZlX02JZKMOJNRTFXof8Azhn/AOSI/J//AMBXRP8AqEjxVL/+cpPzM87fl9o1vB5W/JyX8xbDU7fUYtYjTWrfTI7S2WNRSQTRStKJleQfCo4hDU1ZRiqU/wDOCXmryr53/JTyjrvkvyZF5P0i9bUzFo0V214tu8N/cQSH13SNpDI8ZepUH4qdsVfP/wCaX/OeX5keR/zT1P8AKPRv+cfLzWdaaCa90WSPzDCF1DT45ZIxeyItoy28TCKQgPLy5BYyFZxRV43bf8/XPNWsaHqPmPSvyA1OfTfKzRRecLh9WRE0yeSd4fSh/wBGLS0ChmLKhRiVdFUCVlUz/wCfi3m3yXpsf/ONP50Np0Rhj83eXdTk1OOzjOoPo0Q/SQh50EjKBV1iLceZ8TXFXpOo/wDPxzXvJ8Fp5l86/wDOPfm3y95JupIFXX5ikskMc5oktxZiJTEpqtKyktWi8m4hlX3xrv5gzXPkWfzl5E0oeb3n05NQ0mytbyO2GpLKivEqTygogdTWpB+VdsVfDn/OKH5rWP5i/nZ+Z8Ov/kYn5fefrfStNn1e8bWxqc93bzCERRuEt4Y0HBYmqhPKg5brir6f/wCcqPzx17/nHfyLd+dtE8iSebksJOeoW6anFpwtLJY5HkumkkimLKhVV4JGzHnXoDir4dm/5+Vefbvynb/mZpP/ADjlrE/kKGKCTUNYn1iGGRCSqXDW8At2aWGGQsqzbJJxq/o/EFVfo3afnV5SuPy7t/zTl1IW3lmXSI9bN1OpVo7R4RMOSCp9Sh48BVi/wCppir4Jtf8An5Nrl/YSec7X/nHrzdN+XkfOU+ZEMZlNqjUa5Wy9OhiC1ZnFxwWhq2xIVe//APOGf/OVWu/85ZaDqnmi7/LiXyto0cqxaZeNrEWoLqDK8qTgIsEEkRhKKDyUq3P4W+E4qnX/ADkn/wA473/53eZ/yW12yu7SCPyP5qt9dvFuWkDy20IWThBwjcGT1Yo9mKLSp5VABVfWmKuxV2KuxV2KvHv+cg/y3uvzg/LTz15KsJoYbzXdHv7G1luSwgS4miYQtIUV2CB+JYqrEDcKTtiqO/I7yBP+VX5deRfJt3NFNc6Boel6bcSwcvSkntbaOKV4+YVuLOrFagGh3AOKvlf/AJy0/wCcy/NP/OKuoWl3dflHNq/k+R7KGbzKNdhto4p7lmDRC0W1nmYoqk1JUN0FOuKvMrz/AJ+FeePMrxa3+W3/ADjj5p81+TTdrAmuD1LaW9iZ/T9ezsxayu8dakOWC0FJPSPLiq+g/wDnJb/nMTSPyA1Ly75R0zy1f+b/ADv5jHLS/LemMqztGCQZbiWknox/C4DBHrwc0CI7qq8RuP8AnPbzl+U+qaLD+fP5HX/kLRNYuFtbfXrfWbbWbKCaTdEuvq0aiIcaknmXorEREK1FX0R/zk//AM5b+Vv+cY9E0e8vbS517WfMEwttB0TTKSXWpTHiBwpypHV0UuFc1dFRHZgMVfF35gf8/IvzV/JTSv05+ZH/ADjLqWiaddjjYXA16OWL6wwrHBcutmfSLAMasA+20R3oq9s/5zjs/Ln5s/8AOMXmHzrqPl2znuh5cg1bS5Lu2iuLjTnvlt5GMEzpyjfiQrMnHlTfFXmepf8AOa1/+Rfln8hfy+8mfl5J+YPmPW/KGjTyabp+rC3urJI7C14NLElpdMqyBmYM/BQqMxNN8VfpL+W/mDXvNflnR9X8z+WX8s6vdw+pd6Q95FetZyciOBuIQqSbANUAdaEAgjFXgVr/AM486jb/APOSF7+dBvLQ6VP5LTy8LXlJ9cF+L9ZzKR6fp+l6KBQfU5cjTgAKlV9aYq7FXYq7FXYq+TP+cxf+ceL/AP5yU8p+W/L2nXdpbS6Z5n0bWpGvGkVGt7RpEnVfTjkPqGKVuAICk7Myg1Cr6zxV+EX/AD8p/OHzP5x8weQfy+1b8pvMK+WrfztZLIxlVIPNSxsES1tDFUhpldxHVqgspKhloqr7F/5xM0fyvqfmy71C2/5xXn/K+906wb6vrF1ZWsIlEjLG8CGNUfmykktQ1UMGYVAZV6X+f/8AzkZ+Zf5eearDyf8Alz+Rmr+d7i4so7yXU/rS6dpMHOR4xD9alieJpRwqyM8ZCsp3BxVA/wDON3/OXOqfm35y8zflj58/L658jedtBtI9Rk06W9S/t7mxdo0M8NxGiKeLyoCBzU8hxdiHCKob/nI//nNe1/JrzVpX5b+TvJd/59896jEtyui6bKIUtrdq0kuZ/Tm9MkDkF9M0T45GjUozKvnqb/n5H548j+cPJnkT8y/yCv8AyxqvmbVtOsLeZ9aSW0NrdzrbyXEUi2hWV4WdKxK+4PxSR/CGVfVf/OTf/OXmlf8AOPd95d8rab5bvvN/nXzIW/Q/lzTGCTSopIM08pV/RhqGAf03J4uePBJGRV4XN/znj53/ACm1fQ7f8/PySu/IWh63craW2vW2t2usWdvNJuiXf1ZQIhxDMzF+fFWYREKxCr9MEdZFV0YMrAEEGoIPQg4q+Tv+ccP+ceL/APJTzZ+dXmG9u7SaLzz5nk1qzW2aQvFburPxn5xoBJ6ssmyl1pQ8qkgKvrPFX42a3+W9r+RX/OY/5JaR5Y13XP0f5ksvMd9qNrfa1d3sMsosr9gOM8jDiCFIU1oVB6iuKv2TxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KvzT/AOfrPn3y3o3/ADj/AOdPLF9rtlb61rC6Q+nadLcxrd3aW+sWMsrQwk83VEVmYgUABrirzDyz5l/5xN8hp5L/ADN8t+a/KGj+a/Kvlu/WKx0q502yj1C8vLBI3F7HBGss0qsrKnxqQZHryJFFWX/mp/zkND+ZX/OFmpedvOl7pmkap5u8uX9tDbpJ6ENxfsZ41gtUnkd2d1iLBAztQN2GKom0/M/yN/0I4JG8y6UYF/Lg6A3+lwlRrf6A4fUSOX+9Xqn+6/vOW9MVeb+V/L7f85Gf84HDyV5C1ODU9ctNBsIJrS0uFeZLzT7qC/eykVTVZZY4+Ko1K817HFUy/L7/AJ+dflRo/wCW2j+XLjQ9WXzpp2nQaJ/gqPR7l7ibULeIWgto2ERiEbOtAGIkCGhi5jhir0n/AJ9SXkFh/wA416BdXMqQQQ32uyyySMEjjjS7lZmZmNAqgVJJ2HXFXi//ADl//wA5LflT5t/Mv/nGnUdF/MPQ9Qs9D82S3Wpz22owyxWduVgAlmZWIRNjudtsVfqZ+Xf5z+Q/zcj1CXyV5v0vX008xC7OnXsVz9X9UMY/U9Njx58G4168Wp0OKvyz/wCfjH/OSn5Vef8AyN5K0/yz+Yeh6vdWnnfQLyeGx1KC4eO2gFz6kzCNj+7TkKt03GKv06/L3/nID8tfzZvrjTPJnnrRtevbeE3Mttp+oQ3EyQB1QyFI2LcAzqpalAWUdxir8aNTH5Kfml+cH5t/of8AOjzF+R3m218wX9hqoj15LHTNbexne2a7icvbgSSSq5aJpifi5KnxNRV7h/zg5+anm4/nn5+/LNPzS/5Wr5R07QoL6HzK0SSSQXQlhCW7XqNJ6oImmFPWlUlOUfAiZcVTPyv+avkm5/5zu8xSL5p0uRZvIq6BA4vIWV9ZTVLTlYo3KhuQEcGMHnswpWoxVZ5q/NXyPY/853eXpn80aXEsfkVtBuX+uQhV1l9TugtlI3KguSGjAjJ57qKdMVfrrirsVeVfmJ+ef5eflHNZ2/nXzrpGgTXiNJbxajfw20kqIaMyLIwJUHYkCldsVfmN/wA+vvz+/Lbyt+WI8rax550ew1q+80ar9V0+6v4Ybmb61LGsHCN2Bb1DslPtHYYqzX/n63518taZ5Q/LjQ9S1myhvm866DqTWU08YmOnwLdJNcGInl6KFgrORxqaVriqV/8AP1/8x/J+uf8AOPdutl5i027m1y90u/0QRXUUj3sEMqmSe1CsS6Ikg5OuwDUJ3xVF/wDP0zzfonnf/nG211/y9qFrrWl3mvaRJb3FpOs9vcqskyMFkiJBHJSpoaggjqMVeg6//wA/JP8AnH0flre6w2vQSSvpssB8qvbSLfmYxGI2L25jCqOX7tnJ9Hj8QfjQ4q+TfJX5SebfLH/PvbzdpOr6fcRX15aXmtw2UqETQWH16G7BKDdQ0UTT0O4V/ipuAq9P/MT/AJza/J65/wCcT7rTbXzlZXGt6r5H/QMeiwyh9Rjv7jTfqTpLAPijWJyxaRgEKrVGaq8lX11/zgB5u0TzV+Qv5bxaPq1tfPpWkWWn3628ySNa3kUKM8EwUkpIoZSVahoQe+Kvjn/nC78/Py//ACF1P/nIbyt+Ynmqx8s6nB+Yeu6msGqTC2kntbkokbwLJQy19IsAgY8WRqcWUlV+jH5F/wDOSnkH/nJC31288garLqdpo10lncXLWVxbRNK6cx6RuI0LinWgqNiRRlJVfnv/AM54f85Rfkv+UX5qeT9F8/8A5MQ+cdVtrK1vTq08cAeytJZp1jSJZYnN1wdHf02ZI1ZiVb1OQCr9fIpVmRJFrxdQwqCpoRXcEAj5HFVTFXYq/K3/AJ+Cf+Ta/wCcQf8AwOk/6itNxV+qWKuxV2Kvzg/5+O/lNq+qeU9C/ODyUvDzf+V14uuWrqpLT6fGyveQuFI5IqoJSCd0SVAP3hxV5/8A84Vy33/OWP5meav+cnPMGmyWenQQL5a8l2NwQ5tbaJa3twCNuTO7JyWorJOnRRirE/zB826b/wA4xf8AOaN1+YH5jM9p5W88eV4tJ0nWpkL2lheQ/U1eJ3Wvpgm2bmSPh9dWJWPmQqp/8/Hf+cjPy8/OP8t4fyk/LvWrDzv5t826lpcWmWmhXEWofV2guY52leWBmjjJVTEFLhqSMxHBXZVWTf8APwe1uPyy/L3/AJx41zWWkubTyX508pz6zdRRtIVjtLdxJMQoJozRkDxZlXqwxV5T/wA/N/8AnLr8qPPX5Y6N5T8p+c7DX9R1LWtMvSumzi4S2tbVmkeSd0qsZqVURsRIak8aKxCr3b/n55+avk3XP+cZtc/R3mnTLr/EzaVJoghvYZDqSWuqWck7WoVj6oiXeQrXh+1TFUX/AM5q/md5D1v/AJxI1qaz8y6TdWetaVY2ekGO6heO8urO4gLRWwDHnJC0LclXdChrTjirB/8AnIvy9cf85D/84e+UZfy5kh8zy6Db+Wr64sbFxdG6Ol26RXdoUj5FpI+RZovtnhxALEKVUp/Nv/n53+UP5i/lL5l0Ly/p+p6l5o13Q9R08eWTpNw0ltJNaSJMZ5PT9BoYF5O5jdm4ITxXfiq+xP8An3x/6zt+VH/bJ/7GJsVfZGKuxV+fX/OWP/OeUH/ONHnfyd+Xum+R5/Nmu+ZYYpYbaDUUsmRrm5NpaoOVvNyaaRXA+zTj3rsq/QGIuyIZFCuQCyg8gDTcA0FaeNBiqpirsVdirsVfz4f85F6PrHkP85fzN/ILSIbiLTvz41fylqltPBxAs4p7x/0xLvtWR45S+x+BRUeKqN/5w9i178zfzd/LL8qfMcM0kP8Azj7H5s+uTSLSK6u/rxsrCgqf7lDGYvaM0qKkqvbPI35haD/zil/zln+cyfmncro9p+YcWn3/AJe1+++GyaKAH1Ldrj7MQq/AlyqqYFDEc4+SqUf8/Avze8of85RTflj+Tn5Ta1a+afNN15nstT+u6NMl5b6Za28U8byvcwsYqj1ebAOQqRs0nE+nVV6B5g/NryVD/wA52aIz+a9MVYvIZ8uyMb2Himsvq9xxsGPKguTzUekfjqQKVxV83eR/MGit+R3/ADm9oAv7b9Mr5q833z2XqJ9a+pc7eJJjHXl6XqB1DU4hqjriqZfnB+fUv/Kqv+cR/Kum/mNc+V/JHmHSrDTfNfmTRJSbm1m06wsIXsmuIqm3ZXeQSg7inJlZI3RlXzj/AM5h2v8Azj55NsvKNj+Wv5na75r1lNb068v2l1+XVdLhs4ywe4uZGX0RNyZQnpsGUF+QAIDKv0r/AOfmn5u+SPMv/OMeuvpPm3S75fMsmnHRTb30Mv6RFjq1obr6txY+r6AU+rxrwpRqHFUx/wCc1fzO8h63/wA4ka1NZ+ZdJurPWtKsbPSDHdQvHeXVncQForYBjzkhaFuSruhQ1pxxV9if84v+b9E86/lT5BvdB1a21K3h0TS7SWW1mWZY7m2tIo5onKk8XjYEMp3B64q96xV2Kvz4/N//AJzxT8uvzw8ufkboXkSbzHquq/o5ZrqPU1tVtHvWZm5Rm3lLLDABM7cl+A9Nq4q/QfFXYq7FXYq0wqCAaV7jt9+KvxO/597/AJy+TP8AnFm0/MH8mfzY1i18p+atM8x3189zq0gtYNTt5o4Y0nW5lIjJPp8lBb442R05fHxVfZP5Z/8AOR/5K/nn+dNxZ+RtFbzD5j0jRZo7nzfaaepsrW1MqkWf1xirt6jElOKNGfi4Ofjoqhv+c3vz/wDy10T8svzb8j3/AJ60a38yS+W9Ut00iTUIRfGa5smaGP0eXMPIHUqpFSGUjYjFUN/zhP8A85EflhqH5YflD5JtvP2iyeY18vaTZnSV1CH64LmG0X1IvR5cua8WqtK7Yq9x/Oj/AJyH/LD8urfWvL3mfz9ouk6s+nTutjeajDDclZonEZ9NmDfH+ztv2xV8N/8APtL/AJyL/K/yn+RH5d+UNa8/6Jp+vC41WE6bdajDFdCS61e7eFTG7BqyLIhXxDDxxVvzB+bXkqH/AJzs0Rn816YqxeQz5dkY3sPFNZfV7jjYMeVBcnmo9I/HUgUrir5u8j+YNFb8jv8AnN7QBf236ZXzV5vvnsvUT619S528STGOvL0vUDqGpxDVHXFWN/8AOTv57+Q4fyV/5w/e31jT/MFx5duPKOpano1rdQz3TW+nWKRXMUkIaq1lieE8wBzDKehxV9tf85M/859fkV5l/JnzdaaL5rtfMOoeZtFvNN07RreKV7x7m/geGITQlA0IjZgzF+JHGiVcqCq9k/5xhv8AT/8AnFj/AJx6/LKx/NnzDZ+XJobQQyNq91Fa+nPeSz3kVpWRlBkjibiUFWAjav2ScVfJP5W/85LflTpf/OWP51ea7v8AMPQ4ND1Py/5et7LUZNRhW1uZoILZZEilLcWZCpDAHahxV9W/85o/m95I1b/nG7z5rVl5t0u407zDo+oWOkXUd7C0OoXTJIogtmDUlkrG4KLVhwao+E0VeU2n5n+Rv+hHBI3mXSjAv5cHQG/0uEqNb/QHD6iRy/3q9U/3X95y3pirzfyv5fb/AJyM/wCcDh5K8hanBqeuWmg2EE1paXCvMl5p91BfvZSKpqssscfFUalea9jiqZfl9/z86/KjR/y20fy5caHqy+dNO06DRP8ABUej3L3E2oW8QtBbRsIjEI2daAMRIENDFzHDFXqP/Ppk1/5xy8tGlP8AcjrW3/R7Jir9KMVdir4f/wCc2P8AnNjS/wDnDTTPK93d+XH1+81+4uooLSO9WzKQ2iI0spcwzVo0sa04/tVrtQqvrfyPreoeZvLuhavq2jto99f2VrdXOmvMJns5Zo1keB5AqBmjJ4sQo3BxVlOKuxV2KuxV+Rf/AD9V/NryDe+QIPIsvmrSptctfMugTX2j/W4nu4rb4pHaaAMWVfTdWPID4WB6EYq/Qf8ALP8APv8AKn8xLpPLvkPzvoOq3Fpa800/S763kaK1h4R1WKJto05KuwotQMVfnR+dXmqw/wCcav8AnMfQfzN/MBJIPKXmnywdCsdZeMyW2m36OpZXKAsgom5pstwz/YWQqqiP+fif/OTf5Z/mb+VF7+WHkbzBp/nXzT5wu9LtNKsNCuY9SaOSO9guDK72zOsZpHwVSwdmf7JQSEKsG/5yI0u+/wCcY/zG/wCcT/zO89wXF/5Y8o6DD5Z1y8iiNwllqP1F7b606oGY1aX1RxDFvQIQF+IZV9Xfn3/znF/zjYPJ9zYa3r+m+e7bWFSBPL2kLHqtzeuzKyI0KsFiIYAgytGQwHGr8RiqH/5zx8+eVPLX/ONfmPS7u5tvLs2u6BHDoujXrxWt2xjFu31WK35El4EZVdU5BNhWlMVfA/5baRdf8+24/wAufzikvj5h8k/mPo+g2fmX64ITq+nXs9oLuN7NqK8kCLz/AHQJ+CPjJVlikVV+5f5dfmh5S/NzSf075M8xWWuacJWga5sZ1mRJlVHaN+JqjhXUlWAYBhUb4qzzFXYq/Pj8/f8AnPFPyd/ODyp+TWieQ5/M+s67Hp5MsWpLaJbyX07xIjKbeYngieq7EqqoQfEhV+g+KuxV2KuxV2Kvx0/5+F/nj+X1z50/ITTIfOelSXnlf8yNHutcgS9iaTTYLeaMzPcgMfTEdDy5U40NemKv0z/Ln89/y6/N6e8tfJXnbSdfns0WW4i0++iuJIo3PEMyoxIUnatKVxV+PPm3zv5f/OP/AJyA/Nzyv+fX5w6t5I0LypcQW/l3QrbVjotje27I5aeWYD947II5QDR29YcH4LwxVgv/ADit+YP5O+TP+ctr+fyl53vpvK9x5Wl0mx1PzLqLsLzUJL22Ho2c12I3aIspWMNuzrIVqpXFX0K3nrRf+cVv+cyfzF8wfmc36M0b8w9H04aF5huUJtozbQ2sUttJKoIjHOEq1T8ISFnosgbFX0V+ZX/OXX/OPfnnzf8Al55Liig/MjXbjXLCbSo9EtYtVj0u5D0W+e45iKNYRV34O7qilmQKK4q+L/8AnOHyunlX/nKbyR5w83+dtf8AJHlfXNAGl2fmnRJvRaxvoTcc7aSUK5SJxIpeq/7t5fYSQoqxj8+PLX5AyaAmnedf+cxvOHnGwuZYnTSbDXLbXjNKhqlYraGZEYHcGUrv03oMVfu75Otbaw0DQ7ayed7aGxtI4WugVuGiSJVQyhlUiQgDkCoPKuwxVkeKuxV+WH5/f+tsf841f9sbzJ/1A6hir9T8VdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirzvzz+UPkT8zpLSXzj5K0TzDJZq62z6vpVpftAshBcRm5icoGKioWlaCvTFWCf9Cnfkh/5ZvyZ/4S+l/9k2Ks51b8n/IWv6DYeVdT8kaHe6Fp7I1npVzpNpNYWzRqyoYbZ4jFGVV2AKqKBiB1OKoMfkb+W6+Xj5SH5f8Al4eXzP8AWjpH6Fsv0ebj/fptfR9L1NvtceXvirIfJX5d+VPy2s5dP8o+WNL0CzlkM0lvpVhb2MLykBS7JboilqACpFaAYq+AfzT/ADk/5yOn1zzboPkn/nGeD63zvLHSvNVz5gsfSa2q0cV20bxxH4lpJ6RnBX7LBiDVV9Df84df84+3P/OOv5P+XPy91y4t9QvYEvJdRaIF7Z5r2eSaSNPUUFkQOEqyjnQtxFeIVZV/0Kd+SH/lm/Jn/hL6X/2TYq9D8j/lZ5L/ACxjvIvJ3lDR/L0d4Ua5TSdMtrBZ2jDBDILaNA5UMacq0qadcVefSf8AOKX5JSszv+Tvk1mYksx8saYSSdySTbdcVZj5J/Jb8vfy0u57/wAoeRNB8v3U8XoTT6To9nYyyRcg/pu9vEjMvJQeJNKgHtirFNf/AOcXfye81XGoXmsflf5avrrULmS8urm40W0knmuZd3leVouZZjux5bnc7nFXoPkT8tfKX5X2LaZ5P8s6boNm7+o8Gm2UNpG70pycQqvJqdzU4qxy3/IP8sbTXf8AFMH5c+W49d+ste/pVNDsVv8A607F2n+siH1fVLEkvy5EmtcVauvyC/LG910+aLj8uPLc2uG5W8OqSaHYvfG5RgyzfWGhMvqAgEPy5AitcVetYq7FXnXnj8oPIf5nSWk3nHyTofmGS0V0tn1fSbS/aFZCC4jNzE5QMQCQKVoK4qxTSv8AnGP8ndCvLXUdN/KfylZ3lrIk1vc2/lzTopoZUPJXjkS3DKykVBBBB6Yqynzx+TvkH8zZrW484+SND8wzWqNHbyatpNpfvCjGrLG1zE5UEipAoCcVQfmH8jPy3822Wkabrn5feXtTstHiaDTba90WyuYLKJgoKW0csLLEpCKCqBR8K+AxV8C/8/EPy883ar+XvlX8tPym/KWTUdOj1C11GukG0s7LT1sJS4t1tlC09UyEgqAq0J+I7Yq94/Iuztvzl1bWvM/n7/nGqx8ka3YSWzWmo6nBpl/eXrSmRpGSeO3SZGiKKST1LihqDir7SkjWVWR1DKwIZSKgg7EEHtirxbR/+ca/yk8v3V/fab+WPlm0uL+Ge3upYdDskaaC5UpNExEO8cqkiRfsuDRgcVZ75L/L3yt+W1lJpnlHy1pmgWUkrTvbaVYQWMDTMqqZGjt0RS5CqCxFaADtirFPPH5Cflp+Zl9Hqnm38v8AQNcvowqrdajpNrdT8F2CmSWNmKjspPH2xV6Louhab5asoNO0jT7fT7OAcYra0gSCGNfBI4wqqPkMVYz5n/K3yb521DTdW8xeU9J1a/0w8rG7v9Ot7me1IPIejJLGzJQ7/CRvv1xVneKuxV2Kvzf/AOc9fym/Mfzv5j/Irzf+Xnk//FEvkfzFJq97YDUrTT2kRHtJkUS3bqoDmBlLAOVqDxOKvuT8svMPmPzX5a0vVfNvlRvKusXCym60dtQg1E2pWV0QfWbYCOTmgV/hApy4n4gcVZ5irsVflN/zml+aH5/eeNA8+flh5B/I3V1gvy2mp5lTUbZ47iwcgTtFAApX14+UfxSVVWNQG2Cr7X/5xV0F/K35TeR9Gk8n3PlJ9PsBato93PDcXEJhkdDJLJB8DPcEeuxABrJ8QDVGKvXvNnk7QPPmmzaP5l0Sx1nTpipks9QtYru3cqaqWimVlJB3BptirCPy/wDyE/Lb8qbmW98neQ9E0K7lVke50/TLe3uGRjUoZUQPwr+zy4+2KvQ9f8vaV5r0670jW9NttS0+8jMVzZ3kEdxbzRnqskUqsjKfAgjFXlGm/wDOM/5RaRpt9o1n+V/liLT7543u7QaFYmG4aJuUZlQwkPwJqnIHj+zTFU91r8jfy48x6VpOhat+X/l6/wBK0gMunWF1otlPa2StQMLeGSFkiBoK8FWtMVa1D8jPy31bQ9O8sX35feXrnRNNkeWy0ubRbKSxtZHLFnht2hMUbMXapVQTybxOKsu8peS/L/kHTo9H8saFYaLp0bO6Wem2cNnbKzmrMIoERAWO5IG+Kvy3/NXz7/zkz+bml+cfInl7/nGq08tXHmCG70e58yXnmKylijsbrlbyS0SKF3PpMxBVpCpPJY2IClV+hP8Azj1+Vbfkj+W3kvyLJdrdy6FptvaT3CAhJZwOUrIDuEMjNxrvxpXfFXsmKuxV+d8H/OIWveY/+cq9T/PDzddWNzoOl6ZaQeWLSKSR7iO5W3WBmuEeNUURs1xItGb4pI3BDIQFX6IYq7FXYq7FXYqwrV/y48r6/wCYNE816loFnda3oizppuoSwK1zaLcKUlEUhFV5AkH5mnU4q7Qfy48r+V9b17zJpGgWdlq+vNA2qXsECpPeGBSsZlcCrcQTT5164qv87/l15V/MywGl+bvLem69ZK/qLb6nZQ3kSyAEB1WZGCsAdmFCPHFUl/Lz8l/IP5SiceS/Jej6AbhVWd9N0+C1kmVTUCSSJFZwD05E0xVA3H5B/ljda7/iif8ALny3Jrv1lb39Kvodi1/9aVg6z/WTD6vqhgCH5cgRWuKpha/kz+X9lN5gubfyLoMM3mJJo9Zkj0i0R9TSdi8q3jCIG4DsxLCTlyJJNa4qh7j8j/y6uvLX+DJPIehHy4JDMNHGk2q2Cyli5kW3WMRq/Ik8goau9a4qlml/846flXouiXnlqy/Lfy5Fo940Ul1YDRbM21y8Lco3njaIrKyHdWcMVPQ4qj9V/Ij8tNd0nStA1L8u/Lt5pOk+p+jtPuNEspbSz9U1k+rwPCY4uZ3bgor3xVfqH5Gflvq2h6d5Yvvy+8vXOiabI8tlpc2i2UljayOWLPDbtCYo2Yu1Sqgnk3icVZd5S8l+X/IOnR6P5Y0Kw0XTo2d0s9Ns4bO2VnNWYRQIiAsdyQN8VZLirsVfnh+Sf/OIWu+W/wDnIf8ANX88fO9zY3k2qyG38srbSPI8Fm6LCZJhJGnCZLeKKEBSwo0u9CpKr9D8VdirsVdirsVeX/mH+SX5ffm2bdvOvkrR9fkt1KQS6jp0FzLCpPIrHJIhdATuQpAPfFWReSvy/wDLH5b6euk+U/Lun6HYBi/1XTbOG0h5nqxSFVBY9yRU98VYh5r/AOcfvyu896nPrXmX8tvLWs6lOEEt7qOhWN3cyCNQiB5Z4XdgqgKKnYAAbYqoeXP+cdPyo8nala6zoH5YeV9K1K0Yvb3tj5f0+2uYWKlSY5YoFdSVJBII2JGKo7zj+RH5afmHqH6W81fl55d13UPTSH63qeiWV7cemleKercQu/FamgrQVxVIdP8A+cXvya0m6tr6x/KTyhbXVtJHNBPD5a02OWKWNgyOjrbhlZWAIIIIIqMVTq4/IP8ALG613/FE/wCXPluTXfrK3v6VfQ7Fr/60rB1n+smH1fVDAEPy5AitcVTC1/Jn8v7KbzBc2/kXQYZvMSTR6zJHpFoj6mk7F5VvGEQNwHZiWEnLkSSa1xVI7X/nHH8prGfSrm2/K/ytDNpLK+nyx+X9PR7NkmNwpt2WAGIiVjIClKOS/wBo1xVW0b/nHr8rvLuuf4l0r8ufLtlrIf1V1C30WziulkNausqRBlc1NWBBPc4qzLzn5A8r/mPYJpfm3y3puvWKSrOtrqtjBewLMgZVkEdwjqHAZgGpUAkdziry7/oU78kP/LN+TP8Awl9L/wCybFWaah+S/wCX2reX7HynfeRNBudAsJPVtNIm0ezk0+3k+M84rZojEjfvH3VQfib+Y4qoD8jfy3Xy8fKQ/L/y8PL5n+tHSP0LZfo83H+/Ta+j6Xqbfa48vfFWQ+Svy78qfltZy6f5R8saXoFnLIZpLfSrC3sYXlICl2S3RFLUAFSK0AxV8A/mn+cn/OR0+uebdB8k/wDOM8H1vneWOlearnzBY+k1tVo4rto3jiPxLST0jOCv2WDEGqr6L/5wy/IW+/5xq/KTyv5E1W9hvNSsxdXF9LbljB9Yu55J2SMuqsVjDhAxA5ceVFrQKvqTFXYq/O/8/wD/AJxC178/P+cgfyu8761dWMnkTyjZes9i0khu5tTjuJJ1Bj9P0zDIwtyx5nksToy/EpKr9EMVdirsVdirsVeQ+av+cffyt89anca15k/LXyzrOpXAQTXuo6DYXdzKI1CJzlmhd24qoUVOwAA2GKo7yX+R/wCXP5b3smp+UvIHl/QL2SJoHutK0azsZ2iYqxjMlvCjFSVBKk0qAe2Ksw8z+U9E87abcaP5i0ez1fTrigms7+1iureShqOcUysjUO4qMVYB5C/5x+/LL8rbx9S8ofl/oWiXrq6G6sNLtoLjg/2kEqRhwh/lB4+2KvTdY0bT/MVjdaZqtjBfWV1G0Vxa3UKTQTRt1SSOQFWU9wQRiryXyd/zjZ+VH5e6mNa8s/lv5e0nUQSUu7PSLWGeOvX03SMMgPgpAxVmXnj8r/Jv5nRWsHnHylpHmGK0Z3t49W022v0hdwAzRrcxuFLAAEilaYq35q/LDyb560uz0TzJ5S0jWdMs2je2sdR022u7WBokMaGOGaN0QqhKqVAopIG2Ko7yb5D8tfl1p50nyp5d07QrAyPMbTS7GCyt/VcAM/pW6InJgoqaVNB4YqyvFXYq/O/8tP8AnEDXLP8A5ya/MP8APbzrc2N3FPFDb+VobeR5JLdGtktHllV40EckcEfpgKzK3rSmvQlV+iGKuxV2KuxV2KvF9f8A+cbvyk816jd6vrf5W+VdS1C8cy3N5eeXtPuLieQ9XkllgZ3Y+LEnFWReR/ye8hfljNdXHk7yRofl6W7RY7iTSdJtLB5kQkqsjW0SFgCSQDWmKpV59/IP8tfzTvrbU/OHkLRNdvbZQkVxqOmW9zKqAkhC8iMSgJJ4klfbFUVe/kd+XGpT6LdXn5f+X7ifRI4otLll0ayeSwjicyRpas0JMKo5LKIyoVtxvirJPOnkHyz+Y+nNpHmzy9p+uaezrIbXUrOG7g5rXi4SZWUMK7MBUdjirG/y8/JH8vvylMzeS/JOj6DJOAs0unafBbSyqNwHkjQOwHYEnFWYeafKOh+edNn0fzHo1lq+nT8fVs9QtYrq3k4mo5RTKyGh3FRtirzbyV/zjd+VP5cXy6p5X/Ljy/pN+vLhd2mk2sVynI1ISVY+aj2BAxV7VirsVdir8pT5J/Or83f+coPyu/MXzH+UbeU/Lnk6312xlvH8xabqP1iO4tbyKGYRwMkqc2lQcAj0rVmABxV+rWKuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KtFgu5IHzxVvFXYqtV1cVVgRuKg16bHFV2KuxV2KuxV2KuxV2KuxV2KtBgSQCKjqPDFW8VdirsVdirsVaLBaVIFdh74q3irsVdirsVdirsVdirsVdirsVaDBuhB7bYq3irsVdirsVdirXIV41FaVp3pireKuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxVoMCSARUdR4Yq3irsVdirsVdirRYLSpArsPfFW8VdirsVdirsVdirsVdirSsGFQQR4jFW8VdirsVdirsVaDAkgEVHUeGKt4q7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FWgwNQCDTr7Yq3irsVdirsVdirsVdir5K/5y+/5ygj/wCcZvLOlz6bpH6c80+Y76PSfLujhyhu7yUqvJ+PxenGWXkBQszogK8+aqvkv8xrX/nNr8u/JuqfmJJ5/wDLGoXGm2Mt/e+VbXQ42jihiQySiC4KerNLGtSV9QBuFEZyQGVfY3/OL35y3fnz8jvJv5jee9TtbWe70t7zVL6X07O2QRSyI0r1KxxrxQEnZfoxV5jH/wA/NP8AnGyXVRpA/MqISF/SE7aZqS2nPlxp9YNqIwvfmW9Om/PFX1h5s/NXyl5I8o3fnzV9dt4/LdrbJeSalDyuoDbyFQkkf1ZZGkViy8eAatRTFXgOl/8AOe/5Bav5XvvOcH5j2iaNZ3YsJJ7i0vbaRroxib0ore4t455m4Hl+6jfb5HFWW/ld/wA5dflP+cnljzP5w8q+a0u9J8sxSz6xLJa3UEllDHE85kkhmhSQqY43ZSqsG4sBVlICrEvKn/OfH5BedNH1/XtM/Me0/R2g/VRqE93aXtiI2uzIIERby3haV39J6LEHb4TtirMfyJ/5y1/Kv/nJSfV7X8vfNA1S40pUe6hezurSRYpGKpIq3UMRdCRSq147B+JZaqsE/NX/AJ+AfkN+TWtz+XPMnn2EapbPwubaxtLu/Nu1aMsr2kMkaOpHxIW9Re64q98/Kf8AObyT+eehp5j8ieY7XW9OZvTaW3LK8UlAfTmhkVJYXoQeEiK1CDShGKvI/wA6/wDnNf8AJj/nHzUl0Xzt52gtNUKq5sLa3uL65jVqEGVLSKX0qg1AkKFhutcVRf5Sf85l/k3+emtWvl3yP51h1bVbmymv0tVs7yF1ggcRyFzPBGqOCw+BiHK/GFK74q+bfz8/5zp8v+Rvzd/J3yz5f/MXy+dAu7/X7bzmRd2k4s/q0MK2yzzcj9WIlaTupZk4nYEYq+1fIH5+flt+at9Ppnk3z5omvXsEJuZbbTtSt7qZIFZUMhSJ2YIGdVLUoCyg9Rir1vFXYq/Pv/nJH/nJrzxB+Ymh/kX+SmnWF5501G0/SOpalqRZrHQ7AEUkljQfE7LuAa8Q0YCSNKtFXzR+e/mP/nM7/nGu00fWn88aX560rUNSs7W6Nn5bt4rnT2llXigihiBaGXeL1CWYEivBmVsVfrH+Yv5l+Vvyk0O68y+cddtdG0q2oJLq7k4LyNeKIN2eRqHiiBnb9lTir5a8h/8APxn/AJx6/MfXLby9pH5hRR3t1IIrX6/Y3thDO7GihJruCKMFj8Kq7KzNsoJxV73+dX5/eQv+cdtHtNf/ADC18aNp93dLZQTG1urovcMjyBAlpDM/2UY140264q8d8w/85+fkH5Y8s6J5tvvzEtRp2txTT6csdrdvd3EcE8ls7C0WD6wiiWJ05SRovJW32OKqGq/8/Bf+cf8AQ9L8sazffmJDDZeZI5pNNl/R2ov6iwTG3l9QR2rGDhICp9YJ/N9nfFWKW/8Az84/5xuudN1PVF/MVBHYSxRNC+nX6XMxl5cGgge3Ekq/AeTKpEfw+pw5JyVfUn5XfnT5L/OXypD528o69Df6HIJuV2VeARGCvqrMk6o8ZSlSHUbUb7JBKr5UvP8An6B/zjXZ6kdMb8xA7CT02uItJ1KS2DVp/eralSv+UtV71pvir658x/mz5Q8qeTZ/zC1HXYF8sQ2UWotqcPO5ga0lCmOWP6usjSK4dePBWrUUxV8qap/z8y/5xt0vT7DUj+ZUM8V60ixx2+najLcII2CsZoBa+rCO6+oq8xunIYq+vfy//MLy5+anl/TfNPlLV4NW0fUUMlrd25JRwrFGBDAMrKylXRgGRgVYBgRirMsVdir8oNF/Pn89P+cyPM/m6L8jNc0jyb5E8tXsmlr5kvrFNTutWvY6M7W8UqPGIuJVgCoIR0YuWfhGq9B/I7/nJH8yvJv5sn8iPz3i02bWtRs5NR8t+YtKQwWurQRhi0TwkAJMBHKdglChTieUbuq+oPyp/wCco/yw/O/Xtf8ALPknzSuq6roPL9I24sryD0QsphJD3MEaSAOKVjZh07EYqkH5Tf8AOZX5O/nn5lu/J/kfzmmq6zawz3EtsNPv7celbyLHIyy3NtFE9Gdfsu1QeQqtTir56/5zB/5zV0r8oNc/LjQvKvnzQ0vJPOml6f5rtWuLW4ms9Ict9a+sKzE24UEFnIVkoNxvVV9deSf+cjvyr/MnVI9E8q/mJoGs6lIkkiWdjqttcXDpGOTlY43LEKNzQbDfpirCvMn/ADmZ+S/k7ztf/l3rvny003zFYRvNdW11BdQwwolr9dJe7eAWoPofEB61TsoHMhcVQn5cf85t/kd+beup5Z8q/mNp97qsjmKG1kS4tWncCvGBrqKJZjToIy1d6dDirKPzA/5yk/K/8rfOnl78vfNHmpLDzJr/ANU/R1ibO8mM31y4a1grLDA8MYeVSoMjp0qaDfFUj/Nf/nMn8nfyP8zWnk/zt5zTStauooJorU2F/OPTuHaONmltraWJAzKftutAKmg3xVPP+cjfz38t/kb5R1y/1TzVpWj6xJpWqz6JBqN1DFJeXlrbl0WGKRg0tJGjBVQd2UftDFXzx/zjP/znT+Xnmv8ALHyZqf5h/mp5atPNNzZA6pDcalZ2cq3CyOlXh5oIyygMQFUb7ADFX6CRyLKqujBlYBlZTUEHcEEdQcVeVx/nt+XcvnFvy+TzrpLeaQDXRxfRG85BDKU9PlX1AgLlPthPj48d8Ver4q7FXYq/MX8y/wA5fzr81/8AOSGp/kz+XXmXRNBsbDyzb64Z9S0lr5nYyxRyKSJFNSZlpSgAXxNcVfprAsixRiZ1eQKodlXirNTchSTQE9BU/PFX58f8/Jvyn1384Py+8maV5f0q81G4tfO2gXU0VlBJNIlsVubWWVhGCRHGJwzufhQDkxAFcVfoZir4s/Oz/nOL8h/yz1TzD+X3nP8AMFtI1iK3+r3cMWlapcSW/wBbtw6ES2tnLHy9ORWFHNKitDirFf8An3KPyw0b8sJ/LH5XefpvOWn6RqVy1zez6dc6c8c93SURiG5jjYKF3BHIE1Na7Yq9/wDyp/5yj/LD879e1/yz5J80rquq6Dy/SNuLK8g9ELKYSQ9zBGkgDilY2YdOxGKpB+U3/OZX5O/nn5lu/J/kfzmmq6zawz3EtsNPv7celbyLHIyy3NtFE9Gdfsu1QeQqtTiqV+b/APnOX8jPIHmnWfJnmL8wbXTdb0dHe9trm1vUSPhEJignNv6LyFCOMaSM7k8VUttiqY/kd/zmX+T/APzkbqV7o3kLzfHqGo2kZne0mtLqzmaANxMsa3UURkUEjlwqVqOYWoxV6/8AmX+a/lD8nNGk8wedfMNpommo6x+vdy8A8jdEjUVeRyATxRWagJpQHFXzK/8Az8c/5xwTR310/mjZG0S4jtSostQa69SRZGU/VBam5KUjasnpemDQFgWWqr6C1j89vIPl7yRZ/mPqnmmzsvK95a297balcs0Mc0NzGJYeCOokZ5FNVjCeoenCu2KvnvyH/wA/Gf8AnHr8x9ctvL2kfmFFHe3Ugitfr9je2EM7saKEmu4IowWPwqrsrM2ygnFX27ir88/yK/KrXvLn/OUv/ORfm290i8t9I1Wy8rrpt9NBIlteM9nGZxBKw4SGGSEq4UkpUcqchVV+hmKuxV2KuxV2Kvzz/wCc2vym138xvPP/ADjRqWj6Vd3sGg+erS71GS2gklS1tUeG5aacoCI4/wDRuPNqAFgtasAVX6GYq/Ff/nNP/nNPQPNnnD8v/wAtPJH5zr5W0S7vtSTzj5g0l5Fu7BLQKI4I50UMhdhIvKMmrcKngGDKvpL/AJwj8veRYdY81an5D/5yF8yfmVbpbW8F5p+tao19FZyTSM8dwBJEjK7CJ1BWikcg3IheKr2jSP8AnOL8jtc85X/kC18/W41+wk1KK6t57O+toYW0yOWW75Xc9ulsBCkMjM3q8aKaE4qg/wAv/wDnO78jfzS842/kPyv57h1DWroyLbRrZ3kcFw8aNI6RXEkCwswVSRR/i6IWO2Kp9+ef/OY35Rf844Xtrpfn3zdHp+o3MIuIrKK1ubu4MLMyiRktYpOCkqwBfiGoaVpiqf8A5Ff85Qfln/zkna39z+XvmeLVTp5jF5btDPbXMHqV4F4bmON+LUIDgFCQQGqCMVW+Uv8AnKP8r/PPn7Wfyw0PzSl35r0f60L7ThZXkfpG1dY5gJ5IFgcozAEJIx606GiqQ+Wf+cyvyc84fmDN+Vmj+c0ufNsN1fWT6cLC/QfWLBZHuYxcSWy27GMRP0lIPE8anFX07ir88/8An29+U2u/lD5G8/ab5g0q706e889eYLuCO8gkheS1CW1tFMgkALRyegWRxs4PJSRQ4q/QzFXYq7FXYq7FX56WH5Va9F/zmhqXnhtIvBoUn5dpCupejJ9TN8dQjh+qianD1fTQyGOvIL8RFCCVX1t+dP5ZR/m95T1Dy1J5k1vy8s5il/SHl+/+o36GBxIFWbg4CPSjgqeSkjbrir8+v+cI/wDnJjRPy0/5xe8ree/zd853Ygl1XU7E6lqH13Up3la7uDFGTEk8x+FGptQAU22xV97/AJKf85AeQf8AnInR7zXvy918azp9pdNZTzC0urUpcKiSFCl3DC/2XU1C0369cVfl9/zkR+TOqf8AOOn5l/kTrugfm3581FfNX5jaNY3+n6t5jkuLL6tcXkckkSxRxxVRgShVy4KHia9cVftJir4+/Nz/AJz1/Iv8kdam8ueaPPUKatbmlxZ2VrdX8lue6zNaQyJG69SjMHA344q9q/KT88vIf57aI3mLyJ5mtdZ09G4SyQlo5IHpXjPDMqSwtTcCRFJHxDbfFXw15o/5z68q2P5/+UPLWn/mb5bk8gXPl2/uNUvEvbOS3TVFll9FWvA5CNxjWiBgGD1INVIVfeH5efnR5B/No3y+SvOWkeYGsREbpdMv4LpoBLy4GQROxUNxbiT1oadMVfDf5+f850+X/I35u/k75Z8v/mL5fOgXd/r9t5zIu7ScWf1aGFbZZ5uR+rEStJ3UsycTsCMVfavkD8/Py2/NW+n0zyb580TXr2CE3Mttp2pW91MkCsqGQpE7MEDOqlqUBZQeoxVAfnR/zkT+XP8Azjzp9vqX5g+arXRYrlitvG6yT3M5FOXpW9ukkzhajkyoVWo5EVxV55+Sn/Obn5Lf85BaodC8l+dYbnVuLOlhdW9xY3EqqCT6K3UUYmIVSxEZdlUcmAGKp5+d3/OXf5Sf845ahp2l/mJ5tXRrzUIGubaH9H392zwq5jL/AOh20wUcgR8RFabYqw38yP8AnPv8hfyn1CHSvMX5hW0d7IsTtBaW13fPCkqh1MwtIJfSPFgSr0eh+ziqV/8AOWv/ADlf5b/KT8r9d1jy9560a18zX2hLqnluCa5t5J7yO4IEFxBbSkmVGHIoeJVip60IxVMfyY/5zO/K3zn5U8ivrP5oeWl8yarpekNe2f6VtYpRqVzbxGWH0TICj+sxXhSoPw4q+xsVfnn+RX5Va95c/wCcpf8AnIvzbe6ReW+karZeV102+mgkS2vGezjM4glYcJDDJCVcKSUqOVOQqq/QzFXYq7FXYq7FX55/85tflNrv5jeef+caNS0fSru9g0Hz1aXeoyW0EkqWtqjw3LTTlARHH/o3Hm1ACwWtWAKr9DMVfF/5qf8APwf8hvyb1+98r+ZPPCrq1jJ6V3bWlhe3n1eQdUkkt4HjDr+0vMsvQgHFX0B+Wf50+Sfzi8sL5y8neYYNV0X97yuog6mJoVDSJLFIqyRuqkEo6K1CDShGKsI/KX/nK38qvzz07zLq3kjzamqWfltFk1WX6neWwtkdJZFYrdQRMylYpCCgYfCcVUvyR/5y0/Kj/nI281TT/wAu/Ni6zc6ZFHPdxfUL60MccjFFYfXLeEMOQp8PKm1eoxV8i/8APxP/AJzV078nPKeq+TvI/na0sPPst7ptnPGgMl1plpdJ9YkuCDGyqfS4AHdlEoZRy4nFWGf84r+Xvy31D8xNAuvK3/OWvm3zxrlrDPPdaNfaxLPZ6iqwsknKC4j+wpfmF5Oy0DBqryCr9WPN2lNr2ha1piV5XtldWwoaGs0TIN/pxV8bf8+2vIGuflr/AM4++StH8y6Rd6Tqnq6vPcWN9BJb3MHq6jcmMPFKAyco+DgEDZge+KvurFXYq7FXYq7FX55/kV+VWveXP+cpf+ci/Nt7pF5b6Rqtl5XXTb6aCRLa8Z7OMziCVhwkMMkJVwpJSo5U5Cqr7c8++f8Ay7+V2g3/AJn816vBpOkWAjNzeXLFYo/VkWJK0BNWd1UACpJAxV+DOqf85HeWP+cmvzN/Mybzr/zktrPkHydo13HY+VrLy9ezWCahAOavdSOkTeoCYw/7xS372ilFUAqv2a/5xr0rTPLv5a+X00z8wb/zxpjRT3Nt5g1a6FxcXFvJK7jlMQpKx1KjnVlC8TQAAKvDfM3/AD8u/wCccPKerzaLefmNFNPBI0U01lp9/e2qMPC4traSOQe8bOPfFX2R5O856F+YWj2PmDyzq9tq2l3yepbXlnMs0MiglTRlJFVIIYHdWBVgCCMVfkD/AM5Efkzqn/OOn5l/kTrugfm3581FfNX5jaNY3+n6t5jkuLL6tcXkckkSxRxxVRgShVy4KHia9cVftJir4x/Mv/n4L+Q/5TeYL7yv5g88Aapp7lL6Gz0++vltGDcWWWS1gkjDK2zKGLKdmUHFWW+Tv+czfyd8/eVfNvnbQ/OK3OgeVfT/AEvemwvohbiRQykRyW6yuDWg4IxqCMVTX8j/APnLT8qP+cj7vVLH8uvNg1m40yKOe7j+oX1oY45GKK3+mW0PIEinw1p3xVi/nz/nOf8AI/8ALLzjP5B8y+eVsfMME1tby2f6M1KYJLdKjxK00No8I5LIpr6lFr8VN8VZB+en/OXn5Tf843XFnY+f/Nsem315F9Yt7KO2ubu5eEsyCQx2sUhRCyMAz8VYqwBJBGKo78iP+cq/yu/5yUj1A/l95pj1ObTwhu7V4J7W5hV/suYrmONmQnbmoZK/DyrtirHfOv8Azm3+SP5cebtS8ieZvP8Aa6Xr2nRGa6trm2vEjiX6sLsA3PofVy7REFUEhd2IRVLkKVXfkl/zmr+Tf/OQ+sXPl/yL5xjv9Ugjab6pNaXdlLLEn2nhF3DF6gHUhaso3ZQMVev/AJo/nD5K/JXSRrnnnzLZ6HYM4iSW7l4mWQivCKNQXkalTxRWNATSgOKvxq/5zE/Pr8v/AM1T5W/Mj8ovz01x9Vg13y/o1xoFhq91p1ibSWed3uG0+WKCcuxAV5AShXiGWtCVX7vYq/PP/nCv8p9d/Lv8wf8AnJrVdX0q8srfzB52mutPluYJIo7u2Zri6EsBcASR1uuPNarVStaqQFX6GYq/L+L85Pzw/LX/AJyR/Lf8rvOfmjQ9b0LzjBrV4BY6O1nNbw2ttdyxJzaVzyDQpU1YEVHXfFX6gYq7FXYq7FXYq/I7/n4rqMf5f/m1/wA4t/mHrgYeWdF8wXlvqFwy/uLKS5ksnSVz2+CJ5PlAaYq/WRtQszZm+a5i+qGL1jOZF9H0ePLnzrx4cd61pTfpir8k/wDn595s0jzD+T/5bjQtbtpPImt+b9LtNXv9HuI5bV9PjW4JVXty0bRq8TMaGgliQdRir9E7/wDKf8sn/L6XypceXtHXyWunkNbejCtiloI+XrBvsrRfjEwPIH95z5fFir4W/wCfSl9d6t+TPmDTbiaW+0PTvNWr2WhyXQrz0307aUKFYU4erJI3hyZx2xV5p/zhf/zjj5A88fnL/wA5G+dfMXl+21S+0Tz3rGn6Zb3cSS2VorXEsrSx27KU9UniA5B4BRw4kklV+t3l3yF5Z8ntqT6D5d07S21OQTXxsbGC2N1KF4B5/SRfUbjtVqmm2KvyU/K7/nHXyF+av/OYf/OQGp+adBg1KLysvlmTTtPmjRtP9e902MNNLbleEjIIvgDVQF2YqXCMqr9GvzM8oaZ+XHkr8yvM3kTyzpuneZX8u6nJFdWFjBbXNzcWlnM9oskkKK7hJKcQSeNdsVfkD/z720z8+tM/Ky21z8rPKv5c6hbaxd373+p6zd6gdauLhLh0aO9aE0HAAcE6emyyU5SMzKvqP/nDr/nHn8zfyo/O38xPNvmo+UNEsfM+krLceWvLN/cNGl8k1v6V0trMlY0NLgs5YgPMQi8TRVWD/wDPqvyl5Z876X+ZfnfzXY2up/mNP5o1CDW5r+JJry0jZY3WNPVDNHG8hlqVpzZOBr6ShVX1d5d/5xu/JvyJ/wA5B/438vavZ6R5vvdFuhN5YtLi2iSeN2RZL8Wg/eISAAxUBHPx058yyr5Z/wCcvPyW/LbS/wDnIb/nGKAeRdAgtfM2reaf0zGuk2iR6pKYrJozdqIwJ29SUkeoGPJyepOKv0s8oflB+WH5R3317yx5M8u+Wr6+X6n6+n6ZZWE9wrESeiHhjRnBKBuFTUqDTbFXrmKuxV+Q/kPWLf8ALj/nPb8y7XzM4tn87eWtO/w/cT/ALj0YLANBGTQbm0mA/maGn2jTFXoP/OXn/OTX5/8A/OM9p5g83w+TPKV55Ns7y3trO4lvr1tRkS4KohliQxqDzJqAaD364q+ZP+c+R5/85fmx/wA4yeWNY0vy9PdXdpLePpOq3dz/AIbufMA4+rBKyhXljDLGkIYVcyCNhxkYMqz/AP5yP/JX/nJv/nIHyJd+UvN3k78qrHT4RFLb38N5qMU2megyt6lvJICsXwqUbbiUJBBGKv0C8peWtS0/8jNI0fzbe2mualY+VI4bu9ikN5b3U8FhwM6SyrWTlSvqEAsSW74q+Hv+fV//ADjd+Xifkr5c8/3vlq01PzB5iOqi4vNQgjumggttQurNYLYSqRFGyxcnC7u7NzJUIqqp3/z8s/L/AMseXPyi8g6HpHl3TtP0xPPWgxJZWdlDb26R3JummVYokVVEhYlwB8RJJ3xVLfM/5X+VH/5zs8mRroFlHDb/AJctqSwJbRrCbtLy+sUkaMLxJWGirttxWn2Rirzr/nG782dL/wCcZ/yc/wCcqvNL6Ml7pvl78x/MUFtpahY4ZGuXsrCCA/CwWIs8at8JogNFPTFVC7/Lv8+dW/JDW9eu7n8q/Jvk/VNAn1GTy9aaDwtI7C4tjMnqTxgqswRgVZGko9CGrir68/59lzvrP/OMX5Z/XqTgw6zARIAwMUOsX0KKQdiAigU8Birwv/n1F+VPlGb8nda1q48u2Nzf63rmr21/PcW0czzW0DLEkBLqf3QFTw+zVmNKk4qy7/n1NYpof5dfmToVszmy0X8xPMWnWSOxYx28NvYlVqeu7En3JOKv0/xVLtYtJb+wvba3mMMs0Esccg6o7oVVvoJrir8u/wDn0Tq9rbflBrPk6dPquveWfMep22r2Mnw3EMshUq0iHcfZaOv80TDtiqX/APOT97F51/5y/wD+cbPLWhOZdT8vRapqurGFS/1aymUOgmYbIGW3cUJB/ep/OlVWU/lJ9V0//nNz87LVRHDJc+UdEuI4wAhkVFshI4ApWjMOR8TviqHmfS1/5zx02200W6tbflnIl3HAqrwuH1GaakgWlHMckbb7lWU9CMVYV/z8e/KbyFY6z+SOvN5O0WK61r8ydBt9bvf0bbLLf20pb1Y7uX0+U0bqvxLIWBA3xV+iHlv8j/ym/Kq/i13QfIfljy7fCtvHfWek2NjOPX+D01mjiRhzrx4g/F03xV+bH/OeXlDyT5w/5yU/5xo0nz4kA0PUP0tHeCYhI7iRPTa0glbaqS3AjjIJ3Dle+Kso/wCfoH5Sflb5T/JG+8x2+iaX5f8AMWjXemDy3d6bbQ2N39ZN1EGgiMAjZlEHqPxrROHqgVQYqlH/ADmRqd7qHk7/AJxA8yeZolg1STzn5Hn1SaRQDFPLbCa4VnoKAOGJGw2rTFWd/wDP2iTSrX8m7FrkW6X115j0SC1Z1T134SSSuiE/FQKrMQNsVfV3/OWH5ZeUvP35bedb3zF5W0zWLzSfL2vzaZPf2EF1NZzGykbnbvKjNExaNDVCDVVPVRir5X/5wN/5x3/KDzv+Qv5a69q/5a+V9V1C5sZjc313odjc3Es0V1PGxkllhZmZSnE1J6UxV+mOnXlpf28U1hPFNbkFY3gdXjIQlSFKEjYgjbpSmKvyJ038hv8AnG2P/nLGXX4PzPuG8+DU5tVPlgPS3GqsjySD6yIKEipk+riX1A2x+D93ir9g8VdirsVflj5a/wDW+/Nf/muYv+oqxxV+p2KuxV2KpPe+XdK1KQzXmmWtxIQBzlt43ag6bspOKvzj/wCfftvFa+dv+cqIYY1jjT8xtTVERQqqokmoABsAMVQ/5SfVdP8A+c3PzstVEcMlz5R0S4jjACGRUWyEjgClaMw5HxO+KoeZ9LX/AJzx02200W6tbflnIl3HAqrwuH1GaakgWlHMckbb7lWU9CMVSb/nIv8AL3y958/5zI/IWx17Sbe/tP0DrV7JbzxI8U81olzJAZVYEOI3VWAPdRXaoKqYf85A+VdL8r/85e/84x63pNlFZ3mq23mmyvHgjWMTQ2tg/pBwoHIgXDiprtT+UYqxf/nKuLy1rH/OW35KaV+apt28lf4fvZdKh1Ij9GTa+086sk3q/uiSq21FbZn9JTXmFKrCv+fj3lH8pNI8yfkU2kafo9p5xm856JG0NhFBHNLoplPqm4jhABQTeiIi47yBNueKoX/n5Ha+Y/M/55f849eULCx0e90+YXlzp+neYZpodEvNURwBFdLCQX2WJI135NL6ZHGRgVWQ/wDOR/5K/wDOTf8AzkD5Eu/KXm7yd+VVjp8Iilt7+G81GKbTPQZW9S3kkBWL4VKNtxKEggjFX6dfkNpGt+X/AMuPIul+Y9Vg1bVrLRdNtry/tp2uYbqWKBEMyTOA0oeleZALk8u+KvWcVdir581b/nKn8q9D/MGD8q7zzbEvm+aW3gTSktbuWT1LmJZolaSOBolJjYP8Tjipq1Bir6DxV2KuxV2KuxV+S3/OVf5c+U1/5yb/AOcX7ceWNLEOrXPmt9RjGn2/C8dbaF1a4XhSUhiSC9aEk9cVfqD5Z8keXfJazp5e0DT9JW4KmZbCzhtRIUqFLiFF5EVNK9K4q/Kj/nM78nPLH5xf85P/APOOnlHzDZFtL1az8zTahHbt6D3Qs7Z7wRySIAxWRogj0IYozAMpNQq/TTRPyY/L/wAtz6Pd6T5H0OxuNGR49NnttKtYpbJJFKOtu6RBowykhuJHIE164q/NvXPP/wCYf/OR355+fdE/KLy55J0Wb8vTb6TqXmzzFpgvtZkd2mKxWoVSywrKk3FT8Pwl+YLCPFXnv/ONOlebPKv/ADmt5tsPOGuaDqmtXPkeR9Sn8vWptIJH+tWRQXUJHw3QVULdfgKGu5xV7r5mNtp//OeXlHkEha9/LOdU2CmaZdRvnYD+ZgiEnvxXwGKob8/n0uH/AJzG/wCcZLS0Ful6LPzdc3aRqiy8JtOnWKSTjueRjkCk9eLYq/UXFXYq7FXz5+XP/OVP5V/m35r1XyT5Q82xarrulpcyXlpDa3aiJLWZYJW9aSBYmCyOq1Vzyr8NRir6DxV2KuxV2Koe7/uJv9R/1HFX8p/5F6f5i0nyB+Ufm/8ANzTp9c/IfT9c1GGXTtOmfjZ3z3Lhb/VbeNOU8Incqq8yOKlKAyCO4Vf1JeTW0CXRNMn8rLZDRp7eKaxOnLGto9vIoaNoRCAnAqQRx2pir88/+fhH/KWf84r/APm0vL3/AFERYq/QH8xL7UtM8q+ZrzRk56lb6Xfy2S/zXMdu7RD6XAxV+Ef/AD7tsfz3tvy7uPM/5XeW/wAvdVOs6jfnVNW1281BtcmuVk+KK7eJjQAEOiVoVkEhHKRiVX1Z/wA4pf8AOO35oflz+fvnLzz5tXyboNp5m0VxfeXfLN9cFJLhZbf07sWc67fEshaStA0rhRWRsVYf+Yn5C/lZpv8AzmL+Vvldfy98uQaJqnk/VZ5dMXRrNLK5vInvHEj24iEbyKiVDFSwCjwxV+oXkn8sfy7/ACluJbbyj5V0Hy3carQPHplhaWEl2LcMwqIERpPTDMR141J2qcVfmN/zl5+S35baX/zkN/zjFAPIugQWvmbVvNP6ZjXSbRI9UlMVk0Zu1EYE7epKSPUDHk5PUnFX6WeUPyg/LD8o77695Y8meXfLV9fL9T9fT9MsrCe4ViJPRDwxozglA3CpqVBptir8bvPtv+ZPnL/nN3z3/hrRvK+tav5c0XT/ANB2Xm+a5Fpb2jWtnK09isP2plkmlY7EIZJCPiWoVZ3/AM5AfkF/zkn+cut+RfNXmfT/AMtfKuq+WtWtLm016w1PULa7IDgi2eSZX9RCwDKn2uQIQgO4ZV9b/wDPzyxtp/8AnGz8yp5beN5YYtJ9ORkUuldWsgeLEVFR4Yqzb/nHf/nEr8p/IH5baHotv5J0vUU1LTrSfU7jVLG3vrjUJZ4lldrh5o25jkx4pQInRVGKsH/5+D/lN5L1T8jvP+uXPk/SJ9V0TQGh0u+fTbdrqxhikUqltKY+cSJU0VCAKmnXFU1/5xj/AOcefycufyr/ACp80yflr5Va/by15e1GTU5NDsDcfWfqMEzXDztDz9USVZnLcuVSTXFX3Da3UN9DFcW0yTQyqrxyRsHR1YVDKykggjoRiqvirsVfPmrf85U/lXof5gwflXeebYl83zS28CaUlrdyyepcxLNErSRwNEpMbB/iccVNWoMVfQeKuxV2KuxVQurmOyhmuJm4xxI0jtStFUEk7ewxV+N35F+Y/wA6f+cqG8yfmj+Vmk/l/wDl9oGq6jdwxT3Wire65qTQng8t/JEOJZgaNUqetFZaSOqnf/PqyK803V/+citKu73TLprXzarO+iqV0o3EhuRM1mpA4wkoAgGwVQBsBirPv+cLo7ay/P8A/wCcvNPVI4T+mfLsiwBQv7ow3/xBaD4fiFaeI8Riqz8h5dLb/nMz8/4NJEAhtvLvl6CRbdVWNJkhtfUX4AByUmjU6NUHcEYqk/8Az9l8m6BJ+VVhrbaHYtqcvmbQoJL42kJunhPqqY2m48ylBTiTSm2Kv0j8vfll5P8AKN0b7QvKmk6ZdFDGZ7LTra2lKNQlecUatQ0FRXFWb4q7FVKeaO2jkmldUjjUu7saKqqKkknoAMVeDfkz/wA5Rflf/wA5CXWq2f5e+ak1ubSkikvBFZ3kKwrMzLGS9xBGp5FGoFJOxPTFXv2KuxV2KuxVLtX0ew8wWc+n6pYwXtpOAstvcxJNDIoINHSQFWFQDuMVfk9/zhj+VfkrWPzh/wCcq7S/8oaPdW9h5nsIrOGfTLaWO2jIvSUhV4yEU0GygDFXvn/Pxi6v/I//ADjP+YUflOEWCx2un2RSzQRCGwuL62t7hY0jAAQwuyMAAAjMegxV8vf845eV/wDnIzRfyl8q6N5J8g/lLdeVtR0m1lBmn1GRtRS4hUvLeBapJNJU+qDUBqoAFAUKvef+fdv5D+df+cf9P/MfRfNWo6EbW81pL6x0jQtQmvIdJeVX9eFhMoaMFfRCKSzkIS5J3KqV/wDPwj/lLP8AnFf/AM2l5e/6iIsVfplir8PfyQ8l/nl+VX+Orn/nHi68g/mL5O1/zHqt1Lcak1zFqUNxyEclrcky226AACrSqwPqrxWWmKvb/wDnAPzBoGo6/wDnX+Xur/lNZ+R/NVteWtx5k02yujd6VeC5jeMG3iZnSGOhJ9NGaMrKGU0+FVXw7o35iedPyK8l+ev+cafy6vXsvP8Aa/mFPp2izQFY7w+XpYZdTScPxZjUQnkabJOBt3VfQX5D/mJbf85tfnh+SvmWWOC6i8g+RW1nWpVjTifMl+31CSFwBUFTGs8QPTh4faVen+fPP3n3/nID8+vNnlH8pfLPkrTrz8vba0tNT82+ZtNW+1QG55SpBYhVLrErl/hI48gzl05KrqvLPyU0jzn5T/5zc+pedtd0DVdcuvJFwb+58vWhs45FMweNb2I9LqkaMetYvQOKvW/+csfIPl/z/wD85Xf84yaRr+k29/ZXVv5mnuIJoleOd7Gza6gEqkUdVkiU8WqpHwkUJGKoz/nKryrpXlj/AJyc/wCcT9e0mxhsr+/vPMOn3MsESRmW2gtoBGjcQKhRcygV7MRirDfzG0jQfzS/5zm0Pyx+ZUcN3pGj+UPrnljTL4q9pdajI4d29GT4JH4idqEGpt0JrwACrGv+fpv5X/lv5btvym1/T9J07S/NUvmzTbSH6nBFbzXenAO8/qJGF5pFIIeLkHgX4inqHFX7T4q7FXYq/LD8/v8A1tj/AJxq/wC2N5k/6gdQxV+p+KuxV2KuxV2KvPvzR/Kzyt+c/lvUfKXnLSItU0m/UCaCXkpDKarJG6EPHIh3V1IZT0OKvz6t/wDn1V5JWGLRbr8z/Pt35ViZGXy5NrqfUCFNfTKJAo4dBRVVx/PXfFX2/c/848fl1d/l2Pyok8q2v+DxbC1XShzEaoJPWDBw3qCX1f3nq8/U9T95y574q+Lh/wA+uvKDWa+XpfzX/MOXykpBHlp/MS/o3gH5iL0hbhfS7UCh+/Plvir668wf844eWL78tofys8uXuq+TdEt0gS2l8s3xsb63WKUTEJcOspJlavqlw7ScmLEseWKvkTRv+fWnkzy7LqE+k/nB+Z9hLqM7XV7Ja+ZraF7q4cktLM0enqZJCSas1WPjir7p1n8rYda8gnyA3mTXLeE6ZBpf6Zt9RMet8YY0jFx9cKEm4bjV3KHkxaq0NMVfC1j/AM+tPJml6jqOr2X5wfmfbahqPpfXbyHzNbR3F16K8Y/WlXTw8nAbLyJ4jYYq/QH8tvIsX5a+WNH8rw6xqesR6bD6K32s3f1zULgcmblPNxTm29B8IAUAAUGKviTXv+fbnkxNc1bXPIPn3zn+XY1eX1r+w8ra2bKwlcklmEXpsykkn4eZjUbJGo2xV7H/AM48/wDOGP5c/wDONd/qWu+XY9Q1HzDqcLW99rms3z3moXETOsrIzUSNQzopbjGpbivIniMVeffm3/z758i/mP5uvfPugeY/MfkTzJqAb6/e+VtT+oi9ZyCzzoUb4moCxjZAzfE4Zt8VZx/zjt/zhX+Xv/ON+o6j5h0c6jrfmXUkaO81/Xbv67qMiOwZkDhERASByKoHai82agxVmn/ORP8AzjJ5N/5ya0jTNM81fXbW40m6F7peqaXc/VdR0+4FKvBKUdRXiKhkYVVWoHRWCrwX8uP+fc/5d+S/Nml+efMHmLzR5317SriK60+68zay12LWeFg8ciLGkXJlYBh6hdeQBptir7/xV2Kvm7/nIz/nFL8v/wDnKHTrGz842U8d5pzF9O1bTphbajYsxVmMMpV1oSoJV0dKgNx5AMFXy5ef8+yPL/m1rO28+fnB+YXnDSLSVZYtJ1bzB6toeJ2DgxFum1UMbeBAxV9e/wDOQX/OOHkf/nJny8nl3ztYSSx28v1iyvLWUwXtjccePq28oBANOqsro23JGoKKvkqT/n2R5U8wm3tfPH5q/mF5w0i2dWh0fWPMjS2XFfsq6pErbdjG0ZxV9C/mz/ziV5a/NHyx5W8nWfmbzN5O0fy7C1raW3lXVv0eslsYkhWG4EkU4lVFT4eQ5CrfEeTVVeE/l/8A8+0/Kf5ZS6L+gPzb/Mq2s9JuY7mDTY/MsEVgSk3rsjQQ2EY4SOSXC8eXJt6muKvZP+cnP+cN/LP/ADlZLobeZ/NnmjS4NIJeG00TU4ba1km5cknkhntrhTNHUhJF4sFJFcVSuf8A5wi8qXPnL8uvPj+cvNx1vyTptjpMFz+mU5anaWVzJdKmpObb1JxI8rCUK8ayIeLLTFVDQf8AnA/8utEuPzT9a/13UdL/ADGkvJtX0W91FW0yKe7uBcvPawxQxtHMrheErO7oFUBtsVeO+W/+fWH5b6aLTT9f86ecvM3l6xfnZ+XtT1ymlRUHw1ht4oiSp3HFkHZlYbYq9d/Kf/nBjyz+TPk3zZ5G8u/mB52TTNfgjgRn12NZtJCSSys+mGC1iS3eR5WMh4N6m3Ku9VV35I/84M+VvyC8tecfK3lnzz5yNl5mtHtXa41mH1NPaRZFa50/6vawpBcH1KmTgxJVa9MVTT/nGX/nC3yv/wA4r6hrd95Z83eatUTWFc3NprWqQ3Fp9YkdHe6EMFrbqbhuCq0rcmK/Dir7DxV2Kvhv83v+cC/J35kebbrz95d8z+Y/IPme9jMd9qPlTUzp5vgSpLXChTyY8RUoycju/MiuKs9/5x3/AOcQfI3/ADjfcavq+jzajrfmLWajUfMOu3YvdUuULK5jMoRAqclBIC1YhTIzlVIVYz/zkB/zhD5O/PzzTpnno+YPMHlTzRYWws11by3qC2VxLbgsQkhaOTdebAMvFuJ4sWUKAq858p/8+z/y28j+cfL/AJ80bzb5zttd0t4Jbq6GvKz6xLHP67tqLvbGSQTUVJUjeKN41A4BuTMq+qvz7/IHyf8A85JeVJvKHnSzlmsjNHdQTW8vo3VpdRBlSeCSjBZFV2XdWUqzKykEjFXyd5f/AOfZf5cw65p+u+cfN3nHz1LpssU1lB5k11rm3haFg0fwxRRM1KCoL8GAoUpUYq+Rf+c+/Pn5L+bf+cjvyo8s/mPrEEvl/QtO8w2vmmPjdf6E9/p5lsqtboX5mQxOpj5FG4l6DFX0P+Q//ODX/OPP5jHQfzM0TzLr/wCYOk20sh0mPXdVkurC3e3k9NkSCS3gk4o8f2JeSmgqrLTFX2//AM5Bf84++Uf+cmPJ915L8528z2MksdzDPayCK6tLmIMqTwOyuodVdl+JGUqzKykHFXxdq3/PqX8tvNunfVPNvnzz15iuofTSwv8AUdfSafTYkYFo7VZLZolWQAB+cb9Bw4HfFX6A+R/y9tPJXlLTvJ8upahrtraWjWcl1rdz9dvbuNuQb6zKVUSFgxB+EDjtSmKvge9/59YflxN9Y0yx89eeNN8sTzy3D+WLTX1XSQZW5MixPbu3AjY8mZyP92VxV95flR+VPlj8kvK2l+TfJ2m/o/R9NWQQQerJK3KWRpZGaSVmdmd3ZiSe9BQAAKvl62/597/lXafnI/52p+k/0219Jqn1E3UX6NGoSVLXPpiD1eXIl6etw578e2KvuXFXYq7FX4t/mD+d3kr/AJx8/wCc3/MHmT8wNa/Q2l3XkO1soblrW5uFe4lntnVeNrFKwBEMnxFeI40JrQYq/Z+CdLmKOaJuSSKrqfFWFQd/bFVXFXYq+MPzj/5z/wDyT/IfzJqvlDzf5lubbWtOiglmtItKvp6+vEs0apKkJiJZHU158RWhYEEBV5D/AM+2tI1/VNN/Nz8ydZ0O60a2/MHzlqOu6Va30ZiuDYTEyRyFafZPqFVb9vgWFVKsVXsH/OQH/OEPk78/PNOmeej5g8weVPNFhbCzXVvLeoLZXEtuCxCSFo5N15sAy8W4nixZQoCrznyn/wA+z/y28j+cfL/nzRvNvnO213S3gluroa8rPrEsc/ru2ou9sZJBNRUlSN4o3jUDgG5MyrLfPv8AzgV5V/ML8zovzXvfP/ne11q3niltYLLXIorO0RFVWt7dWtHliglofUjWUBub9A1MVa/Oj/nAnyp+eHn6H8xdV8/+d9M1S19L6lDpGuRW1tYcIUhf6mHtJJLf1ggMvpyLzYs3U4q9D/5y78r/AJOav+XWoaj+d+nR3PlnSHgme5aK6e5tZZpUgR4Xsh9YVnd1VuGxB/efBXFX4p3PlH/nHv8AMPzl+U/kr/nFzyze6vqaebdI1zX9dkt9RYWOl2DtzSSXUVVo4z6vNuCKjtHGpLyemMVfuz/zkF/zjh5H/wCcmfLyeXfO1hJLHby/WLK8tZTBe2Nxx4+rbygEA06qyujbckagoq+SpP8An2R5U8wm3tfPH5q/mF5w0i2dWh0fWPMjS2XFfsq6pErbdjG0ZxV+hvlTytpPkfRtL8vaFYx2OmaXbQ2lnbRV4QwQqERByJJoANyST1JJxVP8Vdir8a/+cI/yH8w+dvz+/OX8+/PmgXmmyRazqWneXrbUrWS3mHqsUM6pKqtSKzEcKOKq/qS91xV+ymKuxV2KuxV2Kvza84/8+yPJnnjzGfNWpfmr+Y51GO6u7qzkXzHbsbBrpi0iWjy2DyRJQhQA9eIAJNMVfUX5Af8AOPFn/wA4+2usWln5281eZxqcsErP5o1cak9uYVZaW/GGERhuVX2Jai1PwjFXyprn/PsDyb5g12DzLefm5+Zj6pbPcNaXbeZoHntFueQkSCaSwaWNWVipAf4l2YnFX1x+Qv5DWv5BaXqWlWnnPzP5nW9uRcmfzPqo1GeEhFj4QssMQRDSpFDVt64q+efza/595+S/zK886j+YOk+b/NPkzWdWCrqr+WtTWyS94qELMDE5V2AHKh4MfiKFizFVJvLv/Psf8q/JXmPy75q8r6/5t0LVNJWMXE+n64EfVmW4+su2oO8DvJ6rUWRY2iRlAHAUrir2P/nJL/nDryd/zkve+Xtb1XU9X0DX9A9QadrWg3i2l9FHIamMu0cgKhviWgDqS3FgGcMq8C/6JaflsdW0fzN/jrz2vmawlkll19fMSnU71nCqBNO9s5UIgaNfR9JijsHZ/hIVfpfirsVYD+avmDUvKfkrzdrejafPqGpadpGo3dlZ2sLTT3NzDbySQxRxoCzs7gKFAJJNMVfnL/z6i/5xnvfyc/LzUPOnmfT5rXzL5yn9V4ruN0ubbTrZnWCN1k+JXlcyTN0LK0QYVTFX6sYq7FXYq7FXmP5wflfB+cXla/8AKtx5h1rQY7toWa/0C/8AqGoRiKRX4pNwkAV6cXBUhlJHvirx3yz/AM4d+TPKf5L3/wCRen6prKeXr2G9gmvDdQfpMreTtPLSVbYQipYpT0ePDZgSSSqp/wDON3/OH3l3/nGCy1bTfLfnHzXqdjfwJBHaaxq0c9vYhWkctZxW9vbpC7NISzAEk096qvnvXP8An1t5M8zzWFxq/wCb/wCZ9/Lp9yt5ZvdeZ7adrW5Q1WaFpdPYxyKQCHUhh44q+qfyA/5xts/+cfTrZtPPvm/zR+lBbBh5p1kaktt9X9ShtwsEIjL+p8Z35cV8MVeCebP+fc3ky88x6v5n8jeePOH5d3OsymbUbbyrrRsbO5kZmZ3MXBmUsWPwh/TX9mMb4q9R/wCcfP8AnCj8uf8AnHPVr/zNov6T1jzLfxPBda7rl+17fyxSMrspIWOIVKLVhGHIABYjFU6/5yI/5xM8nf8AOSMug6jrF7quia5oLyNpmuaDe/UtRthIPiRZCkgKEgHdeSmvBl5NVV5z+Sv/ADgB+Xf5Oebbbz/PqvmDzZ5qthKLfVvMeqteTQerG8T8FjSJTVXcD1BIRyJUg74q9g/5yJ/5xk8m/wDOTWkaZpnmr67a3Gk3QvdL1TS7n6rqOn3ApV4JSjqK8RUMjCqq1A6KwVeC/lx/z7n/AC78l+bNL88+YPMXmjzvr2lXEV1p915m1lrsWs8LB45EWNIuTKwDD1C68gDTbFXp3/OQ3/OG3kP/AJyM1HSPMWqz6poXmbSFEdj5g0C9+o6lDEGZhH6hSRWQMzEVTmnJuDryaqrxzy//AM+1vID61p/mD8wPOHm/8xbzT3SS1TzRrkl3bQujBlIjRUcioFVeRkalGQioxV6J/wA5If8AOEXlr/nJ7UXvPMvnvzlp1nJaQWkukaRrMUGlSiCUyrI9rPazoZORBLf5KGlRXFVL8mP+cI9I/JPzLpfmSw/NP8wdZGnRzRRaZrXmNLrTHSWFoAJLeO1iDBA3JByorqrU+EYq+tvNXlbSvO+jap5f12yjvdM1S2ms7y2lrwmgmQo6GhBFQTuCCOoIOKvzjH/Pqv8ALmWO30q88/8Anu88s2zFoPLk3mBTpkYLFjGsa24YJU1+Fleu5c4q/RLyR5K0X8udA0jyv5dsVsdK0m2itLO2V3cRQxLxVeUjM7HxZmLE7kk4qynFXYq/Gv8A5wj/ACH8w+dvz+/OX8+/PmgXmmyRazqWneXrbUrWS3mHqsUM6pKqtSKzEcKOKq/qS91xV+ymKuxV2KuxVogMCCKg7EHFX5lT/wDPrH8uYNT1h9D86+cvL+gavO9xeeXNJ1lbbTJC9AY+HoMxiIFOLFmC/CrqAAFXrP5Mf84D/l7/AM4/+eJPOvkXWvMmlxSBg/l9NX5aHJW3a3BlgaEzTFOTOplnfjIeQ6ABVCfnH/zgN5M/NjzxcfmLYea/NHk3zBfQR22o3XljVFsDfRoqRj1eUMhDcEVSVIUhQWVm3xVAflP/AM+6/wAuPyT89WPn/wAo+YfNVjewoFubP9NCSy1FuDK73qtAZZjI7GVl9UR+puqKBxxVX/5yC/59/eUf+ckfMF9rvmfz/wCd7aK7a1k/RFhrcC6TBJbRLCjwWtxZziNiF5Eg/bZmFCxxVm/5G/8AOIOnfkT5hl8w2n5mee/MTyWktobLzH5gW/sQJGRvUEKW0X7xeFFYk0BbbfFX1zirsVfN3/OYOta7oP5K/mXc+WdNvNR1aTRbu0tILCB57nneAWvqRxxhmPpCUyGgNAhJ2GKvDv8An2z/AM43v/zjx+T+mfpWyNv5i8zFdY1VZEKyw+otLa2cEBlMMNCyndZXlGKv0CxV2KuxV2KvPPzW/LqH82PKureVLjXNW0SPUViVtQ0O9NjqMHpSpN+5n4vx5cOLAqQyFlIocVfAuk/8+r/JOg3WpX2mfm7+ZtldalIJr6e28y2sMt1ItaPO6acGkYcjQsSdz44q+/NG/LXSrDyXbeRNTlufMGmJpo0q5fWpvrlzf25i9F/rchVRK0ik8zxFa9Bir4RT/n2N5V0UXWneUvzZ/MPyt5eupJZJNB0nzIY7BfVNWWNXhc8fH1fVY92OKvqn8hv+cXfIH/ONuhaloXkbTp7Maowkv72W7kmvbqVUKLI8rn4SoY8QgRFJLBQSSVXybrn/AD628meZ5rC41f8AN/8AM+/l0+5W8s3uvM9tO1rcoarNC0unsY5FIBDqQw8cVfVP5Af8422f/OPp1s2nn3zf5o/Sgtgw806yNSW2+r+pQ24WCERl/U+M78uK+GKvlLRv+fXXlXybdapqHk382vzA8r3ep3E012+ka3FaJKsjs6oyw2yFhHyIUliep64q+p/+cdP+cVvJv/ONFvrbeX59R1PVddmSfVta1m7+t6jfPGXKerKFRaKXYgBRUsSxY74qlk//ADiN5On/ADyt/wA+jJcDXodNNj9WAj+rNN6LWouj8PP1Rbt6P2qcQPpVd/zjb/ziN5O/5xgvvPd/5WluZX826iL6Zbj0+NrCjStDaw8FX93GZpKE1Ygip2rirzX85v8An355M/Njzxc/mJpvmvzN5M1++jSHUbny1qS2f11EVY6vWJyrlEVSVIVqVZGbfFWJaV/z69/Kry1rOgeY/L3mPzhomsaZVrjUbDXhHdaq7Tes730r27sxc/C4h9FWXqtd8VZ1+a3/ADgX5V/Nz8xoPzP1H8wPO+n6vavE1lDpmuRW9rYhIkhdLRXtJJYEmVSZRHKvMu525Yqqfnv/AM4HeVP+cgvOtj571vz3500zUNPRFsINI1qG2tbFlVUd7VZbSZ4HlCL6hjdeZAJ3xVmH/OR3/OGvkD/nJyy0NPM7ahZ6tofEabrmm3Qh1OAKVahkdJEcFlDfEhKtVo2RmJKr5s1H/n09+V/mSyH+JPOfnTWdaE1tKmvXWtRyahBHbiSkEJltpI1iLPzIKM4ZVKyKKgqv0W8heUY/IPl3RvLkWqahqiaZaxWq32q3P1q+uBGKepcTcV5yHueIHgAMVZdirsVfir5j/PTyP+e3/Oan5ES+Rdb/AEumhWPmax1FltLqAW9wtjqFUP1iGPkP8peS++Kv2qxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxVjGpeSfLus3D3eoaDYXdw4AaWezhlkYKKCrOhJoNhviqdafptppNvHaWNrFbQR14RQxrHGtTU0VQAKk12GKo3FXYq7FXYq7FXYq7FXYq7FVN4Y5CCyKxHQkA0xVUxV2KuxVLLjRdPu7mK8nsIJbmLaOZ4UaRB/kuRUfQcVTPFXYq7FXYq7FVC5tYb2KSC4iSWKQFXjkUMjA9QVNQR88VQ2m6RY6ND9X0+ygtIq19OCJIkr48UAGKphirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdiqmsMaMXVFDHqQBU198VVMVdirsVdirsVfKv/OUv/OS95/zjNpNnro/LrWfNGmmO6m1C80wxCDTY7f0qNcs9Sof1DxNKfC2+KvmaD/n5c+naNYea/Mn5C+e9L8r3kFvdrrkdgl1Zra3Cq8dwzgxqI2VlYNy+IH4amgxV+inkHz5oX5n+XdH81+WdQS/0nVrdLq0uEBAeN/FWAZWUgqysAysCrAEEYqy1mCAsxAABJJNAAMVWwzR3CJLE6ujgMrKQVYHoQRsRiqjJfW0M0du9xGs0gJSNnUOwHUqpNTT2xVFYqw78wF80v5c1lfJL6cnmE27jTW1cTGwFx+z9YFv+84ePGp9j0xV8l/84R/nv+YH5z2/5nad+Y8ekLrHk3zTeeXXfRYZ47WRrMBJGX15GZgZA3FiEJWlVBxVn3/OWFj+dl/oPltPyOvrW01Ya3atqj3QtSp0sRy+oP8AS0dac/T5cB6lPsnrir6nxVQubqGyjMtxMkUYpV5GCqK7DckDFVVXVwCrAgioIPUeOKvnj81P+chrL8rfzC/Kf8v7jR5rubz/AHGrwQ3aTKiWf6NhilJdCpMnqGVVABXjud+hVfROKrXdY1LMQqgEkk0AA7nFVG1u4L6NZraZJo26PG4dT8ipIxVEYq/On/nL789Pzp/IHzX5F1PRR5Xm8jeYPMGheXGiuIL2TWVur5pXmY8XjgEfpxNwYMWDUqh3OKv0WxV2KvHvz0/PXyh/zjp5SvfOfnXUGtdPt2SGNIozLcXNxICY4IIxTlI/E0qQqgM7sqKzBV8H+Z/+fivnnyVon+NNf/5xn8z6f5PaNJF1Wa/hE6LLQRtNaehWFGLKAzyAfFtU0BVffX5Gfmrb/nf5B8q+e7Swewi12yju1tZJBI8JYlWQuoUNQg0NBUb0HTFXpv1639f6r9Yj9fjz9LmvqcfHjWtPemKorFXYqowXEV0gkhkWRDWjIwZTQ0O4264qtW7geV7dZkMqKGaMMOaq3Qla1APY4q62u4L1BLbzJMhJHKNgy1BoRUEjY4qp3eoWunhDdXMUAc0UyyKlT4DkRXFUWCCKjcHFVskixKzuwVVBLMxoAB1JJxVbDNHcIksUiyI4BVlIZSD3BGxxVVxV2Kvjb/nIT/nMrRvyU8x6R5C0PyxqfnbzvqkX1iDQNHT95HbnlSa4mZWWJTwPZmCjmwVKMVWIflZ/znRHrnnXS/y3/NH8utW/LbzPrCctJh1KVbux1B6kGKC8jjjUvUUWqBWb4OXMqrKvvjFXYq8n/PT81YPyQ8gea/Plzp76hHoNlJeG0jkETTFSFCByrBakip4mg7HpirJ/y784w/mJ5U8sea7a3e3h13S7DVI4ZCGeJL23S4VGI2JUPQkYqzHFXYq7FXYq7FXzt/zjv/zkLZf85CQeeZ7PR5tNHlbzRqnlpxLMsv1hrARn6wvFV4hxJ9g140+0cVfROKuxV2KuxV2Kvkz/AJyh/wCcrLX/AJxquPIunf4S1HzLqfnHUJNN02z0+SGN2nQwqFJlIFXaZFUdOpJFN1XvP5a+a9U87+W9M1vWfLF55ZvroSmbSb+SGS5tuErxrzaBnQ81UOKGvFgGAaoCrzj/AJys83eYPIH5P/mN5l8q3v1LWNI0a8vbO59GKb0nt09Qt6cyOjUUHZlI9sVTH/nGfzRrnnf8pvy28x+Zbz65q+seX9J1C8uPSji9WW7to5i/CJURa860VQPADFXtzusalmIVQCSSaAAdziqja3cF9Gs1tMk0bdHjcOp+RUkYqiMVfLH/ADitY/nZY6X5t/5XbfWt1fSa5dPo5thagR6WUQRr/okaCnIMV51lofjPTFX1Piqgt1C8rwLMhlQBmjDAuqnoSvUA4qr4qhY762lmktkuI2mjALxh1LqD0JUGor74qisVfB1j+a3n2X/nLnUPy3fWa+UIvIq68mnfVbYUuDeRWvq+v6Xrk8mIoZOH+Tir7xxV2KuxV2KuxV8Hf8/CPzU8/wD5R+RPKWqfl1rP6L1G983aNpdzN9VtrrnZ3kdyrR8bmKVRylEXxABtqBhU4q+8cVfH3/OYv/OSmrf8476H5Xh8q+X4td81eb9attB0OyuZTFbG6uNleZlKkqGKLxDJUuPjUAnFUD+R17/zlG3moJ+bWn+Rx5bltZmaTQJL5byC5+ExKouGdXU7hgf9YPtxZV9n4qom4iEqwGVRKylgnIcioNCQOtAT1xVdNNHbo0krqiKKszEBQPcnFVO1u4L6MTW0yTRno8bh1NPAqSMVRGKuxV2Kvg7/AJxB/NTz/wDmH57/AOciNL846z9f07yz5um0vQofqttB9Vs1kuWWPlBEjSfuvR+KQu21eW5xV944q7FXYq7FXYq+DfzR/NTz/oX/ADlJ+TXkTStZ9Hyjr2i63d6rYfVbZ/Xmsre7dX9Z4mmSj+hskijbcEEgqvsbz2vmV/L2sr5NfT08wG1lGmNqomNgLrj+7+seh+89Ov2uFT7Hpir5I/5wh/Pfz9+dVj+ZNn+Y0WkLrHk/zVfeXXfRYZ47WQ2aosjL68jswMnLixCErSqg4q+48VflZ/zlL+eH/OTn/OOt4/miRvIdz5Lu/MdrpmnwxwanJqotbyZhD64d44vUEa/vGRyvPdVpsFX6p4qh7q7gso2muJkhjX7TyMEUfMsQMVVkdZFDKwZWAIINQQe4OKvnj81P+chrL8rfzC/Kf8v7jR5rubz/AHGrwQ3aTKiWf6NhilJdCpMnqGVVABXjud+hVfROKvnb/nI3/nIay/5x10/ybf3ujzamvmTzPpflpFhmWIwNfrM3rsWVuQQRH4BQsSByXrir6JxV2KoW1vra+DtbXEcwRijGN1cKw6g8SaEeGKorFVGe4itUMk0qxoCAWdgoqTQbnxO2Ksa89+a4vInlrzF5lngaeLRtOvdRkhQgPIlpA87IpOwLBKAnFWG/kJ+bUH57fl95V8/W2nPp0eu2guhaSSiVoTzaNkMgVQ1CpoeIqOw6Yq9dxV8HWP5refZf+cudQ/Ld9Zr5Qi8irryad9VthS4N5Fa+r6/peuTyYihk4f5OKvvHFXYq7FXYq7FXwd/z8I/NTz/+UfkTylqn5daz+i9RvfN2jaXczfVba652d5Hcq0fG5ilUcpRF8QAbagYVOKvvHFXYq/Oj8mPz0/On/oYTWfye/NAeVpbVPLFx5js5vL8F6rCM30NtArvdyVrxZ+a8Dvxo53xV+i+KuxV2KqMlxFE0aPKqtISEVmALECpCg9TTfbFVbFUNbXtveKz288cqqxUsjhgGGxBIJ39sVfBf/Oav57/mn/zj9qH5dax5VTy/N5X1nXdM0DUo7+C6k1L61eySOGh9OSOIRCGJhUnmJCPhZT8Kr72W6heV4FmQyoAzRhgXUHoSvUA9sVV8VfB35Lfmt598zf8AOSf58+Rta1n6z5Z8sWnl+TS7L6rbxi2fUbSK4JE0cSyvyq9ebt7Upir7xxV2KuxV2KuxV8G/85ifmp5//Lrzn/zj1pvk3WfqFh5m85Wuk65F9Vtp/rVnJLblo+U8UjR/uvW3jKNv9oECir7yxVC3d7b2CCS5njhQsFDSOEBY9BViNziqKBrirwH/AJyetfzQvfy516H8nLqG284s1mLCWb6txCfWYvrFPravCG9HnTmPl8VDir1TyLHrkXlvy8nmeSOXW106yXVHhCiN74QoLhk4gLxMvIigAp0AxVlWKqFtdQ3iepBMkqVI5IwZag0IqKjbFVZmCgkmgG5J6AYqh7W8gvoxLbTpNGSQHjcOtR13UkYqicVUIrqGd5Y45kd4iBIqsCyEioDAbjbxxVXxV8Hf84g/mp5//MPz3/zkRpfnHWfr+neWfN02l6FD9VtoPqtmslyyx8oIkaT916PxSF22ry3OKvvHFXwn5b/5zWurz82/Ln5R+Z/yq1zyxqHmH9Ivp13fXFpJDNFYwzzGTjBIx4uIGAoTQkdt8VfdmKuxV2KuxV2Kvkr/AJzw/wDWfPza/wC2Fdf8a4q+D/JH/OSv5ieaPyC8sflx5I/5x383apqFz5L0zQYNT1OzistDmWTTI7T62k8r8ZYGU+olSiupHJ0BqFX1p+S3/OGGlWf/ADjp5W/JP8zozqcdusl1qEVpeTQIlzPezX/ppLCyMVieXj14uQWpQ4q/OH/nBr/nDL/oYLyjr9v50856sv5faL5k1iz0vytp12baOa5RoxNPeSqvKRePBY1qWB5MHQFldV9ifnZ5bs/+fcH/ADjV54i/K3UdTee51FTYT6pcLdtp8+qSw27mHjHGiJFGrNGCp/fENIXLGqr89fJ/kP8A5x/1fynBL56/KX84/MXnLUrZLjUPNDaTqBuTfTIGeWCl76LqjH92ZY5WYAF2bFX1X+Qf/OS35q/lX/zi9+a3mLznp2svqXkuZrTy5f8AmPTri3urq3vWigtHnWYs0pgllBb4mATjGXIXlirI/wAmv+fd2i/m/wCRPL35ifmB+YPmm/8APfmbTbfWRrdtrDxtpz6hClxFHbKARxjVkDAmhIPDgnEKqyP/AJ9VaXq+iWn58adr+pnVNWs/P2pW+oX5NTd3cShJ56mtfVkDP9OKvOP+fjv/ADjH5A8lzeWfzc0jTbiDzVrHnvy9DeXZvriSN0k9Rm4xPIyJvChHEClNqDFX7VMeIJPbfYV/Vir8L/8AnGn8h9O/5+PSebPzl/OfV9S1OwOs3ul6B5dt76S2s9OtYVikqREeYekirRWXkQ0knqF14qvf/wAgf+cANW/5xu/P2Pzf5a16S88hroV7Z21rfXbyXtjLcSI5tVXjxeHmpkVqr1oylhzdV86f85Xf84qW9j+f/wCROnH8z/PE487at5ple4n14SXGjelHbTBNJkMH+jK3q8CCHqiIv7NSq/TH8h/+cVLf8iNavtah/M/zz5oN1ZtZmz8za8NQs4w0kcnqpEII6SjhxD1NFZxT4sVflD/zkx+alj+dn/ORvmzyT+ZGledNd/L7yRFbQweXPKFlNOt5ftHFI0+o+lNEwUs8gjZSG4ogRkrIZFWJ6Vqdh+UH5n/l/wCYf+cafyx/Mvy5Z3Wo29n5o0LVtG1B9KvbCSREMnKa4uH9VVZ6FyVQ8XUpxYOq+wfzuh13/nLr/nJa+/Ia5806joHknyfoUGtaxbaVO1tc6tcT/VmWN5aGsYF1EKcWVeMh/vGUoq+dP+cvP+cQNM/5xlv/AMmLvyR5m1ceVtQ89+X7e58vahftdW8V/G0j291bBqFaReujg1+2u9NsVf0AYq7FX5K/853LBrn/ADkH/wA4neX9fFdAm1m/uTFIx9C4vo5LP0VkQkK3F/TUVrtKy9GIKr61/wCcl/8AnJPyT+T8cvlrzt5Q8w67p+r6ZcSXn6M0KTUbD6m/OGaO4k5Ki1UNyUnZCCaAjFXwX/zkd/zmBpfl3/nF3QtR/IPTtU8t6fq2tQ+U9Lkexktbm1tkgluJZLL435FvT9JJFcnk0nFhKlVVfMF1+Wn/ADjv/hmS2tfyj/OVPOAiMsfm1tE1D69+kh8a3LRC+9Cnq/EQI+XHpJz+PFX6bf8AOH8Hmf8A5yW/5x3Plj889F1UXMklzo15+kUutPvtQs4DFJDO7kxzEkEI0lR6jRsW5EtVV8Bf84s/84af8rS84fnl5Efzjqmi/lf5Y856lZ/oDTLtorjUbiF3hiW5uWVmMEcCKCCWLsa/Ay8yq+lvz9/5wE/Kz8mv+cf/AM3rbRf0xcWsUUvmaztrzVJHhstR0+znihaFIhHyTjK3MS+oX+HkTwTiq+DfOP8AziF5c8ueU/8AnFHWfL2taxYar+aUmjaN5kul1GXlc2mtwWrSxrQ0WOON2iWMDg0fESBiKlV+g3/OHv5daZ/zjx/zkl+df5WeUZbmHysND0LWbexuLl51guHWJXKtISan1WqSalQisW4LRV8q+cNG/Jb/AJyG/MTzz5r8v/lD+Y35xiW+lin1SPUWtdEtpFG8OnSxPC7RoCDGjsfhI4rwKllX1P8A8+pfMmrjSvzb8j3kOq2mmeVPMhh0jTtbYm/0y0uRLS0lr9kx+kKhaJzZ2UDlir5u/K3/AJwg8i+e/wDnJX89/IN1darb+R/L0egXcmhQardLHqF1fWcc8bXUpkMkiwu87LVuYZxRwoZWVfSf/OGXkxv+ceP+ch/zp/JjQNQuZPJ8Gl6Xr+l2VzcPP9TkuBCJFUv0JMzKTWrqkZcswrir9ZsVdir8mv8AnC6BNb/5ye/5yz1rWYlOtWWoaZp9m7nlImmtJdLxQncIyW1qSOgomKqn/P3+CCx/Kjyj5gtj6Wu6V5u0x9HnTadZ3huXZIyCDv6auQO8antiqC/Oj8u9G8p/85o/84/ebNPtpbfVvNcPmFtXc3MsiStY6ObeEKjsVQKhoQgVWPxEciTiqTf85ifkf5T8ifn1/wA4/wD5j6Na3EHmHzR590u31O4N5PIksUQgjCrG7lUHEUIUAEEim+KvVP8An5h+RkXnP8tfOHnx/OvmbTn0HQmRdF0/VfR0W94z8+V3aGNvUY+pQkOtVVB+zirG/wDnGr/nBy0uvJf5Webx+dH5lRNLo/l7Vf0ZD5nVNNQtawXH1ZIBbVFsK8BHy2j+Hl3xVj//AD+B/LnSNS/LTQ/PKxSJ5h0XV7Gy067W6kjWGO8ctJVOYjryjQhyvJabMBiqY67/AM+vNOk8uz68v5oeam/MyO2e6Hmd9XkVW1ED1fsUDpbmQbASCRVoeZIpir5y/MjzYn/OXH/ODU35g+eonu/M3lCRoobxJZIVlu4ry3tDcPHGVjdpLeUc6qVEnJkC9Aq+wv8AnJn8jvKX5yf84xWOv+arSe7v/KPkK61jSZEvJ4ljvV0VZRJIkbhZfiiU0cN3A2Zqqs1/Iv8AK3/ld3/OL35UeXbjzZr/AJd9TQtGl/SHl7UfqN+BBEAI/WKSfu2GzKV3oPDFXwH/AM4S/wDOHNr+adp+bMkn5tef9D/QvnvXNIC6J5hWzW7W2WAi6ugbd/UuZOXxybcqD4RTFX7gflt5HX8tvLOj+WU1rU9aXTYfRGoazd/XNQuByZuU83FObCtAeIooA7Yq/Nr/AJza/wCcYP8AnIf84vzN8p+Yvyu/MI6LodnaW0Jh/S11Yiwu45pnluTBApWYOjIK/E7U9NlEYBxV+rMSuiIsj83CgM1KcjTc07V8MVVMVdir8rf+fgn/AJNr/nEH/wADpP8AqK03FX6pYql+raTY69ZXmmanZw3tleQyW9zbXMSzQTwyqUkjkjcFXR1JDKwIINCKYqqadp1po9pa2FhaxWtraxRwW9vBGscUMUahEjjRAFVVUAKoAAAoMVfgD/zkx+alj+dn/ORvmzyT+ZGledNd/L7yRFbQweXPKFlNOt5ftHFI0+o+lNEwUs8gjZSG4ogRkrIZFWJ6Vqdh+UH5n/l/5h/5xp/LH8y/Llndajb2fmjQtW0bUH0q9sJJEQycpri4f1VVnoXJVDxdSnFg6r9zfz2/5x+8lf8AOSHl+Dyx570+a902C8ivo44bye1YXESSRqxaB0LDjIw4tVd605AEKviT/n1Zodr5W8nfm9oWnqyWOlfmT5hsrSN3ZykFvb2UaLyYkmgAqT1O5xVgv/P2/wDKvTtd8t/l35vspZ7LzPF5o0nQrHUYrmZPq0F6LmUsI0cLzWWNGDgBxSgamKvGf+cx/wDnEPyx/wA4b+SvLv5zflvqurwecPLuu6bLqGq3mpz3E+rJcMyzGcMeHKSTiXCqEeMyI6uG2VfR3/P0r88fMv5d6J+XXkby1f6npreetVmtb+90aF5tTWwtWtlmhtFR0JmlNyvFQytJxMfIBmxV+f8A548g/kXp/lWeT8sPyf8Azi8u+e9Pha40jzENJ1L6w9/EpMZuP9MaNFlbZ3hiRkB5INuJVfuH/wA4geePOP5jflB5J138wNMutO8yTWs0Oow3lq9pO8lrcS2yzvDIqspnSNZfsgHnVRxIxV7uPLGjrrDeYRpNoNYa0Fg2oi2j+uGzWQzC3M/H1PSEhLhOXHkeVK74qnmKuxV2KuxV2KpHr/lnR/NUENrrek2mpQQXEF3FFeW0dwkdxbuJIplWVWAkjYBkYDkp3BBxVPMVfhl/z8I/5xY8qWn5g/lFr8era8bnz1+YOm6fqKPq0jxW0V5LGJGslcE27j9gqSE2CigACr9J/wAhP+cRfKH/ADjtqupav5d1rzDfT39sLSRNX1mW9hWMSLJVIiqqHqoHIgkCoFAzVVfnL/z8B/5x2tdA/Nr8nfN/5aXL6D5786+aTay6xNcS3EUUyxQJHKIJjKihASeKIAelMVfUHkr/AJ9jflz5U8xeVfO935q81ap5t0XULfVJ9ZudXrLqNzEwcpOrRuRAzVqisHKEo8rgmqryP/nPfUvyp/Mv8w/KvkLXNC87efPMmk2kl03lLynMEslhuSjLPqDceSMAFo0bAqjKZCqsnJV89/8AOMYb8l/+cqfJHljyr+XHmb8ttC84aPqsWo+X9bvXu7a4msrS5u0ubeR5JOZUwIpJYuhLqCFkK4q+mP8AnJ78u9D8r/8AOW3/ADjT5x02CSHV/M95rsGqTfWJWWaPTbG1it1WNnKRhUmcEIFDVq1TvirHv+c7/wAjvKflL82PyL/NTS7W4h8z6/8AmR5UsL+6N5O6PbwmNFVImcpHtCgPACu9ftGqr9icVSPSfLOj6BPqV1pmk2llPqdx9bvpba2jhe7uOCx+tOyKDJJxVV5MS1ABWgxVPMVdirsVdirsVSO68s6Pe6pY65caTaTapYxTw2l9JbRvdW8Vxx9VIpipdFk4rzCkBqCtaDFU8xV/Lr5l/OHULO8/Pr8nTbpouged/wA2dXt9b8538FxJp2kQSXxMcdYQq+s5t2f95Iq+mpqOJaSJV+7n/OLP/OIH5e/84p6ZeR+SRd3Fzq0NoL/ULu8eZrz0A5jcRqRCg/eMR6aDYgFmoMVeCf8AP0//AMlJ5e/8DLy7/wATlxV+lOKv5lvJP5m+S/8AnKbzN5w/MD89/JX5heebb9J3Fp5f0fy9p17Joul2SBWCl7W6gdbijKGUEdPUk5tIOCr6T/5w413Vfy7/AD/tPK35X+UPPmm/lP5jsrp7vTvNOlXcVvo2oxQzziS3kkeVVR2iROTyBnM3B+bJG2KqH/OV3/OKlvY/n/8AkTpx/M/zxOPO2reaZXuJ9eElxo3pR20wTSZDB/oyt6vAgh6oiL+zUqv0x/If/nFS3/IjWr7WofzP88+aDdWbWZs/M2vDULOMNJHJ6qRCCOko4cQ9TRWcU+LFX54/8/Nf+ceIYb3yP5xPn/zbI3mbz95f0ttLk1gNpemLPb3Cm40+3MX7idPSqj8m4l32+LZV91/lB/zhha/lD5p0/wA0xfm/+YmvtZrcKNO13zILzTpvWieH99ALePnw58k+LZ1Vv2cVfA//AD8H/Nu583/nd5W/JvWh5pl8iWmkpq2vab5QtHuNS1aSZpeEUiq6crZAkYPUKXc0LiNo1XzP+Y0PlH8tpdC83f8AOL/5Ufml5N84aZdwmSK60XU5tN1KzrWWG7E13cuQSF+FfgYclZKlXRV+qv8AznX+TXlX88vyP1fzh5v0W5TV/LHljV9a0uI3NxbmyvZLFZmWWJHVZCrxICJFalCBTk1VXyr/AM43/wDPvnT/APnIb8qfJPmX85/Omta413odgug6bY3v1Wx0bTRAq2ojj9Mh7j06GRyvEsxVlkI9RlX1D/zkf/ziTpa/knHoen/mD5y0yy8geVtaS2jstbEK6pHFZ81TVFEHGdAIeIVRGFR3RQFIAVfMv/OHX/OEFp+Y/wCTv5f+aH/Ob8yNGbULJpzp+jeZltLC3InlXjBF9Wcovw1pyO5JxV+1EEXoRxx82fgqryc8magpUnuT3OKpQPLGjrrDeYRpNoNYa0Fg2oi2j+uGzWQzC3M/H1PSEhLhOXHkeVK74qnmKuxV2KuxV2KpHr/lnR/NUENrrek2mpQQXEF3FFeW0dwkdxbuJIplWVWAkjYBkYDkp3BBxVOJpVgjeV68UUs1ASaAVOw3OKvw/wD+cbfyOX/n41aeZPzi/OHzJrE9jcavd2Xl/wAvafqL2tlplrbcGRqIP71S4AIpXiZHLlxwVTb/AJxk/KLUPyO/5zN8x+UbrzVf+YrG0/L+Z9IudUuDcXkGnS6hZultK5pX0pDIFoAOBUhVB4hV6N/zi/8Al1on5Uf85ffnv5f8twS22nyaBpV+0UlxLOTcXj29xKxaVmY1eVyKk8QaDbFWPflf+R/lL/nHv/nNa10HyTaT2djqv5e3mq3cc15PdNJd3GqSo7F53dqEQpsT1FepOKoT/n7H+RPlt/Lemfmtp1rLF51TWNC0q1vjdzekkRklKD0GZogQ9DyCV8a4q9Ji/wCfWvlPztBb65+aHnzzR5h88TTQ3t5rtvqYtvSuFYSGKyiaF1hgBACjjyUAGP0tlVViX/OT1lrX/OU3/OROhf8AOOs/mO+0PyVpnl/9P67FYzGG41ZmfisLO1Q6LyiADBlBMrkM6pwVYP8A85F/8+kNJOjW0/5DalcaFqYnthe2F/qlw9lewo6sspkbm6ywuBIoPJTvxCsFqq9l/wCftWmXWtflh5A0+xvWsrq78/6BBBdIWDwSy21+iSqVKsCjEMCCDtsRir5r/wCcv/8AnEXyx/zhv5U8qfnH+WuqavB5u0HzBph1HVLzU57ibVkuWZZjcBjw5SSceYVVR0aRHRuQoq/d/FUjsvK+jabqepa1aaTaW+p6mtul9exW0aXN2tqrLAs8yqHkEQZggYngCQtKnFU8xV2KuxV2KuxVI9X8s6P5gm0241TSbS9m0y4F3YyXNtHM9pchWQTQNIpMcgVmXmtGoSK0JxVCedvMi+TfLuva+8RmXStPvL5o1qS4toXmKinc8aYq/m0/J7zF+Xv582+pfmL/AM5CeQPzI/MbX9Zurr6qdI0y+bQtPtEkZFhsXtLyEkKwYFa8EoF481Z3VfZ3/PvTzD5o8q/mv5y8h6BoHnS3/KifTm1DRE826XcW8ulXkbwBreKV2dBE/qSgKGq/BH4hhIzqs8/5+if84wfl95g/Lnz3+b95pk7ebNLsNNhtrsX1wIlQXsENDb8/S+xKw+z3r13xV+kv5RO0nkXyW7sWZtE0okk1JJtYqknFX42fmt/ziX5Y89/85nP5Tt9R1TSNH8yeUp/M3mWKy1G4STVJJtQuEmtmkZ2KQSyJCzRj4AqERhG4sqr078rfyh07/nEP/nLfQPI35fXN1aeUPO3lS8vbnSZryW4ijvLJpj6i+qWJIEK8WZi49SRa8CBirw//AJy1/Ne3/N3/AJyO138tvP2necNX/LzyZZWbvoHlCzluJNRv54be49W+EUsbCIeuVVwagRqI+Bkd8VeZXmoaT+T3njyT5t/5xh/Kz8zfLU41CC38waHqOi6lJpep6cx+LkZrm5cyCpADHiOQkQxulWVfVH/Pyn8jdO80fmf+Qer6RqF7onmLzP5htvL0+r2d5PHNb2aslHgUPxjlQSuQyheRoGNMVY3+eH/OL/lf/nCbz5+QXnv8qLjUdMn1TzhpvlvW45tQmuf0nbai4Mhl9UkVZI5AwACcmV1QMgOKv3BxVI9J8s6PoE+pXWmaTaWU+p3H1u+ltraOF7u44LH607IoMknFVXkxLUAFaDFU8xV+WH5/f+tsf841f9sbzJ/1A6hir9T8VdirsVdirsVfGX/Oaf5WfnB+dXlKTyX+WeoeWLPTdZtb2015tfN6lwY5PS9H6m9pFMqnaTnzT+Tj3xVA/wDOJX5ff85AflfaaT5V/MzU/Jd15W0LQrTStLGgDUW1H1bJYIIGne6ihiKeij8+K1LlSAoqMVe5/npb/mtc6BbJ+T915ct9c+uRmdvMy3bWhs/Tk5iP6mrOJefp0qpXjz6GhxV+c/5Ff846f85if84+aJe+X/LXmD8sJLO91S81aY3ra1NJ694VMoUpZxDh8IoOv+Vir9G/z0/JbSf+chPy+17yB5mdo7fV7aNGnt6hre5idZoZ4wTv6cqKwUmjAcW2JxV8MeWPy7/5zb/LLRLXyRofmn8vtc02whFpp+u6vHqSakltEOEPrJGhjLqgA+JJzt8cjncqvobyV/zjZ5r138tfOfkf86vzFn873PmxZluZ47SKyh09JI1VUs1RR/dyKJFJVV5AfuwOXJV8peQ/+cef+cyfyj0ZPy18rfmb5Sm8q2itbabrd/a3D6tYWNeKpHD6DpzRSTGjvKi7IJQgUKq9E/5wX/5xQ/Nr/nFPWvOuma/5k8v615T1m5u9QhuImvn1ua+Lxxwy3HqxJEivCrNKoklb1SOLsKsVXl358f8AOOX/ADmF/wA5B6XpmieYtf8Ayxjs9M1i01q2Nk2tQyfWLP1BEH9S0m+Ckh5L16fF4qvtb8hrL/nIy11m/b849Q8kXOkG0ItF8spqQuxeeolDIbyONPS9PnUCrcuNKAGqr5XH/OKH54/843eavNOr/wDOOnmfy9J5Z8y3cmo3XljzPHciGzvZCAz2slupYim28kXwBEZZeCsFXtv/ADj1+SH516d5yvfzE/Oj8z11S9ktJbKy8taC00Og2ccrqxkZJFj9WQBQFLR8l3rLJ8PFVHf85f8A/ONfmn86rr8u/OP5feYbPR/OPkHUZ77Sm1OOR9PuEuvRE8Nx6Su6hvQT4gjfDzWg581VYZ5C/Lb/AJyy8z+btD1r8x/zO8uaFoOm3cNzPonlXT3n/SMcbAtBNNewrJGkgHElZHNCSFDcWCqD/OX/AJxV/MnRfzUu/wA6/wAhvNOl6X5g1e0hsvMGja+k76VqiQIiJKWgDukgWOMUVU3XkJU5yB1VHRPy0/5y6/MbX9HvfPn5leWvJuh2Nxb3E+neT7Ka4uL4ROGaKWW/UmNXA4krM67nlEaDFVf/AJyX/wCcTPP2ufmNpP52fkh5rstA86Wtmum6jbaornT9Us1OyyFI5SG4gKVMZVuMbK8Txhyq+bPzj/5w2/5yt/5yEl8qeaPN35geT49Y8tapbXumaFbrfQ6PCI/3rzyypbSSyXBkjiTgVZRHz4yoSVZV+wfk1Nfj0HRl81PZSa4LO3GpvpwkWya89NfWNuJv3npc68OfxcaV3xVkuKvlT/nLf/nFfSP+cqfKtnpNxqUmja3o90NQ0PWbdOU1jdqPAMjGN6LzUMpqqODyRcVfK2r+TP8AnOTVvLN/5EvtU/Le/t72xl0+bzBJ+kUvJIZo2hd2RIliEpU7kW3EGux64q9Mv/8AnBG082f843+WvyN8wa+F1HQkjubLW7SFiLXU4pZpUmjid1LIFneJgWVmRmIKNQqqwuz8r/8AOddlpsXlhfNP5cyLFGsI8yyw6g+osg+ESNEYDbtNx3INvxLdWO7Yq+iNF/Lv87vy+/Ky40bSfzD03zb5/kuvrK6v5ntJbfTlSWVDJAsVjykVEjDemfi+M7hUoqKvjD8qP+ccv+cxvyc1Lz1quheYPywefzlrVxr2pC6bWpEW7uCxcQhbNCse+wJY/wCVir7G/wCclfy2/Nz82/ybPkry5f8AliDzJrdlDYeYp78XyacYZrR0vTYGJJZUYzFTEZUcCPlyHKmKvkzX/wDnED8/ta/LX8k9F/TnkpfNX5W67ZXemSepqY0yXTtNtYYrRZ2+qtLJP6kX7zjHEjIfhZWG6r3ax/5x0/M3TP8AnIjUfzTg1Ty6/lnzJoFlo/mC1c3o1EG3tuLfUlWP0wDOiEM81fTLDhyAJVfOv5Vf84vf85X/APOONhqX5dflp548nv5Pe8ubjTtT1e3uH1OyjuGBakKQNEZK1bi3qxlqnkoPEKvS/wDnEr/nFD86f+cafzE82Xuo+c/L/mXyn5pu21DWL25jvU8wXV4IJSsqxhPq8XK4lJcGaWsY2ox2VV/yA/5x5/5yD8hfnj5w/MzzjrHkifSvOX1dNbg0o6mboR6faPb2X1SO4gVIzyEZl5yyVXlx3ocVd+Xv/OPP/OQnl/8A5yH1383NY1fyQ+ja7Gul31taNqZuxpFswNv6McluqLc0jj5lpmSvKgIpir9MsVdir88vzw/5xT8+2X5mj87vyI8xaZo3mu7tEsNc0vWopDpOsW6BeLyNbo0iygJGDQDlwQh4yH9RVh2k/wDOLH5x/n/558p+cf8AnI3XfL66R5TnW+0nyr5ZS5NnLfqwYTXj3dWIBVfh5yhgOP7tTIJFXq3/ADmJ/wA41+dfzd1X8ufP35YeYbHR/OXkS8up7H9Jq5srqC8ESzRStHHKw2joPgIZXdSVJDqq+VfzS/5xO/5yy/ObX/IPn/X/ADz5Gi1zydqcV9pmhwxajFo8JjaOYzySiGWaWWR40V0I4hBWOVSWBVfpn+Z/5YP+c35ba75F8yXaW8+u6Q9jeXNkjGOK5kiAaWFJTUokvxKrGpUAE13xV+fvlX8kf+c0vKnlvQvy50r8xfI+laFotpDpttrlvaXU+rfUYEEUK+lcWzQl0jVVFAmw/vGb4sVUP+flPlPVNI/5xm8veXNf8xS65qMWs+XbO81eWBIZLuasiPOYkJVSxNeNT7sxqSqmGu/841/85cX2gyflxb/nfoEnlWSBtPbWptOnTzEbA/u/TbhEUL+l8JcTrIf9+ciWxV9JQf8AOGnlnSP+cf8AUPyE0m9lhsbrTZ7Y6jKgkla+lk+sG7kSoB/0ij8AwogEasKA4q+P7j/nF/8A5y782/lrP+TmufmF5MsfLltpLaXFqFnFfTanf28EXC2tbh3hRUhcKkc0qr6vCu01WDKvuH/nDv8ALT8xPyd/LbSfJf5jXuh3d1onCy02TQjdNF+jooo1iE7XUcTNPz58iqKvHj3rir5TT/nGb/nIv8iPNP5iXP5G+afKU3l7zrrNxrklv5liuxdabe3RLS+ibaJ0dNwoLcvhVKx8gzMq+xP+ccfy9/NDyJperP8Amt+YyecNX1K5S4QW9hFZ2enoqcTDBwRC6k7lmROgogPJmVfF3/Obf/ODf5p/85G/mb5S85eTPzEg0XTtMtbaAw3FzdxS6fPFLM73VmlvGyO7rIAeTRtUcS/CnFV+rUSNGiKzl2VQCxABYgdSAAN/YYqqYq7FX5W/8/BP/Jtf84g/+B0n/UVpuKv1SxV2KuxV+dP5y/8AOKv5k6L+al3+df5DeadL0vzBq9pDZeYNG19J30rVEgRESUtAHdJAscYoqpuvISpzkDqqOiflp/zl1+Y2v6Pe+fPzK8teTdDsbi3uJ9O8n2U1xcXwicM0Ust+pMauBxJWZ13PKI0GKvoD8/bP/nIC6vNJ/wCVM33ky2sxDL+kB5oTUTMZuQ9P0TZpIvDjWvIA18R0VfC/5Kf844/85i/kNb+aLXy3r/5YSJ5i12+8w3v1x9alIvL1Y1lEfCzj4x0jXipqRv8AEcVe5f8AObv/ADj3+dP/ADkVp3kjRvJuq+U7Oy0e9stbvjqjahFM+rWXMR+i0EFwptiJGqrKr1p8WKpb/wA5jf8AOPH56/8AOS/5Y+VvItjqvk22uZo4bjzTLO2pQxPfWxieL9HMkE5EPIScxKganChG+Ks2/O7/AJxb83f85J/lt5Pt/N/mLTtE/MvyzeJq9hrehRXDafbajFIWURpOVl9NlWPkxoyyIJFUgemyrzeTy3/znVr1nHoF15q/LnRUASObX7C3v5791pRpI4Z4Tblz1p6MQr044q+7vyi8n655B8oaLoPmXzZc+a9Ws45Bd6zdwpBLdySSvJUxoSFVAwRByY8FXkzGpxV6RirsVfg3q/lO+/5y6/5zn1SBp5n8rflqtj9b4SOImbTgsgtzQ8eUl/JIGXq0Ucnhir95MVdirsVdirsVflP/AM5Q/wDOOH/OTn55ecdB1LStd8gQaH5R8yQeYPLcdy2qxXnO1ZXhW9CWsyP9n4wjAHsV6BV9dfkDZf8AOQVpe6sfzm1DyZdWTQxfo8eWE1ATLNyPP1jeRxrw40pxBNfAdVXyB/zkF/zjv/zlR+b3nvy55jsdc/LuHTvJmv3OreWo5m1eOf0y4EK3yrazK7+mi8xG4HLlxNKUVfZn5AWn572r61/yui+8oXKMLb9Gf4XS/DKw9T1/rBvUQUPwcOIP7VT0xV8tfm//AM4y/nV5U/OrW/zp/IvX9A+s+Z9PtLDW9L8xrP6JNrHBDHJEYEYleEEZNGRlYNTmHoqryfWf+cOv+cpL78yPJf51S/mT5Q1fzjo6ahAbC/t7+DRNPtriCS2WG0W2gMkwKTzM7OIX5cKtJSuKvqz/AJzH/wCcavOH53D8u/Nn5feYLLR/OvkPUZNQ0yW/VzYzC4EP1iKQpHK6hjChHwMGUMjAcuSqvk783f8AnFD/AJyz/Pi+8jebfMnnfyNbat5P1i11PS9Dto9Rj0kSQMs31meb0ZZpJjJEienxKCMsUljJZWVfrx5aXV00jSl197V9WFpbjUGsldbVrsRr65gEpLiIycuAY8uNOW+Kp3iqjcXEdrFJPM4jjjVnd2NFVVFSSewAxV+E3/PtbyXqX5+/nH+aH/ORetS3Daemqanb6Ikrvx+s37Mz8QT0trN0iAI/3aO6Yq/d/FXYq7FXYqxbzwvmN/L+sL5QewTXjazfo1tUWVrEXXE+n9YEH7z0+VOXD4qdAemKvgb8if8AnDjzPof5d/nR5b/NceW9c1T8w9b1XW5ILJrt9MS4vYlZCWlhhnj4XALKYwXjAVkkL9FUy/5wn/JX/nIr8hbWx8n/AJi+avK+t+T9OtZo7EWcuoXGrQPVPRhWWe2t0+rxjnQNzZQQi0QKFVeRf85Kf846/wDOWn/ORFnN5dv9d/LaHQrbWotU0/0m1iG7AtJHNsJibWZa8WHMCu/RqYq+w/yGsv8AnIy11m/b849Q8kXOkG0ItF8spqQuxeeolDIbyONPS9PnUCrcuPQA1VfLtr/zi3+ev/ONPmjzZqP/ADj15h8s3flbzNfS6pceW/NSXapY3s1A7W0lpuykClTJH8CojLIUD4q9n/Jj8rv+cjrjzpaecfzd/NHSl061S5VPKflixI06UzRNGrT3NzGkx9MtzAPM8lWkgXkrKo7/AJy//wCca/NP51XX5d+cfy+8w2ej+cfIOoz32lNqccj6fcJdeiJ4bj0ld1DegnxBG+HmtBz5qqwzyF+W3/OWXmfzdoetfmP+Z3lzQtB027huZ9E8q6e8/wCkY42BaCaa9hWSNJAOJKyOaEkKG4sFXrP/ADmV/wA433f/ADk55Ct/L+ka4ui63pGq2eu6PeyIXhS/s1lRBKFBPBlmcVAYq3F+L8eJVeARfll/zmh5+vrO18z/AJp+UfJ+lW8kRuJvK2nzXl7dCMgmg1CAKvOnUOgr1jK1XFWf/wDOT3/OKnmvz7518r/m/wDlL5qtvLnn7y9bNYA6gkkmm6nYMzt9XuRGshUD1ZPiEbluQ+yyRuirzy9/L/8A5zU/NN4NO1/8wPJfkLTAaXF35Ws7y81GYAj7IvgyqD1qk0TDwIJGKvWP+cr/AMsPz0/Mvyy3kn8tta8qro2raLd6Trtx5kF8moTNcR+gZIHsoZIlLIWJrGKP0BGwVee/846/lh/zld+VEXkXylr2s/l1P5O0GGz0+4FmNXk1R7C2jEf7tpLeKIzcQN24rXcjtir7+82eW7Xzloes6BfFha6rZXVjOUNG9K5iaF+JNd+LGmKvyv8AIX/OO3/OYH5K+WbP8sPIfn/yQnlnTpbpdO1q8tLttXgtZp3l4tC9vNbl1LllUiQD7PqcQKKv0h/Jryh5m8ieUNI0Xzj5vl82a3biY3mry20dqbh5ZXkAEUdQqxqwRdySFqaVoFXp+KuxV+DX5o+U7/8A5y9/5zji8siedvLH5e2mntqhjldYWjtAt48VBtzmurhYH6MY1dgf3eKv3lxV2KuxV2KuxV+SOkf84j/85Bf84yeZPMq/849eb/Lj+TfMF7JfjQ/MqXAGmXE2x9EwRSFlRQqhg6syBFkidkDlVQ/KX/nDb/nIf8uPzztvzh1X8wPLHmKbWrWCy8zyXkd7FMtrJNC9xb6dBDAIgIo4I0t3eSMdS8Q7qvVvzv8A+ca/zh0r85W/Ov8AI7X9Bh1HU9Hi0fWtK8xrcfVbhIWUpIrWyMx2jj2rGyFNnZZGQKvM/J//ADif/wA5K6N+eOhfnTrXnryfq95dWUGm67bvFfwxWentcAz2mmRpDRwsShopZXiJld/UQ/bdVmX/ADmh/wA49f8AORf/ADkZc3Pl3y5rPka18mQahpmpact+2pxasJ7OMMRO0VtcQlTMXpxpVONaNWqr6F/Iyx/5yUttfuG/N7UfItxoRtJBEnltNTF6LvnHwLG7jjj9Lhz5dWrxp3xVgP8AzlH/AM4neZvzE83+WPzb/KfzXB5Y/MHy9A1mkt7G0mn6lYsXP1e6CLIVA9ST4hG/INQgFY3jVeW3P5I/85a/nhfadYfmV+ZuieS/LdrNHLdxeRWvIdSv/TYHj9YmCvEGp1WXjvVoXoAFWd/851/849/nD/zkSnk3SvIGqeVrPSNE1O012X9NvfR3banZGVYOLW0E6NBwlPJSqvyH2qbBVJP+cxP+cevz3/5yU/Lfyh5K0/VfJttcslvdeaJbh9ShjfUbYxPF+jmS3nIgL+pyEqB6cKHrRV92fltH5vi8s6Onn2XTJfMghI1F9GWddPabk1PQFz+848eNeQHxVoAKDFWcYq7FX4Nf842+U7//AJyo/wCcxfzN/NK8nnk8teR9Slhs/wB6/oT3dsDp9iqD7PEJA9y1Ds/p8h+8OKv3lxV2KuxV2KoW+soNStri0uohLBcRvFLG32XjdSrKfYgkYq/K3yX/AM41f85Jf84pPqvl38kPM3lPXfJF1eTXen6b5tW+W60s3DFnjje0I5oNiWMtGar+irM5ZV9Jf84+fld+fGkeZL/zb+cX5pWWqi4tZLa38taHYiLSbVmkRxKJpY45pHUKQOScviNZGAAxV4J/zk5+SH/OV3596X528jwa3+XMHk/WrhktfU/S8OppYxXSXFuJWW2njE1I0EhXkpPLjQEUVev/APOPvlP/AJyh8o6p5d0r8w9V/L+48o6dZfVJk0VdVOqMILYxW5RriGKKvNUMhPVeXFakUVeZ3/8Azjz/AM5B3H/OScH5yRax5IGiQwDy+tux1MXp8tm+NywMXoGP69wYgP64i50PELtirvzN/wCcef8AnITzL/zkJoP5taDq/kiLR/L8T6Zp1veNqf1s6Xcg/WfXjjt2Rrj95JwKzKn2Kgb4qyT89f8AnFT8wYvzSX88PyL806bo3mq7sotO1zTNbilfS9Xt4gio0jwK8iOqRRrQKK+mhWSIh+aqUWf5af8AOYX5n6rpsnnT8yvK/kbRraaGa4tvJ9lPc3d2I3DNGz6gremGApVZmXf4omAoVVT/AJzA/wCcevz1/Ofz5+X2v+QdW8nWmk+Sry31jT01ltRS7bU1b94Jhb280bwcUTiFaNvtVPQ4qt/5zI/5x6/Pf8/Lr8th5S1byba2nla70vzBMNTbUopX1+yaWvD0YJ1azKsvFDwkryq3Sir9A/Ka62miaQvmV7N9aFpbDUm09ZFs2vPTX1zbiYmQRGTlwDnlxpy3xVkGKuxV+WH5/f8ArbH/ADjV/wBsbzJ/1A6hir9T8VdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVfMP/OQ//OIf5d/85QyaPJ58tb+4/RSzLbJa6jPbR0lKli0aNwZvh2anKm1aYqmv5Af84u+Rv+cao9bj8lx6gg1hrZrr69qVxeAm2Egj4CZiqf3rVKgFtqk8RRV9E4q7FXYq7FXYq7FXYq7FXYq8I/PP/nGj8uP+ckbXSbP8w/L36Xi0qWWaypeXdo0LzKquQ1pNExDBVqCSNgcVZ3+WX5Z+XPye8taX5Q8pWDWOj6Ysq2tu1xPcmMSyvM/725kkkarux+JjStBRQAFWeYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYqw3y3+XflfybqOvatoXl+x02+16dbrVbm0to4Zb2deVJJ2QAu9XY1O9WY9WJKrMsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVUp4I7mOSGaNZI5FZHR1DKysKEEHYgjYg4qxfyT5C8uflrpUWheVNDs9G0yJ5ZI7Oxt0t4EeZzI5CRgAcmJPT8MVZbirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVYboH5d+V/Kur67r+jeX7Gw1TXXhk1S8traOKe9eEMI2ndAC5Xm1Canc+OKsyxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KsN8mfl35X/LmHULfyt5fsdGh1C7lv7uOwto7dJruUKrzOsYALsFUE+wxVmWKuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV8meQf+cHPyV/LHzpD+YXlzyc1r5jhlu5or19V1K44SXiSRzMIp7qSL4llcbpty+Ghpir6zxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV//Z)
:::

</div>

Figure 7.20 -- Translation output part three

Here, we can see that []{#_idIndexMarker404}although our model clearly
makes a decent attempt at translating English into German, it is far
from perfect and makes several mistakes. It certainly would not be able
to fool a native German speaker! Next, we will discuss a couple of ways
we could improve our sequence-to-sequence translation model.


Next steps {#_idParaDest-129}
==========

::: {#_idContainer208}
While we have shown our sequence-to-sequence model to be effective at
performing language translation, the model we trained from scratch is
not a perfect translator by any means. This is, in part, due to the
relatively small size of our training data. We trained our model on a
set of 30,000 English/German sentences. While this might seem very
large, in order to train a perfect model, we would require a training
set that\'s several orders of magnitude larger.

In theory, we would require several examples of each word in the entire
English and German languages for our model to truly understand its
context and meaning. For context, the 30,000 English sentences in our
training set consisted of just 6,000 unique words. The average
vocabulary of an English speaker is said to be between 20,000 and 30,000
words, which gives us an idea of just how many examples sentences we
would need to train a model that performs perfectly. This is probably
why the most accurate translation tools are owned by companies with
access to vast amounts of language data (such as Google).


Summary {#_idParaDest-130}
=======

::: {#_idContainer208}
In this chapter, we covered how to build sequence-to-sequence models
from scratch. We learned how to code up our encoder and decoder
components individually and how to integrate them into a single model
that is able to translate sentences from one language into another.

Although our sequence-to-sequence model, which consists of an encoder
and a decoder, is useful for sequence translation, it is no longer
state-of-the-art. In the last few years, combining sequence-to-sequence
models with attention models has been done to achieve state-of-the-art
performance.

In the next chapter, we will discuss how attention networks can be used
in the context of sequence-to-sequence learning and show how we can use
both techniques to build a chat bot.
