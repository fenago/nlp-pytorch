<img align="right" src="../logo.png">


*Lab 8*: Building a Chatbot Using Attention-Based Neural Networks
=====================================================================


In this lab, we will look at the following topics:

-   The theory of attention within neural networks
-   Implementing attention within a neural network to construct a
    chatbot


The theory of attention within neural networks
==============================================


In the previous chapter, in our sequence-to-sequence model for sentence
translation (with no attention implemented), we used both encoders and
decoders. The encoder obtained a hidden state from
the input sentence, which was a representation of our sentence. The
decoder then used this hidden state to perform the translation steps. A
basic graphical illustration of this is as follows:


![](./images/B12365_08_1.jpg)


Consider the following example:

I will be traveling to Paris, the capital city of France, on the 2nd of
March. My flight leaves from London Heathrow airport and will take
approximately one hour.

Let\'s say that we are training a model to predict the next word in a
sentence. We can first input the start of the sentence:

The capital city of France is \_\_\_\_\_.

We would expect our model to be able to retrieve the word **Paris**, in
this case. If we were to use our basic sequence-to-sequence model, we
would transform our entire input into a hidden
state, which our model would then try to extract the relevant
information out of. This includes all the extraneous information about
flights. You may notice here that we only need to look at a small part
of our input sentence in order to identify the relevant information
required to complete our sentence:

I will be traveling to **Paris, the capital city of France**, on the 2nd
of March. My flight leaves from London Heathrow airport and will take
approximately one hour.

Therefore, if we can train our model to only use the relevant
information within the input sentence, we can make more accurate and
relevant predictions. We can implement **attention** within our networks
in order to achieve this.

There are two main types of attention mechanisms that we can implement:
local and global attention.



Comparing local and global attention
------------------------------------

The two forms of attention that we can
implement within our networks are
very similar, but with subtle key differences. We
will start by looking at local attention.


In the following diagram, we can see this in practice:

![](./images/B12365_08_2.jpg)


The **global attention** model works in a very
similar way. However, instead of only looking at a few of the hidden
states, we want to look at all of our model\'s hidden states---hence the
name global. We can see a graphical illustration of a global attention
layer here:

![](./images/B12365_08_3.jpg)


We can see in the preceding diagram that although
this appears very similar to our local attention
framework, our model is now looking at all the
hidden states and calculating the global weights across all of them.
This allows our model to look at any given part of
the input sentence that it considers relevant, instead of being limited
to a local area determined by the local attention methodology. Our model
may wish to only look at a small, local area, but this is within the
capabilities of the model. An easy way to think of the global attention
framework is that it is essentially learning a mask that only allows
through hidden states that are relevant to our prediction:


![](./images/B12365_08_4.jpg)


#### Building a chatbot using sequence-to-sequence neural networks with attention

The easiest way to illustrate exactly how to
implement attention within our neural network is
to work through an example. We will now go through the steps required to
build a chatbot from scratch using a sequence-to-sequence model with an
attention framework applied.

As with all of our other NLP models, our first
step is to obtain and process a dataset to use to
train our model.


Acquiring our dataset
---------------------

To train our chatbot, we need a dataset of conversations by which our
model can learn how to respond. Our chatbot will take a line of
human-entered input and respond to it with a generated sentence.
Therefore, an ideal dataset would consist of a number of lines of
dialogue with appropriate responses. The perfect
dataset for a task such as this would be actual chat logs from
conversations between two human users. Unfortunately, this data consists
of private information and is very hard to come by within the public
domain, so for this task, we will be using a dataset of movie scripts.

Movie scripts consist of conversations between two or more characters.
While this data is not naturally in the format we would like it to be
in, we can easily transform it into the format that we need. Take, for
example, a simple conversation between two characters:

-   **Line 1**: Hello Bethan.
-   **Line 2**: Hello Tom, how are you?
-   **Line 3**: I\'m great thanks, what are you doing this evening?
-   **Line 4**: I haven\'t got anything planned.
-   **Line 5**: Would you like to come to dinner with me?

Now, we need to transform this into input and output pairs of call and
response, where the input is a line in the script (the call) and the
expected output is the next line of the script (the response). We can
transform a script of *n* lines into *n-1* pairs of input/output:


![](./images/B12365_08_05.jpg)


We can use these input/output pairs to train our
network, where the input is a proxy for human input and the output is
the response that we would expect from our model.

The first step in building our model is to read this data in and perform
all the necessary preprocessing steps.


Processing our dataset
----------------------

Fortunately, the dataset provided for this example has already been
formatted so that each line represents a single
input/output pair. We can first read the data in and examine some lines:

```
corpus = "movie_corpus"
corpus_name = "movie_corpus"
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
with open(datafile, 'rb') as file:
    lines = file.readlines()
    
for line in lines[:3]:
    print(str(line) + '\n')
```


This prints the following result:


![](./images/B12365_08_06.jpg)


You will first notice that our lines are as expected, as the second half
of the first line becomes the first half of the
next line. We can also note that the call and response halves of each
line are separated by a tab delimiter (`/t`) and that each of
our lines is separated by a new line delimiter (`/n`). We will
have to account for this when we process our dataset.

The first step is to create a vocabulary or corpus that contains all the
unique words within our dataset.



Creating the vocabulary
-----------------------

In the past, our corpus has comprised of several dictionaries consisting
of the unique words in our corpus and lookups
between word and indices. However, we can do this in a far more elegant
way by creating a vocabulary class that consists of all of the elements
required:

1.  We start by creating our `Vocabulary` class. We initialize
    this class with empty dictionaries---`word2index` and
    `word2count`. We also initialize the
    `index2word` dictionary with
    placeholders for our padding tokens, as well as our
    **Start-of-Sentence** (**SOS**) and **End-of-Sentence** (**EOS**)
    tokens. We keep a running count of the number
    of words in our vocabulary, too (which is 3 to start with as our
    corpus already contains the three tokens mentioned). These are the
    default values for an empty vocabulary; however, they will be
    populated as we read our data in:
    ```
    PAD_token = 0 
    SOS_token = 1
    EOS_token = 2
    class Vocabulary:
        def __init__(self, name):
            self.name = name
            self.trimmed = False
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD", SOS_token:                           "SOS", EOS_token: "EOS"}
            self.num_words = 3
    ```
    

2.  Next, we create the functions that we will use to populate our
    vocabulary. `addWord` takes a word as input. If this is a
    new word that is not already in our vocabulary, we add
    this word to our indices, set the count of
    this word to 1, and increment the total number of words in our
    vocabulary by 1. If the word in question is already in our
    vocabulary, we simply increment the count of this word by 1:
    ```
    def addWord(self, w):
        if w not in self.word2index:
            self.word2index[w] = self.num_words
            self.word2count[w] = 1
            self.index2word[self.num_words] = w
            self.num_words += 1
        else:
            self.word2count[w] += 1
    ```
    

3.  We also use the `addSentence` function to apply the
    `addWord` function to all the words within a given
    sentence:

    ```
    def addSentence(self, sent):
        for word in sent.split(' '):
            self.addWord(word)
    ```
    

4.  To remove low-frequency words from our vocabulary, we can implement
    a `trim` function. The function first loops through the
    word count dictionary and if the occurrence of the word is greater
    than the minimum required count, it is appended to a new list:
    ```
    def trim(self, min_cnt):
        if self.trimmed:
            return
        self.trimmed = True
        words_to_keep = []
        for k, v in self.word2count.items():
            if v >= min_cnt:
                words_to_keep.append(k)
        print('Words to Keep: {} / {} = {:.2%}'.format(
            len(words_to_keep), len(self.word2index),    
            len(words_to_keep) / len(self.word2index)))
    ```
    

5.  Finally, our indices are rebuilt from the new
    `words_to_keep` list. We set all the indices to their
    initial empty values and then repopulate them by looping through our
    kept words with the `addWord` function:
    ```
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD",\
                           SOS_token: "SOS",\
                           EOS_token: "EOS"}
        self.num_words = 3
        for w in words_to_keep:
            self.addWord(w)
    ```
    

We have now defined a vocabulary class that can be
easily populated with our input sentences. Next, we actually need to
load in our dataset to create our training data.



Loading the data
----------------

We will start loading in the data using the
following steps:

1.  The first step for reading in our data is to perform any necessary
    steps to clean the data and make it more human-readable. We start by
    converting it from Unicode into ASCII format. We can easily use a
    function to do this:
    ```
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    ```
    

2.  Next, we want to process our input strings so that they are all in
    lowercase and do not contain any trailing whitespace or punctuation,
    except the most basic characters. We can do this by using a series
    of regular expressions:
    ```
    def cleanString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s
    ```
    

3.  Finally, we apply this function within a wider
    function---`readVocs`. This function reads our data file
    into lines and then applies the `cleanString` function to
    every line. It also creates an instance of the
    `Vocabulary` class that we created earlier, meaning this
    function outputs both our data and vocabulary:

    ```
    def readVocs(datafile, corpus_name):
        lines = open(datafile, encoding='utf-8').\
            read().strip().split('\n')
        pairs = [[cleanString(s) for s in l.split('\t')]               for l in lines]
        voc = Vocabulary(corpus_name)
        return voc, pairs
    ```
    

    Next, we filter our input pairs by their maximum length. This is
    again done to reduce the potential dimensionality of our model.
    Predicting sentences that are hundreds of words long would require a
    very deep architecture. In the interest of training time, we want to
    limit our training data here to instances where the input and output
    are less than 10 words long.

4.  To do this, we create a couple of filter functions. The first one,
    `filterPair`, returns a Boolean value based on whether the
    current line has an input and output length that is less than the
    maximum length. Our second function, `filterPairs`, simply
    applies this condition to all the pairs within our dataset, only
    keeping the ones that meet this condition:
    ```
    def filterPair(p, max_length):
        return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length
    def filterPairs(pairs, max_length):
        return [pair for pair in pairs if filterPair(pair,             max_length)]
    ```
    

5.  Now, we just need to create one final function that applies all the
    previous functions we have put together and
    run it to create our vocabulary and data pairs:

    ```
    def loadData(corpus, corpus_name, datafile, save_dir, max_length):
        voc, pairs = readVocs(datafile, corpus_name)
        print(str(len(pairs)) + " Sentence pairs")
        pairs = filterPairs(pairs,max_length)
        print(str(len(pairs))+ " Sentence pairs after           trimming")
        for p in pairs:
            voc.addSentence(p[0])
            voc.addSentence(p[1])
        print(str(voc.num_words) + " Distinct words in           vocabulary")
        return voc, pairs
    max_length = 10 
    voc, pairs = loadData(corpus, corpus_name, datafile,                       max_length)
    ```
    

    We can see that our input dataset consists of
    over 200,000 pairs. When we filter this to sentences where both the
    input and output are less than 10 words long, this reduces to just
    64,000 pairs consisting of 18,000 distinct words:

    
![](./images/B12365_08_07.jpg)
    


6.  We can print a selection of our processed input/output pairs in
    order to verify that our functions have all worked correctly:

    ```
    print("Example Pairs:")
    for pair in pairs[-10:]:
        print(pair)
    ```
    

    The following output is generated:


![](./images/B12365_08_08.jpg)


It appears that we have successfully split our dataset into input and
output pairs upon which we can train our network.

Finally, before we begin building the model, we must remove the rare
words from our corpus and data pairs.



Removing rare words
-------------------

As previously mentioned, including words that only occur a few times
within our dataset will increase the
dimensionality of our model, increasing our model\'s complexity and the
time it will take to train the model. Therefore, it is preferred to
remove them from our training data to keep our model as streamlined and
efficient as possible.

You may recall earlier that we built a `trim` function into
our vocabulary, which will allow us to remove infrequently occurring
words from our vocabulary. We can now create a function to remove these
rare words and call the `trim` method from our vocabulary as
our first step. You will see that this removes a large percentage of
words from our vocabulary, indicating that the majority of the words in
our vocabulary occur infrequently. This is expected as the distribution
of words within any language model will follow a long-tail distribution.
We will use the following steps to remove the words:

1.  We first calculate the percentage of words that we will keep within
    our model:

    ```
    def removeRareWords(voc, all_pairs, minimum):
        voc.trim(minimum)
    ```
    

    This results in the following output:

    
![](./images/B12365_08_09.jpg)
    


2.  Within this same function, we loop through all the words in the
    input and output sentences. If for a given pair either the input or
    output sentence has a word that isn\'t in our
    new trimmed corpus, we drop this pair from our
    dataset. We print the output and see that even though we have
    dropped over half of our vocabulary, we only drop around 17% of our
    training pairs. This again reflects how our corpus of words is
    distributed over our individual training pairs:

    ```
    pairs_to_keep = []
    for p in all_pairs:
        keep = True
        for word in p[0].split(' '):
            if word not in voc.word2index:
                keep = False
                break
        for word in p[1].split(' '):
            if word not in voc.word2index:
                keep = False
                break
        if keep:
            pairs_to_keep.append(p)
    print("Trimmed from {} pairs to {}, {:.2%} of total".\
           format(len(all_pairs), len(pairs_to_keep),
                  len(pairs_to_keep)/ len(all_pairs)))
    return pairs_to_keep
    minimum_count = 3
    pairs = removeRareWords(voc, pairs, minimum_count)
    ```
    

    This results in the following output:

![](./images/B12365_08_10.png)


Now that we have our finalized dataset, we need to
build some functions that transform our dataset into batches of tensors
that we can pass to our model.



Transforming sentence pairs to tensors
--------------------------------------

We know that our model will not take raw text as input, but rather,
tensor representations of sentences. We will also not process our
sentences one by one, but instead in smaller
batches. For this, we require both our input and output sentences to be
transformed into tensors, where the width of the tensor represents the
size of the batch that we wish to train on:

1.  We start by creating several helper functions, which we can use to
    transform our pairs into tensors. We first create a
    `indexFromSentence` function, which grabs the index of
    each word in the sentence from the vocabulary and appends an EOS
    token to the end:
    ```
    def indexFromSentence(voc, sentence):
        return [voc.word2index[word] for word in\
                sent.split(' ')] + [EOS_token]
    ```
    

2.  Secondly, we create a `zeroPad` function, which pads any
    tensors with zeroes so that all of the sentences within the tensor
    are effectively the same length:
    ```
    def zeroPad(l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l,\
                    fillvalue=fillvalue))
    ```
    

3.  Then, to generate our input tensor, we apply
    both of these functions. First, we get the indices of our input
    sentence, then apply padding, and then transform the output into
    `LongTensor`. We will also obtain the lengths of each of
    our input sentences out output this as a tensor:
    ```
    def inputVar(l, voc):
        indexes_batch = [indexFromSentence(voc, sentence)\
                         for sentence in l]
        padList = zeroPad(indexes_batch)
        padTensor = torch.LongTensor(padList)
        lengths = torch.tensor([len(indexes) for indexes\                            in indexes_batch])
        return padTensor, lengths
    ```
    

4.  Within our network, our padded tokens should generally be ignored.
    We don\'t want to train our model on these padded tokens, so we
    create a Boolean mask to ignore these tokens. To do so, we use a
    `getMask` function, which we apply to our output tensor.
    This simply returns `1` if the output consists of a word
    and `0` if it consists of a padding token:
    ```
    def getMask(l, value=PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m
    ```
    

5.  We then apply this to our `outputVar` function. This is
    identical to the `inputVar` function, except that along
    with the indexed output tensor and the tensor of lengths, we also
    return the Boolean mask of our output tensor. This Boolean mask just
    returns `True` when there is a word within the output
    tensor and `False` when there is a
    padding token. We also return the maximum length of sentences within
    our output tensor:
    ```
    def outputVar(l, voc):
        indexes_batch = [indexFromSentence(voc, sentence) 
                         for sentence in l]
        max_target_len = max([len(indexes) for indexes in
                              indexes_batch])
        padList = zeroPad(indexes_batch)
        mask = torch.BoolTensor(getMask(padList))
        padTensor = torch.LongTensor(padList)
        return padTensor, mask, max_target_len
    ```
    

6.  Finally, in order to create our input and output batches
    concurrently, we loop through the pairs in our batch and create
    input and output tensors for both pairs using the functions we
    created previously. We then return all the necessary variables:
    ```
    def batch2Train(voc, batch):
        batch.sort(key=lambda x: len(x[0].split(" ")),\
                   reverse=True)
        
        input_batch = []
        output_batch = []
        
        for p in batch:
            input_batch.append(p[0])
            output_batch.append(p[1])
            
        inp, lengths = inputVar(input_batch, voc)
        output, mask, max_target_len = outputVar(output_                                   batch, voc)
        
        return inp, lengths, output, mask, max_target_len
    ```
    

7.  This function should be all we need to transform our training pairs
    into tensors for training our model. We can validate that this is
    working correctly by performing  a single
    iteration of our `batch2Train` function on a random
    selection of our data. We set our batch size to `5` and
    run this once:

    ```
    test_batch_size = 5
    batches = batch2Train(voc, [random.choice(pairs) for _\                            in range(test_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches
    ```
    

    Here, we can validate that our input tensor has been created
    correctly. Note how the sentences end with
    padding (0 tokens) where the sentence length is less than the
    maximum length for the tensor (in this instance, 9). The width of
    the tensor also corresponds to the batch size (in this case, 5):


![](./images/B12365_08_11.jpg)


We can also validate the corresponding output data and mask. Notice how
the `False` values in the mask overlap with the padding tokens
(the zeroes) in our output tensor:


![](./images/B12365_08_12.jpg)


Now that we have obtained, cleaned, and
transformed our data, we are ready to begin training the attention-based
model that will form the basis of our chatbot.



Constructing the model
----------------------

We start, as with our other sequence-to-sequence
models, by creating our encoder. This will transform the initial tensor
representation of our input sentence into hidden states.

### Constructing the encoder

We will now create the encoder by taking the
following steps:

1.  As with all of our PyTorch models, we start by creating an
    `Encoder` class that inherits from `nn.Module`.
    All the elements here should look familiar to the ones used in
    previous chapters:

    ```
    class EncoderRNN(nn.Module):
        def __init__(self, hidden_size, embedding,\
                     n_layers=1, dropout=0):
            super(EncoderRNN, self).__init__()
            self.n_layers = n_layers
            self.hidden_size = hidden_size
            self.embedding = embedding
    ```
    

    Next, we create our **Recurrent** **Neural**
    **Network** (**RNN**) module. In this chatbot, we will be using a
    **Gated Recurrent Unit** (**GRU**) instead of the **Long**
    **Short-Term** **Memory** (**LSTM**) models we saw before. GRUs are
    slightly less complex than LSTMs as
    although they still control the flow of
    information through the RNN, they don\'t have separate forget and
    update gates like the LSTM. We use GRUs in this instance for a few
    main reasons:

    a\) GRUs have proven to be more computationally efficient as there
    are fewer parameters to learn. This means that our model will train
    much more quickly with GRUs than with LSTMs.

    b\) GRUs have proven to have similar performance levels over short
    sequences of data as LSTMs. LSTMs are more useful when learning
    longer sequences of data. In this instance we are only using input
    sentences with 10 words or less, so GRUs should produce similar
    results.

    c\) GRUs have proven to be more effective at learning from small
    datasets than LSTMs. As the size of our training data is small
    relative to the complexity of the task we\'re trying to learn, we
    should opt to use GRUs.

2.  We now define our GRU, taking into account the size of our input,
    the number of layers, and whether we should implement dropout:

    ```
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                      dropout=(0 if n_layers == 1 else\
                               dropout), bidirectional=True)
    ```
    

    Notice here how we implement bidirectionality into our model. You
    will recall from previous chapters that a bidirectional RNN allows
    us to learn from a sentence moving
    sequentially forward through a sentence, as well as moving
    sequentially backward. This allows us to better capture the context
    of each word in the sentence relative to those that appear before
    and after it. Bidirectionality in our GRU means our encoder looks
    something like this:

    
![](./images/B12365_08_13.jpg)
    


    We maintain two hidden states, as well as outputs at each step,
    within our input sentence.

3.  Next, we need to create a forward pass for our
    encoder. We do this by first embedding our input sentences and then
    using the `pack_padded_sequence` function on our
    embeddings. This function \"packs\" our padded sequence so that all
    of our inputs are of the same length. We then pass out the packed
    sequences through our GRU to perform a forward pass:
    ```
    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded,
                                          input_lengths)
        outputs, hidden = self.gru(packed, hidden)
    ```
    

4.  After this, we unpack our padding and sum the GRU outputs. We can
    then return this summed output, along with our
    final hidden state, to complete our forward pass:
    ```
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
    outputs = outputs[:, :, :self.hidden_size] + a \
              outputs[:, : ,self.hidden_size:]
    return outputs, hidden
    ```
    

Now, we will move on to creating an attention module in the next
section.

### Constructing the attention module

Next, we need to build our attention module, which
we will apply to our encoder so that we can learn from the relevant
parts of the encoder\'s output. We will do so as follows:

1.  Start by creating a class for the attention model:
    ```
    class Attn(nn.Module):
        def __init__(self, hidden_size):
            super(Attn, self).__init__()
            self.hidden_size = hidden_size
    ```
    
2.  Then, create the `dot_score` function within this class.
    This function simply calculates the dot product of our encoder
    output with the output of our hidden state by our encoder. While
    there are other ways of transforming these two tensors into a single
    representation, using a dot product is one of the simplest:
    ```
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)
    ```
    
3.  We then use this function within our forward pass. First, calculate
    the attention weights/energies based on the `dot_score`
    method, then transpose the results, and return the softmax
    transformed probability scores:
    ```
    def forward(self, hidden, encoder_outputs):
        attn_energies = self.dot_score(hidden, \
                                       encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    ```
    

Next, we can use this attention module within our
decoder to create an attention-focused decoder.

### Constructing the decoder

We will now construct the decoder, as follows:

1.  We begin by creating our `DecoderRNN` class, inheriting
    from `nn.Module` and defining the initialization
    parameters:
    ```
    class DecoderRNN(nn.Module):
        def __init__(self, embedding, hidden_size, \
                     output_size, n_layers=1, dropout=0.1):
            super(DecoderRNN, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers
            self.dropout = dropout
    ```
    
2.  We then create our layers within this module. We will create an
    embedding layer and a corresponding dropout layer. We use GRUs again
    for our decoder; however, this time, we do not need to make our GRU
    layer bidirectional as we will be decoding the output from our
    encoder sequentially. We will also create two linear layers---one
    regular layer for calculating our output and one layer that can be
    used for concatenation. This layer is twice the width of the regular
    hidden layer as it will be used on two concatenated vectors, each
    with a length of `hidden_size`. We also initialize an
    instance of our attention module from the last section in order to
    be able to use it within our `Decoder` class:
    ```
    self.embedding = embedding
    self.embedding_dropout = nn.Dropout(dropout)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers,  dropout=(0 if n_layers == 1 else dropout))
    self.concat = nn.Linear(2 * hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
    self.attn = Attn(hidden_size)
    ```
    
3.  After defining all of our layers, we need to
    create a forward pass for the decoder. Notice how the forward pass
    will be used one step (word) at a time. We start by getting the
    embedding of the current input word and making a forward pass
    through the GRU layer to get our output and hidden states:
    ```
    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
    ```
    
4.  Next, we use the attention module to get the attention weights from
    the GRU output. These weights are then multiplied by the encoder
    outputs to effectively give us a weighted sum of our attention
    weights and our encoder output:
    ```
    attn_weights = self.attn(rnn_output, encoder_outputs)
    context = attn_weights.bmm(encoder_outputs.transpose(0,
                                                         1))
    ```
    
5.  We then concatenate our weighted context vector with the output of
    our GRU and apply a `tanh` function to get out final
    concatenated output:
    ```
    rnn_output = rnn_output.squeeze(0)
    context = context.squeeze(1)
    concat_input = torch.cat((rnn_output, context), 1)
    concat_output = torch.tanh(self.concat(concat_input))
    ```
    
6.  For the final step within our decoder, we
    simply use this final concatenated output to predict the next word
    and apply a `softmax` function. The forward pass finally
    returns this output, along with the final hidden state. This forward
    pass will be iterated upon, with the next forward pass using the
    next word in the sentence and this new hidden state:
    ```
    output = self.out(concat_output)
    output = F.softmax(output, dim=1)
    return output, hidden
    ```
    

Now that we have defined our models, we are ready to define the training
process



Defining the training process
-----------------------------

The first step of the training process is to define the measure of loss
for our models. As our input tensors may consist
of padded sequences, owing to our input sentences all being of different
lengths, we cannot simply calculate the difference between the true
output and the predicted output tensors. To account for this, we will
define a loss function that applies a Boolean mask over our outputs and
only calculates the loss of the non-padded tokens:

1.  In the following function, we can see that we calculate
    cross-entropy loss across the whole output tensors. However, to get
    the total loss, we only average over the elements of the tensor that
    are selected by the Boolean mask:
    ```
    def NLLMaskLoss(inp, target, mask):
        TotalN = mask.sum()
        CELoss = -torch.log(torch.gather(inp, 1,\                        target.view(-1, 1)).squeeze(1))
        loss = CELoss.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, TotalN.item()
    ```
    

2.  For the majority of our training, we need two main functions---one
    function, `train()`, which performs training on a single
    batch of our training data and another function,
    `trainIters()`, which iterates through our whole dataset
    and calls `train()` on each of the individual batches. We
    start by defining `train()` in order to train
    on a single batch of data. Create the
    `train()` function, then get the gradients to 0, define
    the device options, and initialize the variables:
    ```
    def train(input_variable, lengths, target_variable,\
              mask, max_target_len, encoder, decoder,\
              embedding, encoder_optimizer,\
              decoder_optimizer, batch_size, clip,\
              max_length=max_length):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)
        loss = 0
        print_losses = []
        n_totals = 0
    ```
    

3.  Then, perform a forward pass of the inputs and sequence lengths
    though the encoder to get the output and hidden states:
    ```
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    ```
    

4.  Next, we create our initial decoder input, starting with SOS tokens
    for each sentence. We then set the initial hidden state of our
    decoder to be equal to that of the encoder:

    ```
    decoder_input = torch.LongTensor([[SOS_token for _ in \
                                       range(batch_size)]])
    decoder_input = decoder_input.to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    ```
    

    Next, we implement teacher forcing. If you recall from the last
    chapter, teacher forcing, when generating output sequences with some
    given probability, we use the true previous output token rather than
    the predicted previous output token to generate the next word in our
    output sequence. Using teacher forcing helps our
    model converge much more quickly; however, we
    must be careful not to make the teacher forcing ratio too high or
    else our model will be too reliant on the teacher forcing and will
    not learn to generate the correct output independently.

5.  Determine whether we should use teacher forcing for the current
    step:
    ```
    use_TF = True if random.random() < teacher_forcing_ratio else False
    ```
    

6.  Then, if we do need to implement teacher forcing, run the following
    code. We pass each of our sequence batches through the decoder to
    obtain our output. We then set the next input as the true output
    (`target`). Finally, we calculate and accumulate the loss
    using our loss function and print this to the console:
    ```
    for t in range(max_target_len):
    decoder_output, decoder_hidden = decoder(
      decoder_input, decoder_hidden, encoder_outputs)
    decoder_input = target_variable[t].view(1, -1)
    mask_loss, nTotal = NLLMaskLoss(decoder_output, \
         target_variable[t], mask[t])
    loss += mask_loss
    print_losses.append(mask_loss.item() * nTotal)
    n_totals += nTotal
    ```
    

7.  If we do not implement teacher forcing on a given batch, the
    procedure is almost identical. However, instead of using the true
    output as the next input into the sequence, we use the one generated
    by the model:
    ```
    _, topi = decoder_output.topk(1)
    decoder_input = torch.LongTensor([[topi[i][0] for i in \
                                       range(batch_size)]])
    decoder_input = decoder_input.to(device)
    ```
    

8.  Finally, as with all of our models, the final steps are to perform
    backpropagation, implement gradient clipping, and step through both
    of our encoder and decoder optimizers to update the weights using
    gradient descent. Remember that we clip out
    gradients in order to prevent the
    vanishing/exploding gradient problem, which was discussed in earlier
    chapters. Finally, our training step returns our average loss:
    ```
    loss.backward()
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    return sum(print_losses) / n_totals
    ```
    

9.  Next, as previously stated, we need to create the
    `trainIters()` function, which repeatedly
    calls our training function on different
    batches of input data. We start by splitting our data into batches
    using the `batch2Train` function we created earlier:
    ```
    def trainIters(model_name, voc, pairs, encoder, decoder,\
                   encoder_optimizer, decoder_optimizer,\
                   embedding, encoder_n_layers, \
                   decoder_n_layers, save_dir, n_iteration,\
                   batch_size, print_every, save_every, \
                   clip, corpus_name, loadFilename):
        training_batches = [batch2Train(voc,\
                           [random.choice(pairs) for _ in\
                            range(batch_size)]) for _ in\
                            range(n_iteration)]
    ```
    

10. We then create a few variables that will allow us to count
    iterations and keep track of the total loss over each epoch:
    ```
    print('Starting ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
    ```
    

11. Next, we define our training loop. For each iteration, we get a
    training batch from our list of batches. We then extract the
    relevant fields from our batch and run a
    single training iteration using these
    parameters. Finally, we add the loss from this batch to our overall
    loss:
    ```
    print("Beginning Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, \
              max_target_len = training_batch
        loss = train(input_variable, lengths,\
                     target_variable, mask, max_target_len,\
                     encoder, decoder, embedding, \
                     encoder_optimizer, decoder_optimizer,\
                     batch_size, clip)
        print_loss += loss
    ```
    

12. On every iteration, we also make sure we print our progress so far,
    keeping track of how many iterations we have completed and what our
    loss was for each epoch:
    ```
    if iteration % print_every == 0:
        print_loss_avg = print_loss / print_every
        print("Iteration: {}; Percent done: {:.1f}%;\
        Mean loss: {:.4f}".format(iteration,
                              iteration / n_iteration \
                              * 100, print_loss_avg))
        print_loss = 0
    ```
    

13. For the sake of completion, we also need to save our model state
    after every few epochs. This allows us to revisit any historical
    models we have trained; for example, if our model were to begin
    overfitting, we could revert back to an earlier iteration:
    ```
    if (iteration % save_every == 0):
        directory = os.path.join(save_dir, model_name,\
                                 corpus_name, '{}-{}_{}'.\
                                 format(encoder_n_layers,\
                                 decoder_n_layers, \
                                 hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': voc.__dict__,
                    'embedding': embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
    ```
    

Now that we have completed all the necessary steps
to begin training our model, we need to create functions to allow us to
evaluate the performance of the model.



Defining the evaluating process
-------------------------------

Evaluating a chatbot is slightly different from
evaluating other sequence-to-sequence models. In our text translation
task, an English sentence will have one direct translation into German.
While there may be multiple correct translations, for the most part,
there is a single correct translation from one language into another.

For chatbots, there are multiple different valid outputs. Take the
following three lines from some conversations with a chatbot:

**Input**: *\"Hello\"*

**Output**: *\"Hello\"*

**Input**: *\"Hello\"*

**Output**: *\"Hello. How are you?\"*

**Input**: \"*Hello\"*

**Output**: *\"What do you want?\"*

Here, we have three different responses, each one equally valid as a
response. Therefore, at each stage of our conversation with our chatbot,
there will be no single \"correct\" response. So, evaluation is much
more difficult. The most intuitive way of testing
whether a chatbot produces a valid output is by
having a conversation with it! This means we need to set up our chatbot
in a way that enables us to have a conversation with it to determine
whether it is working well:

1.  We will start by defining a class that will allow us to decode the
    encoded input and produce text. We do this by
    using what is known as a **greedy encoder**. This simply means that
    at each step of the decoder, our model takes the word with the
    highest predicted probability as the output. We start by
    initializing the `GreedyEncoder()` class with our
    pretrained encoder and decoder:
    ```
    class GreedySearchDecoder(nn.Module):
        def __init__(self, encoder, decoder):
            super(GreedySearchDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
    ```
    

2.  Next, define a forward pass for our decoder. We pass the input
    through our encoder to get our encoder\'s output and hidden state.
    We take the encoder\'s final hidden layer to be the first hidden
    input to the decoder:
    ```
    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = \
                        self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:decoder.n_layers]
    ```
    

3.  Then, create the decoder input with SOS tokens and initialize the
    tensors to append decoded words to (initialized as a single zero
    value):
    ```
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    ```
    

4.  After that, iterate through the sequence, decoding one word at a
    time. We perform a forward pass through the
    encoder and add a `max` function to obtain the
    highest-scoring predicted word and its score, which we then append
    to the `all_tokens` and` all_scores` variables.
    Finally, we take this predicted token and use it as the next input
    to our decoder. After the whole sequence has been iterated over, we
    return the complete predicted sentence:

    ```
    for _ in range(max_length):
        decoder_output, decoder_hidden = self.decoder\
            (decoder_input, decoder_hidden, encoder_outputs)
        decoder_scores, decoder_input = \
             torch.max (decoder_output, dim=1)
        all_tokens = torch.cat((all_tokens, decoder_input),\
                                dim=0)
        all_scores = torch.cat((all_scores, decoder_scores),\
                                dim=0)
        decoder_input = torch.unsqueeze(decoder_input, 0)
    return all_tokens, all_scores
    ```
    

    All the pieces are beginning to come together. We have the defined
    training and evaluation functions, so the final step is to write a
    function that will actually take our input as text, pass it to our
    model, and obtain a response from the model. This will be the
    \"interface\" of our chatbot, where we actually have our
    conversation.

5.  We first define an `evaluate()` function, which takes our
    input function and returns the predicted output words. We start by
    transforming our input sentence into indices using our vocabulary.
    We then obtain a tensor of the lengths of each of these sentences
    and transpose it:
    ```
    def evaluate(encoder, decoder, searcher, voc, sentence,\
                 max_length=max_length):
        indices = [indexFromSentence(voc, sentence)]
        lengths = torch.tensor([len(indexes) for indexes \
                                in indices])
        input_batch = torch.LongTensor(indices).transpose(0, 1)
    ```
    

6.  Then, we assign our lengths and input tensors
    to the relevant devices. Next, run the inputs through the searcher
    (`GreedySearchDecoder`) to obtain the word indices of the
    predicted output. Finally, we transform these word indices back into
    word tokens before returning them as the function output:
    ```
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, scores = searcher(input_batch, lengths, \
                              max_length)
    decoded_words = [voc.index2word[token.item()] for \
                     token in tokens]
    return decoded_words
    ```
    

7.  Finally, we create a `runchatbot` function, which acts as
    the interface with our chatbot. This function takes human-typed
    input and prints the chatbot\'s response. We create this function as
    a `while` loop that continues until we terminate the
    function or type `quit` as our input:
    ```
    def runchatbot(encoder, decoder, searcher, voc):
        input_sentence = ''
        while(1):
            try:
                input_sentence = input('> ')
                if input_sentence == 'quit': break
    ```
    

8.  We then take the typed input and normalize it, before passing the
    normalized input to our `evaluate()` function, which
    returns the predicted words from the chatbot:
    ```
    input_sentence = cleanString(input_sentence)
    output_words = evaluate(encoder, decoder, searcher,\
                            voc, input_sentence)
    ```
    

9.  Finally, we take these output words and format
    them, ignoring the EOS and padding tokens, before printing the
    chatbot\'s response. Because this is a `while` loop, this
    allows us to continue the conversation with the chatbot
    indefinitely:
    ```
    output_words[:] = [x for x in output_words if \
                       not (x == 'EOS' or x == 'PAD')]
    print('Response:', ' '.join(output_words))
    ```
    

Now that we have constructed all the functions necessary to train,
evaluate, and use our chatbot, it\'s time to begin the final
step---training our model and conversing with our trained chatbot.



Training the model
------------------

As we have defined all the necessary functions,
training the model just becomes a case or initializing our
hyperparameters and calling our training functions:

1.  We first initialize our hyperparameters. While these are only
    suggested hyperparameters, our models have been set up in a way that
    will allow them to adapt to whatever hyperparameters they are
    passed. It is good practice to experiment with different
    hyperparameters to see which ones result in an optimal model
    configuration. Here, you could experiment with increasing the number
    of layers in your encoder and decoder, increasing or decreasing the
    size of the hidden layers, or increasing the batch size. All of
    these hyperparameters will have an effect on how well your model
    learns, as well as a number of other factors, such as the time it
    takes to train the model:
    ```
    model_name = 'chatbot_model'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.15
    batch_size = 64
    ```
    

2.  After that, we can load our checkpoints. If we have previously
    trained a model, we can load the checkpoints
    and model states from previous iterations. This saves us from having
    to retrain our model each time:
    ```
    loadFilename = None
    checkpoint_iter = 4000
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
    ```
    

3.  After that, we can begin to build our models. We first load our
    embeddings from the vocabulary. If we have already trained a model,
    we can load the trained embeddings layer:
    ```
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    ```
    

4.  We then do the same for our encoder and decoder, creating model
    instances using the defined hyperparameters. Again, if we have
    already trained a model, we simply load the trained model states
    into our models:
    ```
    encoder = EncoderRNN(hidden_size, embedding, \
                         encoder_n_layers, dropout)
    decoder = DecoderRNN(embedding, hidden_size, \ 
                         voc.num_words, decoder_n_layers,
                         dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    ```
    

5.  Last but not least, we specify a device for
    each of our models to be trained on. Remember, this is a crucial
    step if you wish to use GPU training:

    ```
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    ```
    

    If this has all worked correctly and your models have been created
    with no errors, you should see the following:

    
![](./images/B12365_08_14.jpg)
    

6.  We also need to define the learning rates of our models and our
    decoder learning ratio. You will find that your model performs
    better when the decoder carries out larger
    parameter updates during gradient descent. Therefore, we introduce a
    decoder learning ratio to apply a multiplier to the learning rate so
    that the learning rate is greater for the decoder than it is for the
    encoder. We also define how often our model prints and saves the
    results, as well as how many epochs we want our model to run for:
    ```
    save_dir = './'
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    epochs = 4000
    print_every = 1
    save_every = 500
    ```
    

7.  Next, as always when training models in PyTorch, we switch our
    models to training mode to allow the parameters to be updated:
    ```
    encoder.train()
    decoder.train()
    ```
    

8.  Next, we create optimizers for both the encoder and decoder. We
    initialize these as Adam optimizers, but other optimizers will work
    equally well. Experimenting with different optimizers may yield
    different levels of model performance. If you have trained a model
    previously, you can also load the optimizer states if required:
    ```
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), \
                                   lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), 
                   lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(\
                                       encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(\
                                       decoder_optimizer_sd)
    ```
    

9.  The final step before running the training is
    to make sure CUDA is configured to be called if you wish to use GPU
    training. To do this, we simply loop through the optimizer states
    for both the encoder and decoder and enable CUDA across all of the
    states:
    ```
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    ```
    

10. Finally, we are ready to train our model. This
    can be done by simply calling the `trainIters` function
    with all the required parameters:

    ```
    print("Starting Training!")
    trainIters(model_name, voc, pairs, encoder, decoder,\
               encoder_optimizer, decoder_optimizer, \
               embedding, encoder_n_layers, \
               decoder_n_layers, save_dir, epochs, \
                batch_size,print_every, save_every, \
                clip, corpus_name, loadFilename)
    ```
    

    If this is working correctly, you should see the following output
    start to print:


![](./images/B12365_08_15.jpg)


Your model is now training! Depending on a number
of factors, such as how many epochs you have set your model to train for
and whether you are using a GPU, your model may take some time to train.
When it is complete, you will see the following output. If everything
has worked correctly, your model\'s average loss will be significantly
lower than when you started training, showing that your model has
learned something useful:


![](./images/B12365_08_16.jpg)


Now that our model has been trained, we can begin the evaluation process
and start using our chatbot.

### Evaluating the model

Now that we have successfully created and trained our model, it is time to evaluate its performance. We will do so by taking the following steps:

1.  To begin the evaluation, we first switch our
    model into evaluation mode. As with all other PyTorch models, this
    is done to prevent any further parameter updates occurring within
    the evaluation process:
    ```
    encoder.eval()
    decoder.eval()
    ```
    

2.  We also initialize an instance of `GreedySearchDecoder` in
    order to be able to perform the evaluation and return the predicted
    output as text:
    ```
    searcher = GreedySearchDecoder(encoder, decoder)
    ```
    

3.  Finally, to run the chatbot, we simply call the
    `runchatbot` function, passing it `encoder`,
    `decoder`, `searcher`, and `voc`:

    ```
    runchatbot(encoder, decoder, searcher, voc)
    ```
    

    Doing so will open up an input prompt for you to enter your text:


![](./images/B12365_08_17.jpg)


Entering your text here and pressing *Enter* will
send your input to the chatbot. Using our trained model, our chatbot
will create a response and print it to the console:


![](./images/B12365_08_18.jpg)


You can repeat this process as many times as you like to have a
\"conversation\" with the chatbot. At a simple conversational level, the
chatbot can produce surprisingly good results:


![](./images/B12365_08_19.jpg)


However, once the conversation gets more complex,
it will become obvious that the chatbot isn\'t capable of the same level
of conversation as a human:


![](./images/B12365_08_20.jpg)


In many cases, your chatbot\'s responses may be nonsensical:


![](./images/B12365_08_21.jpg)


#### Summary

In this lab, we applied all the knowledge we learned from our
recurrent models and our sequence-to-sequence models and combined them
with an attention mechanism to construct a fully working chatbot. While
conversing with our chatbot is unlikely to be indistinguishable from
talking to a real human, with a considerably larger dataset we might
hope to achieve an even more realistic chatbot.

Although sequence-to-sequence models with attention were
state-of-the-art in 2017, machine learning is a rapidly progressing
field and since then, there have been multiple improvements made to
these models. In the final chapter, we will discuss some of these
state-of-the-art models in more detail, as well as cover several other
contemporary techniques used in machine learning for NLP, many of which
are still in development.
