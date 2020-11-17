
*[]{#_idTextAnchor156}Chapter 9*: The Road Ahead {#_idParaDest-147}
================================================

::: {#_idContainer280}
The field of machine learning is rapidly expanding, with new revelations
being made almost yearly. The field of machine learning for NLP is no
exception, with advancements being made rapidly and the performance of
machine learning models on NLP tasks incrementally increasing.

So far in this book, we have discussed a number of machine learning
methodologies that allow us to build models to perform NLP tasks such as
classification, translation, and approximating conversation via a
chatbot. However, as we have seen so far, the performance of our models
has been worse and relative to that of a human being. Even using the
techniques we have examined so far, including sequence-to-sequence
networks with attention mechanisms, we are unlikely to train a chatbot
model that will match or outperform a real person. However, we will see
in this chapter that recent developments in the field of NLP have been
made that bring us one step closer to the goal of creating chatbots that
are indistinguishable from humans.

In this chapter, we will explore a couple of state-of-the art machine
learning models for NLP and examine some of the features that result in
superior performance. We will then turn to look at several other NLP
tasks that are currently the focus of much research, and how machine
learning techniques might be used to solve them.

In this chapter, we will cover the following topics:

-   Exploring state-of-the-art NLP machine learning
-   Future NLP tasks
-   Semantic role labeling
-   Constituency parsing
-   Textual entailment
-   Machine comprehension


Exploring state-of-the-art NLP machine learning {#_idParaDest-148}
===============================================

::: {#_idContainer280}
While the techniques we have learned in this book so far are highly
useful methodologies for training our own machine learning model from
scratch, they are far from the most sophisticated models being developed
globally. Companies and research groups are constantly striving to
create the most advanced machine learning models that will achieve the
highest performance on a number of NLP tasks.

Currently, there are two NLP models that have the best performance and
could be considered state-of-the-art: **BERT** and **GPT-2**. Both
models are forms of **generalized language models**. We will discuss
these in more detail in the upcoming sections.

[]{#_idTextAnchor158}

BERT {#_idParaDest-149}
----

**BERT**, which stands for **Bidirectional Encoder Representations from
Transformers**, was developed by Google in 2018 and is widely considered
to be the leading model in the field of NLP, having achieved leading
performance in natural language inference and question-answering tasks.
Fortunately, this has been released as an open source model, so this can
be downloaded and used for NLP tasks of your own.

BERT was released as a pre-trained model, which means users can download
and implement BERT without the need to retrain the model from scratch
each time. The pre-trained model is trained on several corpuses,
including the whole of Wikipedia (consisting of 2.5 billion words) and
another corpus of books (which includes a further 800 million words).
However, the main element of BERT that sets it apart from other similar
models is the fact that it provides a deep, bidirectional, unsupervised
language representation, which is shown to provide a more sophisticated,
detailed representation, thus leading to improved performance in NLP
tasks.

### Embeddings

While traditional embedding layers (such as GLoVe) form a single
representation of a word that is agnostic to the meaning of the word
within the sentence, the bidirectional BERT model attempts to form
representations based on its context. For example, in these two
sentences, the word *bat* has two different meanings.

"The bat flew past my window"

"He hit the baseball with the bat"

Although the word *bat* is a noun in both sentences, we can discern that
the context and meaning of the word is obviously very different,
depending on the other words around it. Some words may also have
different meanings, depending on whether they are a noun or verb within
the sentence:

"She used to match to light the fire"

"His poor performance meant they had no choice but to fire him"

Using the bidirectional language model to form context-dependent
representations of words is what truly makes BERT stand out as a
state-of-the-art model. For any given token, we obtain its input
representation by combining the token, position, and segment embeddings:

<div>

::: {#_idContainer231 .IMG---Figure}
![Figure 9.1 -- BERT architecture ](2_files/B12365_09_1.jpg)
:::

</div>

Figure 9.1 -- BERT architecture

However, it is important to understand how the model arrives at these
initial context-dependent token-embeddings.

### Masked language modeling

In order to create this bidirectional language representation, BERT uses
two different techniques, the first of which is masked language
modeling. This methodology effectively hides 15% of the words within the
input sentences by replacing them with a masking token. The model then
tries to predict the true values of the masked words, based on the
context of the other words in the sentence. This prediction is made
bidirectionally in order to capture the context of the sentence in both
directions:

**Input**: *We \[MASK\_1\] hide some of the \[MASK\_2\] in the sentence*

**Labels**: *MASK\_1 = randomly, MASK\_2 = words*

If our model can learn to predict the correct context-dependent words,
then we are one step closer to context-dependent representation.

### Next sentence prediction

The other technique that BERT uses to learn the language representation
is next sentence prediction. In this methodology, our model receives two
sentences and our model learns to predict whether the second sentence is
the sentence that follows the first sentence; for example:

**Sentence A**: *\"I like to drink coffee\"*

**Sentence B**: *\"It is my favorite drink\"*

**Is Next Sentence?**: *True*

**Sentence A**: *\"I like to drink coffee\"*

**Sentence B**: *\"The sky is blue\"*

**Is Next Sentence?**: *False*

By passing our model pairs of sentences like this, it can learn to
determine whether any two sentences are related and follow one another,
or whether they are just two random, unrelated sentences. Learning these
sentence relationships is useful in a language model as many NLP-related
tasks, such as question-answering, require the model to understand the
relationship between two sentences. Training a model on next sentence
prediction allows the model to identify some kind of relationship
between a pair of sentences, even if that relationship is very basic.

BERT is trained using both methodologies (masked language modeling and
next sentence prediction), and the combined loss function of both
techniques is minimized. By using two different training methods, our
language representation is sufficiently robust and learns how sentences
are formed and structured, as well as how different sentences relate to
one another.

[]{#_idTextAnchor159}

BERT--Architecture {#_idParaDest-150}
------------------

The model architecture builds upon many of the principles we have seen
in the previous chapters to provide a sophisticated language
representation using bidirectional encoding. There are two different
variants of BERT, each consisting of a different number of layers and
attention heads:

-   **BERT Base**: 12 transformer blocks (layers), 12 attention heads,
    \~110 million parameters
-   **BERT Large**: 24 transformer blocks (layers), 16 attention heads,
    \~340 million parameters

While BERT Large is just a deeper version of BERT Base with more
parameters, we will focus on the architecture of BERT Base.

BERT is built by following the principle of a **transformer**, which
will now be explained in more detail.

### Transformers

The model architecture builds upon many of the principles we have seen
so far in this book. By now, you should be familiar with the concept of
encoders and decoders, where our model learns an encoder to form a
representation of an input sentence, and then learns a decoder to decode
this representation into a final output, whether this be a
classification or translation task:

<div>

::: {#_idContainer232 .IMG---Figure}
![Figure 9.2 -- Transformer workflow ](2_files/B12365_09_2.jpg)
:::

</div>

Figure 9.2 -- Transformer workflow

However, our transformer adds another element of sophistication to this
approach, where a transformer actually has a stack of encoders and a
stack of decoders, with each decoder receiving the output of the final
encoder as its input:

<div>

::: {#_idContainer233 .IMG---Figure}
![Figure 9.3 -- Transformer workflow for multiple encoders
](2_files/B12365_09_3.jpg)
:::

</div>

Figure 9.3 -- Transformer workflow for multiple encoders

Within each encoder layer, we find two constituent parts: a
self-attention layer and a feed-forward layer. The self-attention layer
is the layer that receives the model\'s input first. This layer causes
the encoder to examine other words within the input sentence as it
encodes any received word, making the encoding context aware. The output
from the self-attention layer is passed forward to a feed-forward layer,
which is applied independently to each position. This can be illustrated
diagrammatically like so:

<div>

::: {#_idContainer234 .IMG---Figure}
![Figure 9.4 -- Feedforward layer ](2_files/B12365_09_4.jpg)
:::

</div>

Figure 9.4 -- Feedforward layer

Our decoder layers are almost identical in structure to our encoders,
but they incorporate an additional attention layer. This attention layer
helps the decoder focus on the relevant part of the encoded
representation, similar to how we saw attention working within our
sequence-to-sequence models:

<div>

::: {#_idContainer235 .IMG---Figure}
![Figure 9.5 -- Attention methodology ](2_files/B12365_09_5.jpg)
:::

</div>

Figure 9.5 -- Attention methodology

We know that our decoders take input from our final encoder, so one
linked encoder/decoder might look something like this:

<div>

::: {#_idContainer236 .IMG---Figure}
![Figure 9.6 -- Linked encoder/decoder array ](2_files/B12365_09_6.jpg)
:::

</div>

Figure 9.6 -- Linked encoder/decoder array

This should provide you with a useful overview of how the different
encoders and decoders are stacked up within the larger model. Next, we
will examine the individual parts in more detail.

### Encoders

The unique property of transformers is that words flow through the
encoder layers individually and each word in each position has its own
path. While there are some dependencies within the self-attention layer,
these don\'t exist within the feed-forward layer. The vectors for the
individual words are obtained from an embedding layer and then fed
through a self-attention layer before being fed through a feed-forward
network:

<div>

::: {#_idContainer237 .IMG---Figure}
![Figure 9.7 -- Encoder layout ](2_files/B12365_09_7.jpg)
:::

</div>

Figure 9.7 -- Encoder layout

Self-attention is arguably the most complex component of the encoder, so
we will examine this in more detail first. Let\'s say we have a
three-word input sentence; for example, *\"This is fine\"*. For each
word within this sentence, we represent them as a single word vector
that was obtained from the embedding layer of our model. We then extract
three vectors from this single word vector: a query vector, a key
vector, and a value vector. These three vectors are obtained by
multiplying our word vector by three different weight matrices that are
obtained while training the model.

If we call our word embeddings for each word in our input sentence,
*Ethis*, *Eis*, and *Efine*, we can calculate our query, key, and value
vectors like so:

**Query vectors**:

![](2_files/Formula_09_001.png)

![](2_files/Formula_09_002.png)

![](2_files/Formula_09_003.png)

![](2_files/Formula_09_004.png)

**Key vectors**:

![](2_files/Formula_09_005.png)

![](2_files/Formula_09_006.png)

![](2_files/Formula_09_007.png)

![](2_files/Formula_09_008.png)

**Value vectors**:

![](2_files/Formula_09_009.png)

![](2_files/Formula_09_010.png)

![](2_files/Formula_09_011.png)

![](2_files/Formula_09_012.png)

Now that we know how to calculate each of these vectors, it is important
to understand what each of them represents. Essentially, each of these
is an abstraction of a concept within the attention mechanism. This will
become apparent once we see how they are calculated.

Let\'s continue with our working example. We need to consider each word
within our input sentence in turn. To do this, we calculate a score for
each pair of query/key vectors in our sentence. This is done by
obtaining the dot product of each query/key vector pair for each word
within our input sentence. For example, to calculate the scores for the
first word in the sentence, \"this\", we calculate the dot product
between the query vector for \"this\" and the key vector in position 0.
We repeat this for the key vectors in all other positions within the
input sentence, so we obtain *n* scores for the first word in our input
sentence, where *n* is the length of the sentence:

**Scores (\"this\")**:

![](2_files/Formula_09_013.png)

![](2_files/Formula_09_014.png)

![](2_files/Formula_09_015.png)

Next, we apply a softmax function to each of these scores so that the
value of each is now between 0 and 1 (as this helps prevent exploding
gradients and makes gradient descent more efficient and easily
calculable). We then multiply each of these scores by the value vectors
and sum these all up to obtain a final vector, which is then passed
forward within the encoder:

**Final vector (\"this\")**:

![](2_files/Formula_09_016.png)

![](2_files/Formula_09_017.png)

![](2_files/Formula_09_018.png)

![](2_files/Formula_09_019.png)

We then repeat this procedure for all the words within the input
sentence so that we obtain a final vector for each word, incorporating
an element of self-attention, which is then passed along the encoder to
the feed-forward network. This self-attention process means
that[]{#_idIndexMarker489} our encoder knows where to look within the
input sentence to obtain the information it needs for the task.

In this example, we only learned a single matrix of weights for our
query, key, and value vectors. However, we can actually learn multiple
different matrices for each of these elements and apply these
simultaneously across our input sentence to obtain our final outputs.
This is []{#_idIndexMarker490}what\'s known as **multi-headed
attention** and allows us to perform more complex attention
calculations, relying on multiple different learned patterns rather than
just a single attention mechanism.

We know that BERT incorporates 12 attention heads, meaning that 12
different weight matrices are learned for *Wq*, *Wk*, and *Wv*.

Finally, we need a way for our encoders to account for the order of
words in the input sequence. Currently, our model treats each word in
our input sequence independently, but in reality, the order of the words
in the input sequence will make a huge difference to the overall
meaning[]{#_idIndexMarker491} of the sentence. To account for this, we
use **positional encoding**.

To apply this, our model takes each input embedding and adds a
positional encoding vector to each one individually. These positional
vectors are learned by our model, following a specific pattern to help
them determine the position of each word in the sequence. In theory,
adding these positional vectors to our initial embeddings should
translate into meaningful distances between our final vectors, once they
are projected into the individual query, key, and value vectors:

*x0 = Raw Embedding*

*t0 = Positional Encoding*

*E0 = Embedding with Time Signal*

*x0 + t0 = E0*

Our model learns different positional encoding vectors for each position
(*t*[0]{.subscript}, *t*[1]{.subscript}, and so on), which
we[]{#_idIndexMarker492} then apply to each word in our input sentence
before these even enter our encoder:

<div>

::: {#_idContainer257 .IMG---Figure}
![Figure 9.8 -- Adding input to the encoder ](2_files/B12365_09_8.jpg)
:::

</div>

Figure 9.8 -- Adding input to the encoder

Now that we have []{#_idIndexMarker493}covered the main components of
the encoder, it\'s time to look at the other side of the model and see
how the decoder is constructed.

### Decoders

The components in []{#_idIndexMarker494}decoders are much the same of
those in encoders. However, rather than receiving the raw input sentence
like encoders do, the decoders in our transformer receive their inputs
from the outputs of our encoders.

Our stacked encoders process our input sentence and we are left with a
set of attention vectors, *K* and *V*, which are used within the
encoder-decoder attention layer of our decoder. This allows it to focus
only on the relevant parts of the input sequence:

<div>

::: {#_idContainer258 .IMG---Figure}
![Figure 9.9 -- Stacked decoders ](2_files/B12365_09_9.jpg)
:::

</div>

Figure 9.9 -- Stacked decoders

At each time step, our[]{#_idIndexMarker495} decoders use a combination
of the previously generated words in the sentence and the *K,V*
attention vectors to generate the next word in the sentence. This
process is repeated iteratively until the decoder generates an \<END\>
token, indicating that it has completed generating the final output. One
given time step on the transformer decoder may look like this:

<div>

::: {#_idContainer259 .IMG---Figure}
![Figure 9.10 -- Transformer decoder ](2_files/B12365_09_10.jpg)
:::

</div>

Figure 9.10 -- Transformer decoder

It is worth noting here that the self-attention layers within the
decoders operate in a slightly different way to those found in our
encoders. Within our decoder, the self-attention layer
[]{#_idIndexMarker496}only focuses on earlier positions within the
output sequence. This is done by masking any future positions of the
sequence by setting them to minus infinity. This means that when the
classification happens, the softmax calculation always results in a
prediction of 0.

The encoder-decoder attention layer works in the same way as the
multi-headed self-attention layer within our encoder. However, the main
difference is that it creates a query matrix from the layer below and
takes the key and values matrix from the output of the encoders.

These encoder and decoder parts comprise our transformer, which forms
the basis for BERT. Next, we will look at some of the applications of
BERT and examine a few variations that have shown increased performance
at specific tasks.

[]{#_idTextAnchor160}

Applications of BERT {#_idParaDest-151}
--------------------

Being state-of-the-art, BERT obviously has a number of practical
applications. Currently, it is being[]{#_idIndexMarker497} used in a
number of Google products that you probably use on a daily basis;
namely, suggested replies and smart compose in Gmail (where Gmail
predicts your expected sentence based on what you are currently typing)
and autocomplete within the Google search engine (where you type the
first few characters you wish to search for and the drop-down list will
predict what you are going to search for).

As we saw in the previous chapter, chatbots are one of the most
impressive things NLP deep learning can be used for, and the use of BERT
has led to some very impressive chatbots indeed. In fact,
question-answering is one of the main things that BERT excels at,
largely due to the fact that it is trained on a large knowledge base
(Wikipedia) and is able to answer questions in a syntactically correct
way (due to being trained with next sentence prediction in mind).

We are still not at the stage where chatbots are indistinguishable from
conversations with real humans, and the ability of BERT to draw from its
knowledge base is extremely limited. However, some of the results
achieved by BERT are promising and, taking into account how quickly the
field of NLP machine learning is progressing, this suggests that this
may become a reality very soon.

Currently, BERT is only able to address a very narrow type of NLP task
due to the way it is trained. However, there are many variations of BERT
that have been changed in subtle ways to achieve increased performance
at specific tasks. These include, but are not limited to, the following:

-   **roBERTa**: A variation []{#_idIndexMarker498}of BERT, built by
    Facebook. Removes the next sentence prediction element of BERT, but
    enhances the word masking strategy by implementing dynamic masking.
-   **xlm**/**BERT**: Also built []{#_idIndexMarker499}by Facebook, this
    model applies a dual-language training mechanism to BERT that allows
    it to learn relationships between words in different languages. This
    allows BERT to be used effectively for machine translation tasks,
    showing improved performance over basic sequence-to-sequence models.
-   **distilBERT**: A more compact []{#_idIndexMarker500}version of
    BERT, retaining 95% of the original but halving the number of
    learned parameters, reducing the model's total size and training
    time.
-   **ALBERT**: This Google[]{#_idIndexMarker501} trained model uses its
    own unique training method called sentence order prediction. This
    variation of BERT has been shown to outperform the standard BERT
    across a number of tasks and is now considered state-of-the-art
    ahead of BERT (illustrating just how quickly things can change!).

While BERT is perhaps the most well known, there are also other
transformer-based models that are considered state-of-the-art. The major
one that is often considered a rival to BERT is GPT-2.

[]{#_idTextAnchor161}

GPT-2 {#_idParaDest-152}
-----

GPT-2, while similar to BERT, differs in some subtle ways. While both
models are based upon the[]{#_idIndexMarker502} transformer architecture
previously outlined, BERT uses a form of attention known as
self-attention, while GPT-2 uses masked self-attention. Another subtle
difference between the two is that GPT-2 is constructed in such a way
that it can to output one token at a time.

This is because GPT-2 is essentially auto-regressive in the way it
works. This means that when it generates an output (the first word in a
sentence), this output is added recursively to the input. This input is
then used to predict the next word in the sentence and is repeated until
a complete sentence has been generated. You can see this in the
following example:

**Step 1:**

**Input**: *What color is the sky?*

**Output**: *The \...*

We then add the predicted output to the end of our input and repeat this
step:

**Step 2:**

**Input**: *What color is the sky? The*

**Output**: *sky*

We repeat this process until we have generated the entire sentence:

**Step 3:**

**Input**: *What color is the sky? The sky*

**Output**: *is*

**Step 4:**

**Input**: *What color is the sky? The sky is*

**Output**: *blue*

This is one of the key trade-offs in terms of performance between BERT
and GPT-2. The fact that BERT[]{#_idIndexMarker503} is trained
bidirectionally means this single-token generation is not possible;
however, GPT-2 is not bidirectional, so it only considers previous words
in the sentence when making predictions, which is why BERT outperforms
GPT-2 when predicting missing words within a sentence.

[]{#_idTextAnchor162}

Comparing self-attention and masked self-attention {#_idParaDest-153}
--------------------------------------------------

This difference[]{#_idIndexMarker504} is also apparent in the way the
two[]{#_idIndexMarker505} different models implement attention. Since
BERT is bidirectional, its attention mechanism is able to consider the
context of the whole input sentence and determine where exactly in the
input sentence to look:

<div>

::: {#_idContainer260 .IMG---Figure}
![Figure 9.11 -- Self-Attention mechanism ](2_files/B12365_09_11.jpg)
:::

</div>

Figure 9.11 -- Self-Attention mechanism

On the other []{#_idIndexMarker506}hand, GPT-2 is not bidirectional,
so[]{#_idIndexMarker507} the masked self-attention mechanism is only
able to look at tokens it has already seen and is not able to \"look
ahead\":

<div>

::: {#_idContainer261 .IMG---Figure}
![Figure 9.12 -- Masked self-attention mechanism
](2_files/B12365_09_12.jpg)
:::

</div>

Figure 9.12 -- Masked self-attention mechanism

[]{#_idTextAnchor163}

GPT-2 -- Ethics {#_idParaDest-154}
---------------

One very interesting consideration of GPT-2 was the author\'s
consideration of ethics when considering[]{#_idIndexMarker508} whether
to release the model to the public or not.

The language generation capabilities of GPT-2 mean that it is capable of
generating very realistic and syntactically correct sentences using very
few prompts. By providing it with just a couple of lines of input to
identify a subject matter, the model is able to generate paragraphs of
detailed information about this subject, regardless of whether it is
actually true or not! One, now infamous, example was the \"Unicorn\"
story, where the model received one sentence input:

**Input**: *\"In a shocking finding, scientists discovered a herd of
unicorns living in a remote, previously unexplored valley, in the Andes
Mountains. Even more surprising to the researchers was the fact that the
unicorns spoke perfect English.\"*

The model gave the following as output:

**Output**: *The scientist named the population, after their distinctive
horn, Ovid\'s Unicorn. These four-horned, silver-white unicorns were
previously unknown to science.*

*Now, after almost two centuries, the mystery of what sparked this odd
phenomenon is finally solved.*

*Dr. Jorge Pérez, an evolutionary biologist from the University of La
Paz, and several companions, were exploring the Andes Mountains when
they found a small valley, with no other animals or humans. Pérez
noticed that the valley had what appeared to be a natural fountain,
surrounded by two peaks of rock and silver snow.*

This is only the first three paragraphs of a much larger nine-paragraph
response that the model generated from very little input. What\'s
notable is that the sentences all make perfect sense (regardless of the
impossible subject matter!), that the paragraphs flow together in a
logical order, and that the model was able to generate all of this from
a very small input.

While this is extremely impressive in terms of performance and what
it\'s possible to achieve from building deep NLP models, it does raise
some concerns about the ethics of such models and how they can be used
(and abused!).

With the rise of \"fake news\" and the spread of misinformation using
the internet, examples like this illustrate how simple it is to generate
realistic text using these models. Let\'s consider an example where an
agent wishes to generate fake news on a number of subjects online. Now,
they don\'t even need to write the fake information themselves. In
theory, they could train NLP models to do this for them, before
disseminating this fake information on the internet. The authors of
GPT-2 paid particular attention to this when training and releasing the
model to the public, noting that the model had the potential to be
abused and misused, therefore only releasing the larger more
sophisticated models to the public once they saw no evidence of misuse
of the smaller models.

This may become a key focus of NLP deep learning moving forward. As we
approach chatbots and text generators such as GPT-2 that can approach
human levels of sophistication, the []{#_idIndexMarker509}uses and
misuses of these models need to be fully understood. Studies have shown
that GPT-2 generated text was deemed to be almost as credible (72%) as
real human-written articles from the New York Times (83%). As we
continue to develop even more sophisticated deep NLP models in the
future, these numbers are likely to converge as model-generated text
becomes more and more realistic moving forward.

Furthermore, the authors of GPT-2 also demonstrated that the model can
be fine-tuned for misuse. By fine-tuning GPT-2 on ideologically extreme
positions and generating text, it was shown that propaganda text can be
generated, which supports these ideologies. While it was also shown that
counter-models could be trained to detect these model-generated texts,
we may again face further problems here in the future as these models
become even more sophisticated.

These ethical considerations are worth keeping in mind as NLP models
become even more complex and performant over time. While the models you
train for your own purposes may not have been intended for any misuse,
there is always the possibility that they could be used for purposes
that were unintended. Always consider the potential applications of any
model that you use.


Future NLP tasks {#_idParaDest-155}
================

::: {#_idContainer280}
While the majority of this book has been focused on text classification
and sequence generation, there are []{#_idIndexMarker510}a number of
other NLP tasks that we haven\'t really touched on. While many of these
are more interesting from an academic perspective rather than a
practical perspective, it\'s important to understand these tasks as they
form the basis of how language is constructed and formed. Anything we,
as NLP data scientists, can do to better understand the formation of
natural language will only improve our understanding of the subject
matter. In this section, we will discuss, in more detail, four key areas
of future development in NLP:

-   Constituency parsing
-   Semantic role labeling
-   Textual entailment
-   Machine comprehension

[]{#_idTextAnchor165}

Constituency parsing {#_idParaDest-156}
--------------------

Constituency parsing (also known as syntactic parsing) is the act of
identifying parts of a sentence and []{#_idIndexMarker511}assigning a
syntactic structure to it. This []{#_idIndexMarker512}syntactic
structure is largely determined by the use of context-free grammars,
meaning that using syntactic parsing, we can identify the underlying
grammatical structure of a given sentence and map it out. Any sentence
can be broken down into a \"parse tree,\" which is a graphical
representation of this underlying sentence structure, while syntactic
parsing is the methodology by which this underlying structure is
detected and determines how this tree is built.

We will begin by discussing this underlying grammatical structure. The
idea of a \"constituency\" within a sentence is somewhat of an
abstraction, but the basic assumption is that a sentence consists of
multiple \"groups\" of words, each one of which is a constituency.
Grammar, in its basic form, can be said to be an index of all possible
types of constituencies that can occur within a sentence.

Let\'s first consider the most basic []{#_idIndexMarker513}type of
constituent, the **noun phrase**. Nouns within a sentence are fairly
simple to identify as they are words that define objects or entities. In
the following sentence, we can identify three nouns:

"Jeff the chef cooks dinner"

Jeff - Proper noun, denotes a name

Chef - A chef is an entity

Dinner - Dinner is an object/thing

However, a noun phrase is slightly different as each noun phrase should
refer to one single entity. In the preceding sentence, even though
*Jeff* and *chef* are both nouns, the phrase *Jeff the chef* refers to
one single person, so this can be considered a noun phrase. But how can
we determine syntactically that the noun phrase refers to a single
entity? One simple[]{#_idIndexMarker514} way is to place the phrase
before a verb and see if the sentence makes
[]{#_idIndexMarker515}syntactic sense. If it does, then chances are, the
phrase is a noun phrase:

Jeff the chef cooks...

Jeff the chef runs...

Jeff the chef drinks...

There exist a variety of different phrases that we are able to identify,
as well as a number of complex grammatical rules that help us to
identify them. We first identify the individual grammatical features
that each sentence can be broken down into:

<div>

::: {#_idContainer262 .IMG---Figure}
![](3_files/B12365_09_29.jpg)
:::

</div>

Now that we know[]{#_idIndexMarker516} that sentences are composed of
[]{#_idIndexMarker517}constituents, and that constituents can be made up
of several individual grammars, we can now start to map out our
sentences based on their structure. For example, take the following
example sentence:

"The boy hit the ball"

We can start by breaking this sentence down into two parts: a noun
phrase and a verb phrase:

<div>

::: {#_idContainer263 .IMG---Figure}
![Figure 9.13 Breaking down a sentence into its grammatical components
](3_files/B12365_09_13.jpg)
:::

</div>

Figure 9.13 -- Breaking down a sentence into its grammatical components

We then repeat this []{#_idIndexMarker518}process for each of the
phrases to split them into even[]{#_idIndexMarker519} smaller
grammatical components. We can split this noun phrase into a determiner
and a noun:

<div>

::: {#_idContainer264 .IMG---Figure}
![Figure 9.14 Breaking down the noun phrase ](3_files/B12365_09_14.jpg)
:::

</div>

Figure 9.14 -- Breaking down the noun phrase

Again, we do this for the verb phrase to break it down into a verb and
another noun phrase:

<div>

::: {#_idContainer265 .IMG---Figure}
![Figure 9.15 -- Breaking down the verb phrase
](3_files/B12365_09_15.jpg)
:::

</div>

Figure 9.15 -- Breaking down the verb phrase

We can iterate again and []{#_idIndexMarker520}again, breaking down the
various parts of our sentence into smaller and
[]{#_idIndexMarker521}smaller chunks until we are left with a **parse
tree**. This parse []{#_idIndexMarker522}tree conveys the entirety of
the syntactic structure of our sentence. We can see the parse tree of
our example in its entirety here:

<div>

::: {#_idContainer266 .IMG---Figure}
![Figure 9.16 -- Parse tree of the sentence ](3_files/B12365_09_16.jpg)
:::

</div>

a

Figure 9.16 -- Parse tree of the sentence

While these parse trees allow us to see the syntactic structure of our
sentences, they are far from perfect. From this structure, we can
clearly see that there are two noun phrases with a verb
[]{#_idIndexMarker523}taking place. However, from the preceding
[]{#_idIndexMarker524}structure, it is not clear what is actually taking
place. We have an action between two objects, but it is not clear from
syntax alone what is taking place. Which party is doing the action to
whom? We will see that some of this ambiguity is captured by semantic
role labeling.

[]{#_idTextAnchor166}

Semantic role labeling {#_idParaDest-157}
----------------------

Semantic role labeling is[]{#_idIndexMarker525} the process of assigning
labels[]{#_idIndexMarker526} to words or phrases within a sentence that
indicates their semantic role within a sentence. In broad terms, this
involves identifying the predicate of the sentence and determining how
each of the other terms within the sentence are related to this
predicate. In other words, for a given sentence, semantic role labeling
determines \"Who did what to whom and where/when?\"

So, for a given sentence, we can generally break down a sentence into
its constituent parts, like so:

<div>

::: {#_idContainer267 .IMG---Figure}
![Figure 9.17 Breaking down a sentence into its constituent parts
](3_files/B12365_09_17.jpg)
:::

</div>

Figure 9.17 Breaking down a sentence into its constituent parts

These parts of a[]{#_idIndexMarker527} sentence have specific
semantic[]{#_idIndexMarker528} roles. The **predicate** of any given
sentence represents the event occurring within the sentence, while all
the other parts of the sentence relate back to a given predicate. In
this sentence, we can label our \"Who\" as the agent of the predicate.
The **agent** is the thing that causes the event. We can also label our
\"Whom\" as the theme of our predicate. The **theme** is the element of
our sentence most affected by the event in question:

<div>

::: {#_idContainer268 .IMG---Figure}
![Figure 9.18 -- Breaking down the roles ](3_files/B12365_09_18.jpg)
:::

</div>

Figure 9.18 -- Breaking down the roles

In theory, each[]{#_idIndexMarker529} word or phrase in a sentence can
be labeled with its specific semantic component. An almost comprehensive
table for this is as follows:

<div>

::: {#_idContainer269 .IMG---Figure}
![](3_files/B12365_09_Table_01.jpg)
:::

</div>

By performing semantic role labeling, we can assign a specific role to
every part of a sentence. This is very useful in NLP as it allows a
model to \"understand\" a sentence better so that rather
[]{#_idIndexMarker530}than a sentence just being an assortment
of[]{#_idIndexMarker531} roles, it is understood as a combination of
semantic roles that better convey what is actually happening in the
event being described by the sentence.

When we read the sentence *\"The boy kicked the ball\"*, we inherently
know that there is a boy, there is a ball, and that the boy is kicking
the ball. However, all the NLP models we have looked at so far would
comprehend this sentence by looking at the individual words in the
sentence and creating some representation of them. It is unlikely that
the fact that there are two \"things\" and that object one (the boy) is
performing some action (kicking) on object two (the ball) would be
understood by the systems we have seen so far. Introducing an element of
semantic roles to our models could better help our systems form more
realistic representations of sentences by defining the subjects of the
sentences and the interactions between them.

One thing that semantic role labeling helps with greatly is the
identification of sentences that convey the same meaning but are not
grammatically or syntactically the same; such as the following, for
example:

The man bought the apple from the shop

The shop sold the man an apple

The apple was bought by the man from the shop

The apple was sold by the shop to the man

These sentences have essentially the same meaning, although they clearly
do not contain all the same words in the same order. By applying
semantic role labeling to these sentences, we can determine that the
predicate/agent/theme are all the same.

We previously saw how constituency parsing/syntactic parsing can be used
to identify the syntactic []{#_idIndexMarker532}structure of a sentence.
Here, we[]{#_idIndexMarker533} can see how we can break down the simple
sentence \"I bought a cat\" into its constituent parts -- pronoun, verb,
determinant, and noun:

<div>

::: {#_idContainer270 .IMG---Figure}
![Figure 9.19 -- Constituency parsing ](3_files/B12365_09_19.jpg)
:::

</div>

Figure 9.19 -- Constituency parsing

However, this does not shed any insight on the semantic role each part
of the sentence is playing. Is the cat being bought by me or am I being
bought by the cat? While the syntactic role is useful for understanding
the structure of the sentence, it doesn\'t shed as much light on the
semantic meaning. A useful analogy is that of image captioning. In a
model trained to label images, we would hope to achieve a caption that
describes what is in an image. Semantic labeling is the opposite of
this, where we take a sentence and try to abstract a mental \"image\" of
what action is taking place in the sentence.

But what context is semantic role labeling useful for in NLP? In short,
any NLP task that requires an element of \"understanding\" the content
of text can be enhanced by the addition of roles. This could be anything
from document summarization, question-answering, or sentence
translation. For example, using semantic role labeling to identify the
predicate of our sentence and the related semantic components, we could
train a model to identify the components that contribute essential
information to the sentence and drop those that do not.

Therefore, being able[]{#_idIndexMarker534} to train models to perform
accurate and[]{#_idIndexMarker535} efficient semantic role labeling
would have useful applications for the rest of NLP. The earliest
semantic role labeling systems were purely rule-based, consisting of
basic sets of rules derived from grammar. These have since evolved to
incorporate statistical modeling approaches before the recent
developments in deep learning, which meant that it is possible to train
classifiers to identify the relevant semantic roles within a sentence.

As with any classification task, this is a supervised learning problem
that requires a fully annotated sentence in order to train a model that
will identify the semantic roles of previously unseen sentences.
However, the availability of such annotated sentences is highly scarce.
The gigantic language models, such as BERT, that we saw earlier in this
chapter are trained on raw sentences and do not require labeled
examples. However, in the case of semantic role labeling, our models
require the use of correctly labeled sentences to be able to perform
this task. While datasets do exist for this purpose, they are not large
and versatile enough to train a fully comprehensive, accurate model that
will perform well on a variety of sentences.

As you can probably imagine, the latest state-of-the-art approaches to
solving the semantic role labeling task have all been neural
network-based. Initial models used LSTMs and bidirectional LSTMs
combined with GLoVe embeddings in order to perform classification on
sentences. There have also been variations of these models that
incorporate convolutional layers, which have also shown good
performance.

However, it will be no surprise to learn that these state-of-the-art
models are BERT-based. Using BERT has shown exemplary performance in a
whole variety of NLP-related tasks, and semantic role labeling is no
exception. Models incorporating BERT have been trained holistically to
predict part-of-speech tags, perform syntactic parsing, and perform
semantic role labeling simultaneously and have shown good results.

Other studies have also shown that graph convolutional networks are
effective at semantic labeling. Graphs are constructed with nodes and
edges, where the nodes within the graph represent semantic constituents
and the edges represent the relationships between parent and child
parts.

A number of open source models for semantic role labeling are also
available. The SLING parser[]{#_idIndexMarker536} from Google is trained
to perform semantic []{#_idIndexMarker537}annotations of data. This
model uses a bidirectional LSTM to encode sentences and a
transition-based recurrent unit for decoding. The model simply takes
text tokens as input and outputs roles without any further symbolic
representation:

<div>

::: {#_idContainer271 .IMG---Figure}
![Figure 9.20 -- Bi-directional LSTM (SLING) ](3_files/B12365_09_20.jpg)
:::

</div>

Figure 9.20 -- Bi-directional LSTM (SLING)

It is worth noting that SLING is still a work in progress. Currently, it
is not sophisticated enough to extract facts accurately from arbitrary
texts. This indicates that there is much []{#_idIndexMarker538}work to
be done in the field before a true, accurate[]{#_idIndexMarker539}
semantic role parser can be created. When this is done, a semantic role
parser could easily be used as part of an ensemble machine learning
model to label semantic roles within a sentence, which is then used
within a wider machine learning model to enhance the model\'s
\"understanding\" of text.

[]{#_idTextAnchor167}

Textual entailment {#_idParaDest-158}
------------------

Textual entailment[]{#_idIndexMarker540} is another methodology by which
we can train []{#_idIndexMarker541}models in an attempt to better
understand the meaning of a sentence. In textual entailment, we attempt
to identify a directional relationship between two pieces of text. This
relationship exists whenever the truth from one piece of text follows
from another piece of text. This means that, given two texts, if the
second text can be held to be true by the information within the first
text, we can say that there is a positive directional relationship
between these two texts.

This task is often set up in the following fashion, with our first text
labeled as text and our second text labeled as our hypothesis:

**Text**: *If you give money to charity, you will be happy*

**Hypothesis**: *Giving money to charity has good consequences*

This is an []{#_idIndexMarker542}example of **positive textual
entailment**. If the []{#_idIndexMarker543}hypothesis follows from the
text, then there can be said to be a directional relationship between
the two texts. It is important to set up the example with a
text/hypothesis as this defines the direction of the
[]{#_idIndexMarker544}relationship. The majority of the time, this
relationship is not symmetrical. For example, in this example, sentence
one entails sentence two (we can infer sentence two to be true based on
the information in sentence one). However, we cannot infer that sentence
one is true based on the information in sentence two. While it is
possible that both statements are indeed true, if we cannot deduce that
there is a directional relationship between the two, we cannot infer one
from the other.

There also[]{#_idIndexMarker545} exists a **negative textual
entailment**. This is when the statements are contradictory; such as the
following, for example:

**Text**: *If you give money to charity, you will be happy*

**Hypothesis**: *Giving money to charity has bad consequences*

In this example, the text does not entail the hypothesis; instead, the
text contradicts the hypothesis. Finally, it is also possible to
determine that there is **no textual entailment** between two
[]{#_idIndexMarker546}sentences if there is no relationship between
them. This means that the two statements are not necessarily
contradictory, but rather that the text does not entail the hypothesis:

**Text**: *If you give money to charity, you will be happy*

**Hypothesis**: *Giving money to charity will make you relaxed*

The ambiguity of natural language makes this an interesting task from an
NLP perspective. Two sentences can have a different syntactic structure,
a different semantic structure, and consist of entirely different words
but still have very similar meanings. Similarly, two sentences can
consist of the same words and entities but have very different meanings.

This is where using models to be able to quantify the meaning of text is
particularly useful. Textual entailment is also a unique problem in that
two sentences[]{#_idIndexMarker547} may not have exactly the same
meaning, yet one []{#_idIndexMarker548}can still be inferred from the
other. This requires an element of linguistic deduction that is not
present in most language models. By incorporating elements of linguistic
deduction in our models going forward, we can better capture the meaning
of texts, as well as be able to determine whether two texts contain the
same information, regardless of whether their representations are
similar.

Fortunately, simple textual entailment models are not difficult to
create, and LSTM-based models have been shown to be effective. One setup
that may prove effective is that of a Siamese LSTM network.

We set up our model as a multi-class classification problem where two
texts can be positively or negatively entailed or have no entailment. We
feed our two texts into a dual-input model, thereby obtaining embeddings
for the two texts, and pass them through bidirectional LSTM layers. The
two outputs are then compared somehow (using some tensor operation)
before they\'re fed through a final LSTM layer. Finally, we perform
classification on the output using a softmax layer:

<div>

::: {#_idContainer272 .IMG---Figure}
![Figure 9.21 -- Siamese LSTM network ](3_files/B12365_09_21.jpg)
:::

</div>

Figure 9.21 -- Siamese LSTM network

While these models are far from perfect, they represent the first steps
toward creating a fully[]{#_idIndexMarker549} accurate textual
entailment model and open up the []{#_idIndexMarker550}possibilities
toward integrating this into language models moving forward.

[]{#_idTextAnchor168}

Machine comprehension {#_idParaDest-159}
---------------------

So far in this book, we []{#_idIndexMarker551}have referred mostly to
NLP, but[]{#_idIndexMarker552} being able to process language is just
part of the picture. When you or I read a sentence, we not only read,
observe, and process the individual words, but we also build an inherent
understanding of what the sentence actually means. Being able to train
models that not only comprehend a sentence but can also form an
understanding of the ideas being expressed within it is arguably the
next step in NLP. The true definition of this field is very loosely
defined, but it is often referred to as machine[]{#_idIndexMarker553}
comprehension or **natural language understanding** (**NLU**).

At school, we are taught reading comprehension from a young age. You
probably learned this skill a long time[]{#_idIndexMarker554} ago and is
something you now take for granted. Often, you probably don\'t
[]{#_idIndexMarker555}even realize you are doing it; in fact, you are
doing it right now! Reading comprehension is simply the act of reading a
text, understanding this text, and being able to answer questions about
the text. For example, take a look at the following text:

As a method of disinfecting water, bringing it to its boiling point at
100 °C (212 °F) is the oldest and most effective way of doing this since
it does not affect its taste. It is effective despite contaminants or
particles present in it, and is a single step process that eliminates
most microbes responsible for causing intestine-related diseases. The
boiling point of water is 100 °C (212 °F) at sea level and at normal
barometric pressure.

Given that you understand this text, you should now be able to answer
the following questions about it:

**Q**: *What is the boiling point of water?*

**A**: *100 °C (212 °F)*

**Q**: *Does boiling water affect its taste?*

**A**: *No*

This ability to understand text and answer questions about it form the
basis for our machine comprehension task. We wish to be able to train a
machine learning model that can not only form an understanding of a
text, but also be able to answer questions about it in grammatically
correct natural language.

The benefits of this are numerous, but a very intuitive use case would
be to build a system that acts as a knowledge base. Currently, the way
search engines work is that we run a search (in Google or a similar
search engine) and the search engine returns a selection of documents.
However, to find a particular piece of information, we must still infer
the correct []{#_idIndexMarker556}information from our returned
document. The entire process might []{#_idIndexMarker557}look something
like this:

<div>

::: {#_idContainer273 .IMG---Figure}
![Figure 9.22 -- Process of finding information
](3_files/B12365_09_22.jpg)
:::

</div>

Figure 9.22 -- Process of finding information

In this example, to answer the question *\"What is the boiling point of
water?\",* we first formulate our question. Then, we search for the
subject matter on a search engine. This would probably be some reduced
representation of the question; for example, *\"water boiling point\"*.
Our search engine would then return some relevant documents, most likely
the Wikipedia entry for water, which we would then manually have to
search and use it to infer the answer to our question. While this
methodology is effective, machine comprehension models would allow this
process to be streamlined somewhat.

Let\'s say we have a perfect model that is able to fully comprehend and
answer questions on a text corpus. We could train this model on a large
source of data such as a large text scrape of the internet or Wikipedia
and form a model that acts as a large knowledge base. By doing this, we
would then be able to query the knowledge base with real questions and
the answers would be returned automatically. This removes the
knowledge[]{#_idIndexMarker558} inference step of our diagram as
the[]{#_idIndexMarker559} inference is taken care of by the model as the
model already has an understanding of the subject matter:

<div>

::: {#_idContainer274 .IMG---Figure}
![Figure 9.23 -- New process using a model ](3_files/B12365_09_23.jpg)
:::

</div>

Figure 9.23 -- New process using a model

In an ideal world, this would be as simple as typing *\"What is the
boiling point of water?\"* into a search engine and receiving *\"100 °C
(212 °F)\"* back as an answer.

Let\'s assume we have a simplified version of this model to begin with.
Let\'s assume we already know the document that the answer to our asked
question appears in. So, given the Wikipedia page on water, can we train
a model to answer the question *\"What is the boiling point of
water?\".* A simple way of doing this to begin with, rather than
incorporating the elements of a full language model, would be to simply
return the passage of the Wikipedia page that contains the answer to our
question.

An architecture that[]{#_idIndexMarker560} we could train to achieve
this task[]{#_idIndexMarker561} might look something like this:

<div>

::: {#_idContainer275 .IMG---Figure}
![Figure 9.24 -- Architecture of the model ](3_files/B12365_09_24.jpg)
:::

</div>

Figure 9.24 -- Architecture of the model

Our model takes our question that we want answered and our document that
contains our question as inputs. These are then passed through an
embedding layer to form a tensor-based representation of each, and then
an encoding layer to form a further reduced vector representation.

Now that our question and documents are represented as vectors, our
matching layer attempts to determine where in the document vectors we
should look to obtain the answer to our question. This is done through a
form of attention mechanism whereby our question determines what parts
of our document vectors we should look at in order to answer the
question.

Finally, our fusing layer is designed to capture the long-term
dependencies of our matching layer, combine all the received information
from our matching layer, and perform a decoding step to obtain our final
answers. This layer takes the form of a bidirectional RNN that decodes
our matching layer output into final predictions. We predict two
values[]{#_idIndexMarker562} here -- a start point and an endpoint --
using a []{#_idIndexMarker563}multiclass classification. This represents
the start and end points within our document that contain the answer to
our initial question. If our document contained 100 words and the
sentence between word 40 and word 50 contained the answer to our
question, our model would ideally predict 40 and 50 for the values of
the start and end points. These values could then be easily used to
return the relevant passage from the input document.

While returning relevant areas of a target document is a useful model to
train, it is not the same as a true machine comprehension model. In
order to do that, we must incorporate elements of a larger language
model.

In any machine comprehension task, there are actually three elements at
play. We already know that there is a question and answer, but there is
also a relevant context that may determine the answer to a given
question. For example, we can ask the following question:

*What day is it today?*

The answer may differ, depending on the context in which the question is
asked; for example, Monday, Tuesday, March the 6th, Christmas Day.

We must also note that the relationship between the question and answer
is bidirectional. When given a knowledge base, it is possible for us to
generate an answer given a question, but []{#_idIndexMarker564}it also
follows that we are[]{#_idIndexMarker565} able to generate a question
given an answer:

<div>

::: {#_idContainer276 .IMG---Figure}
![Figure 9.25 -- Relationship between the question and answer
](3_files/B12365_09_25.jpg)
:::

</div>

Figure 9.25 -- Relationship between the question and answer

A true machine []{#_idIndexMarker566}comprehension may be able to
perform **question generation** (**QG**), as well as
**question-answering** (**QA**). The most obvious solution
to[]{#_idIndexMarker567} this is to train two separate models, one for
each task, and compare their results. In theory, the output of our QG
model should equal the input of our QA model, so by comparing the two,
we can provide simultaneous evaluation:

<div>

::: {#_idContainer277 .IMG---Figure}
![Figure 9.26 -- Comparison between QG and QA models
](3_files/B12365_09_26.jpg)
:::

</div>

Figure 9.26 -- Comparison between QG and QA models

However, a more comprehensive[]{#_idIndexMarker568} model would be able
to perform[]{#_idIndexMarker569} these two tasks simultaneously, thereby
generating a question from an answer and answering a question, much like
humans are able to do:

<div>

::: {#_idContainer278 .IMG---Figure}
![Figure 9.27 Dual model representation ](3_files/B12365_09_27.jpg)
:::

</div>

Figure 9.27 -- Dual model representation

In fact, recent []{#_idIndexMarker570}advances in NLU have meant that
such []{#_idIndexMarker571}models are now a reality. By combining many
elements, we are able to create a neural network structure that is able
to perform the function of the dual model, as illustrated
[]{#_idIndexMarker572}previously. This is known as the **dual ask-answer
network**. In fact, our model contains most of the components of neural
networks that we have seen in this book so far, that is, embedding
layers, convolutional layers, encoders, decoders, and attention layers.
The full architecture of the ask-answer network looks similar to the
following:

<div>

::: {#_idContainer279 .IMG---Figure}
![Figure 9.28 -- Architecture of ask-answer network
](3_files/B12365_09_28.jpg)
:::

</div>

Figure 9.28 -- Architecture of ask-answer network

We can[]{#_idIndexMarker573} make []{#_idIndexMarker574}the following
observations here:

-   The model's **inputs** are the question, answer, and context, as
    previously outlined, but also the question and answer shifted right
    ward.
-   Our **embedding** layer convolves across GLoVe embedded vectors for
    characters and words in order to create a combined representation.
-   Our **encoders** consist of LSTMs, with applied attention.
-   Our **outputs** are also RNN-based and decode our output one word at
    a time to generate final questions and answers.

While pre-trained ask-answer networks exist, you could practice
implementing your newly acquired PyTorch skills and try building and
training a model like this yourself.

Language comprehension models like these are likely to be one of the
major focuses of study within NLP[]{#_idIndexMarker575} over the coming
years, and new papers are []{#_idIndexMarker576}likely to be published
with great frequency moving forward.


Summary {#_idParaDest-160}
=======

::: {#_idContainer280}
In this chapter, we first examined several state-of-the-art NLP language
models. BERT, in particular, seems to have been widely accepted as the
industry standard state-of-the-art language model, and BERT and its
variants are widely used by businesses in their own NLP applications.

Next, we examined several areas of focus for machine learning moving
forward; namely semantic role labeling, constituency parsing, textual
entailment, and machine comprehension. These areas will likely make up a
large percentage of the current research being conducted in NLP moving
forward.

Now that you have a well-rounded ability and understanding when it comes
to NLP deep learning models and how to implement them in PyTorch,
perhaps you\'ll feel inclined to be a part of this research moving
forward. Whether this is in an academic or business context, you now
hopefully know enough to create your own deep NLP projects from scratch
and can use PyTorch to create the models you need to solve any NLP task
you require. By continuing to improve your skills and by being aware and
keeping up to date with all the latest developments in the field, you
will surely be a successful, industry leading NLP data scientist!
