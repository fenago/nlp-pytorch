
*Chapter 4*: Text Preprocessing, Stemming, and Lemmatization
=================================================================================


Textual data can be gathered from a number of different sources and
takes many different forms. Text can be tidy and readable or raw and
messy and can also come in many different styles and formats. Being able
to preprocess this data so that it can be converted into a standard
format before it reaches our NLP models is what we\'ll be looking at in
this chapter.

Stemming and lemmatization, similar to tokenization, are other forms of
NLP preprocessing. However, unlike tokenization, which reduces a
document into individual words, stemming and lemmatization are attempts
to reduce these words further to their lexical roots. For example,
almost any verb in English has many different variations, depending on
tense:

*He jumped*

*He is jumping*

*He jumps*

While all these words are different, they all relate to the same root
word -- **jump**. Stemming and lemmatization are both techniques we can
use to reduce word variations to their common roots.

In this chapter, we will explain how to perform preprocessing on textual
data, as well as explore both stemming and lemmatization and show how
these can be implemented in Python.

In this chapter, we will cover the following topics:

-   Text preprocessing
-   Stemming
-   Lemmatization
-   Uses of stemming and lemmatization


Technical requirements
======================


For the text preprocessing in this chapter, we will mostly use inbuilt
Python functions, but we will also use the external
`BeautifulSoup` package. For stemming and lemmatization, we
will use the NLTK Python package. All the code in this chapter can be
found at
<https://github.com/PacktPublishing/Hands-On-Natural-Language-Processing-with-PyTorch-1.x/tree/master/Chapter4>.


Text preprocessing
==================


Textual data can come in a variety of formats and styles. Text may be in
a structured, readable format or in a more raw,
unstructured format. Our text may contain punctuation and symbols that
we don\'t wish to include in our models or may contain HTML and other
non-textual formatting. This is of particular concern when scraping text
from online sources. In order to prepare our text so that it can be
input into any NLP models, we must perform preprocessing. This will
clean our data so that it is in a standard format. In this section, we
will illustrate some of these preprocessing steps in more
detail.



Removing HTML
-------------

When scraping text from online sources, you may find that your text
contains HTML markup and other non-textual
artifacts. We do not generally want to include these in our NLP inputs
for our models, so these should be removed by default. For example, in
HTML, the `<b>` tag indicates that the text following it
should be in bold font. However, this does not contain any textual
information about the content of the sentence, so we should remove this.
Fortunately, in Python, there is a package called
`BeautifulSoup` that allows us to remove all HTML in a few
lines:

```
input_text = "<b> This text is in bold</br>, <i> This text is in italics </i>"
output_text =  BeautifulSoup(input_text, "html.parser").get_text()
print('Input: ' + input_text)
print('Output: ' + output_text)
```


This returns the following output:


![](./images/B12365_04_01.jpg)

Figure 4.1 -- Removing HTML

The preceding screenshot shows that the HTML has been successfully
removed. This could be useful in any situations where HTML code may be
present within raw text data, such as when scraping a web page for
data.



Converting text into lowercase
------------------------------

It is standard practice when preprocessing text to convert everything
into lowercase. This is because any two words that
are the same should be considered semantically identical, regardless of
whether they are capitalized or not. \'`Cat`\',
\'`cat`\', and \'`CAT`\' are all the same words but
just have different elements capitalized. Our models will generally
consider these three words as separate entities as they are not
identical. Therefore, it is standard practice to convert all words into
lowercase so that these words are all semantically and structurally
identical. This can be done very easily within Python using the
following lines of code:

```
input_text = ['Cat','cat','CAT']
output_text =  [x.lower() for x in input_text]
print('Input: ' + str(input_text))
print('Output: ' + str(output_text))
```


This returns the following output:


![Figure 4.2 -- Converting input into lowercase
](./images/B12365_04_02.jpg)

Figure 4.2 -- Converting input into lowercase

This shows that the inputs have all been transformed into identical
lowercase representations. There are a few examples where capitalization
may actually provide additional semantic information. For example, *May*
(the month) and *may* (meaning *might*) are semantically different and
*May* (the month) will always be capitalized. However, instances like
this are very rare and it is much more efficient to convert everything
into lowercase than trying to account for these rare examples.

It is worth noting that capitalization may be
useful in some tasks such as part of speech tagging, where a capital
letter may indicate the word\'s role in the sentence, and named entity
recognition, where a capital letter may indicate that a word is a proper
noun rather than the non-proper noun alternative; for example, *Turkey*
(the country) and *turkey* (the bird).



Removing punctuation
--------------------

Sometimes, depending on the type of model being constructed, we may wish
to remove punctuation from our input text. This is
particularly useful in models where we are aggregating word counts, such
as in a bag-of-words representation. The presence of a full stop or a
comma within the sentence doesn\'t add any useful information about the
semantic content of the sentence. However, more complicated models that
take into account the position of punctuation within the sentence may
actually use the position of the punctuation to infer a different
meaning. A classic example is as follows:

*The panda eats shoots and leaves*

*The panda eats, shoots, and leaves*

Here, the addition of a comma transforms the sentence describing a
panda\'s eating habits into a sentence describing an armed robbery of a
restaurant by a panda! Nevertheless, it is still important to be able to
remove punctuation from sentences for the sake of consistency. We can do
this in Python by using the `re` library, to match any
punctuation using a regular expression, and the `sub()`
method, to replace any matched punctuation with an empty character:

```
input_text = "This ,sentence.'' contains-£ no:: punctuation?"
output_text = re.sub(r'[^\w\s]', '', input_text)
print('Input: ' + input_text)
print('Output: ' + output_text)
```


This returns the following output:


![Figure 4.3 -- Removing punctuation from input
](./images/B12365_04_03.jpg)

Figure 4.3 -- Removing punctuation from input

This shows that the punctuation has been removed from the input
sentence.

There may be instances where we may not wish to directly remove
punctuation. A good example would be the use of
the ampersand (`&`), which in almost every instance is used
interchangeably with the word \"`and`\". Therefore, rather
than completely removing the ampersand, we may instead opt to replace it
directly with the word \"`and`\". We can easily implement this
in Python using the` .replace()` function:

```
input_text = "Cats & dogs"
output_text = input_text.replace("&", "and")
print('Input: ' + input_text)
print('Output: ' + output_text)
```


This returns the following output:


![Figure 4.4 -- Removing and replacing punctuation
](./images/B12365_04_04.jpg)

Figure 4.4 -- Removing and replacing punctuation

It is also worth considering specific circumstances where punctuation
may be essential for the representation of a
sentence. One crucial example is email addresses. Removing the
`@` from email addresses doesn\'t make the address any more
readable:

`name@gmail.com`

Removing the punctuation returns this:

namegmailcom

So, in instances like this, it may be preferable to remove the whole
item altogether, according to the requirements and purpose of your NLP
model.



Replacing numbers
-----------------

Similarly, with numbers, we also want to standardize our outputs.
Numbers can be written as digits (9, 8, 7) or as actual words (nine,
eight, seven). It may be worth transforming these
all into a single, standardized representation so that 1 and one are not
treated as separate entities. We can do this in Python using the
following methodology:

```
def to_digit(digit):
    i = inflect.engine()
    if digit.isdigit():
        output = i.number_to_words(digit)
    else:
        output = digit
    return output
input_text = ["1","two","3"]
output_text = [to_digit(x) for x in input_text]
print('Input: ' + str(input_text))
print('Output: ' + str(output_text))
```


This returns the following output:


![](./images/B12365_04_05.jpg)

Figure 4.5 -- Replacing numbers with text

This shows that we have successfully converted our digits into text.

However, in a similar fashion to processing email
addresses, processing phone numbers may not require the same
representation as regular numbers. This is illustrated in the following
example:

```
input_text = ["0800118118"]
output_text = [to_digit(x) for x in input_text]
print('Input: ' + str(input_text))
print('Output: ' + str(output_text))
```


This returns the following output:


![Figure 4.6 -- Converting a phone number into text
](./images/B12365_04_06.jpg)

Figure 4.6 -- Converting a phone number into text

Clearly, the input in the preceding example is a
phone number, so the full text representation is not necessarily fit for
purpose. In instances like this, it may be preferable to drop any long
numbers from our input text.


Stemming and lemmatization
==========================


In language, **inflection** is how different grammatical categories such
as tense, mood, or gender can be expressed by modifying a common root
word. This often involves changing the prefix or
suffix of a word but can also involve modifying
the entire word. For example, we can make
modifications to a verb to change its tense:

*Run -\> Runs (Add \"s\" suffix to make it present tense)*

*Run -\> Ran (Modify middle letter to \"a\" to make it past tense)*

But in some cases, the whole word changes:

*To be -\> Is (Present tense)*

*To be -\> Was (Past tense)*

*To be -\> Will be (Future tense -- addition of modal)*

There can be lexical variations on nouns too:

*Cat -\> Cats (Plural)*

*Cat -\> Cat\'s (Possessive)*

*Cat -\> Cats\' (Plural possessive)*

All these words relate back to the root word cat. We can calculate the
root of all the words in the sentence to reduce the whole sentence to
its lexical roots:

*\"His cats\' fur are different colors\" -\> \"He cat fur be different
color\"*

Stemming and lemmatization is the process by which we arrive at these
root words. **Stemming** is an algorithmic process in which the ends of
words are cut off to arrive at a common root, whereas lemmatization uses
a true vocabulary and structural analysis of the word itself to arrive
at the true roots, or **lemmas**, of the word. We will cover both of
these methodologies in detail in the following
sections.



Stemming
--------

**Stemming** is the algorithmic process by which
we trim the ends off words in order to arrive at their lexical roots, or
**stems**. To do this, we can use different **stemmers** that each
follow a particular algorithm in order to return
the stem of a word. In English, one of the most common stemmers is the
Porter Stemmer.

The **Porter Stemmer** is an algorithm with a
large number of logical rules that can be used to return the stem of a
word. We will first show how to implement a Porter Stemmer in Python
using NLTK before moving on and discussing the algorithm in more detail:

1.  First, we create an instance of the Porter Stemmer:
    ```
    porter = PorterStemmer()
    ```
    

2.  We then simply call this instance of the stemmer on individual words
    and print the results. Here, we can see an example of the stems
    returned by the Porter Stemmer:

    ```
    word_list = ["see","saw","cat", "cats", "stem", "stemming","lemma","lemmatization","known","knowing","time", "timing","football", "footballers"]
    for word in word_list:
        print(word + ' -> ' + porter.stem(word))
    ```
    

    This results in the following output:

    
    ![Figure 4.7 -- Returning the stems of words
    ](./images/B12365_04_07.jpg)
    

    Figure 4.7 -- Returning the stems of words

3.  We can also apply stemming to an entire
    sentence, first by tokenizing the sentence and then by stemming each
    term individually:
    ```
    def SentenceStemmer(sentence):
        tokens=word_tokenize(sentence)
        stems=[porter.stem(word) for word in tokens]
        return " ".join(stems)
    SentenceStemmer('The cats and dogs are running')
    ```
    

This returns the following output:


![Figure 4.8 -- Applying stemming to a sentence
](data:application/octet-stream;base64,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)

Figure 4.8 -- Applying stemming to a sentence

Here, we can see how different words are stemmed using the Porter
Stemmer. Some words, such as `stemming` and
`timing`, reduce to their expected stems of `stem`
and `time`. However, some words, such as `saw`,
don\'t reduce to their logical stem (`see`). This illustrates
the limitations of the Porter Stemmer. Since stemming applies a series
of logical rules to the word, it is very difficult to define a set of
rules that will correctly stem all words. This is especially true in the
cases of words in English where the word changes completely, depending
on the tense (is/was/be). This is because there are no generic rules
that can be applied to these words to transform them all into the same
root stem.

We can examine some of the rules the Porter
Stemmer applies in more detail to understand exactly how the
transformation into the stem occurs. While the actual Porter algorithm
has many detailed steps, here, we will simplify some of the rules for
ease of understanding:


![Figure 4.9 -- Rules of the Porter Stemmer algorithm
](./images/B12365_04_09.jpg)

Figure 4.9 -- Rules of the Porter Stemmer algorithm

While it is not essential to understand every rule within the Porter
Stemmer, it is key that we understand its limitations. While the Porter
Stemmer has been shown to work well across a corpus, there will always
be words that it cannot reduce to their true stems correctly. Since the
rule set of the Porter Stemmer relies on the conventions of English word
structure, there will always be words that do not fall within the
conventional word structure and are not correctly transformed by these
rules. Fortunately, some of these limitations can be overcome through
the use of lemmatization.



Lemmatization
-------------

**Lemmatization** differs from stemming in that it reduces words to
their **lemma** instead of their stem. While the stem of a word is
processed and reduced to a string, a word\'s lemma is its
true lexical root. So, while the stem of the word
`ran` will just be *ran*, its lemma is the true lexical root
of the word, which would be `run`.

The lemmatization process uses both inbuilt pre-computed lemmas and
associated words, as well as the context of the word within the sentence
to determine the correct lemma for a given word. In this
example, we will look at using the **WordNet**
**Lemmatizer** within NLTK. WordNet is a large database of English words
and their lexical relationships to one another. It contains one of the
most robust and comprehensive mappings of the English language,
specifically with regard to words\' relationships to their lemmas.

We will first create an instance of our lemmatizer and call it on a
selection of words:

```
wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize('horses'))
print(wordnet_lemmatizer.lemmatize('wolves'))
print(wordnet_lemmatizer.lemmatize('mice'))
print(wordnet_lemmatizer.lemmatize('cacti'))
```


This results in the following output:


![Figure 4.10 -- Lemmatization output
](data:application/octet-stream;base64,/9j/4AAQSkZJRgABAgEBAwEDAAD/7QAsUGhvdG9zaG9wIDMuMAA4QklNA+0AAAAAABABAwAAAAEAAQEDAAAAAQAB/+4AE0Fkb2JlAGSAAAAAAQUAAklE/9sAhAACAgICAgICAgICAwICAgMEAwMDAwQFBAQEBAQFBQUFBQUFBQUFBwgICAcFCQoKCgoJDAwMDAwMDAwMDAwMDAwMAQMCAgMDAwcFBQcNCwkLDQ8NDQ0NDw8MDAwMDA8PDAwMDAwMDwwODg4ODgwRERERERERERERERERERERERERERH/wAARCACKAHYDAREAAhEBAxEB/8QBogAAAAcBAQEBAQAAAAAAAAAABAUDAgYBAAcICQoLAQACAgMBAQEBAQAAAAAAAAABAAIDBAUGBwgJCgsQAAIBAwMCBAIGBwMEAgYCcwECAxEEAAUhEjFBUQYTYSJxgRQykaEHFbFCI8FS0eEzFmLwJHKC8SVDNFOSorJjc8I1RCeTo7M2F1RkdMPS4ggmgwkKGBmElEVGpLRW01UoGvLj88TU5PRldYWVpbXF1eX1ZnaGlqa2xtbm9jdHV2d3h5ent8fX5/c4SFhoeIiYqLjI2Oj4KTlJWWl5iZmpucnZ6fkqOkpaanqKmqq6ytrq+hEAAgIBAgMFBQQFBgQIAwNtAQACEQMEIRIxQQVRE2EiBnGBkTKhsfAUwdHhI0IVUmJy8TMkNEOCFpJTJaJjssIHc9I14kSDF1STCAkKGBkmNkUaJ2R0VTfyo7PDKCnT4/OElKS0xNTk9GV1hZWltcXV5fVGVmZ2hpamtsbW5vZHV2d3h5ent8fX5/c4SFhoeIiYqLjI2Oj4OUlZaXmJmam5ydnp+So6SlpqeoqaqrrK2ur6/9oADAMBAAIRAxEAPwD7+Yq8z80/nP8AlH5H816D5F84/mX5a8r+c/NHofojQ9T1O2tdQvRdTtbW5gt5JFkYSTKUQgfEwKipBxV6ZirsVdirsVdirsVdirsVdirsVdirsVdirsVfkp/z9Xbyf5J0j/nGX85dS8r297rvk/8AOLy4bnU7Kzgk1qXR7GPUNRl0+C4f03ZGeHksRkCGSh2O+Ksstf8An4D+ZvlTz1+XGn/n5/zid5g/JT8s/wA3NYg0Ty75ovdYhvriG5u2C2w1GwS0iaAsHDPGzq8ahiFk4nFX1d/zk1/zk/5A/wCcXfJmneZvOFtf6/rfmW+Gk+WfLOjIs2qazqDAERQRkiiLyXm+/HkoAZ2RGVfJ+tf852f85EflppNr+YH54/8AODHmbyH+UMzwte69pfmWx1rUNKt5qATXmmR2sEkYqwB9ZoeJPEnkQCq+vvOv51a/J+TWhfm3/wA4+flw/wDzkQfM6WV3pWk6fq9tovr2F2rM1ybm7ilCmKgVo+BcMeJAKtRV8JeRP+fpGq/m75Z0+1/J/wD5xe81fmN+cEhu31jyppF+ZNN0K2imeK2mv9bk02FAZwvJVEAH7JflSqr638kf85Q6kf8AnHHzb+fv56flF5h/ImXyJ+kf0t5b1VXmvJlseCxS2XqwWbulzJIEiLog5V+IpSRlXzRc/wDOc/8AzlVD+Xa/ndB/zgdqN9+T15pw1mzvofOdlJqraVIgmg1CXTodPnuFiaJg7fujxWrFuFGKr7H/ACj/AOch9A8/f842+Wv+ckPOFpD+Xnl3UvLs/mLVobm5NzFp8Fo0yzH1vShMg/ckrRAWqAATir5I0b/nOf8A5yM/NHS7vz/+QP8Azg9r35gflBbyzmy1/WfM+n6DfatBByV5LPTZYJpGHJSB6RnqRx+1VQq+sf8AnGT/AJyc8i/85ReRrzzZ5Ss7/wAvax5ev30jzL5a1hFj1LR9RjFWhnRSao25R6DlQghXV0VV9H4q7FXYq7FX5Qf8/PvP3kHR7j/nFLy3r3mzRrHWNL/Oryr5p1LTLu6hW4g0CA38M2ozwM3JbYOrKZCONQRXY4qhP+fpP5kfllf/AJE/kyIfOegX91rv5geV/Nug+leQTSXWjxpc+pqlpxdi1uI7gVlX4aP13xVKf+c7/Muj6P8Amj/zhD/zl7blfzC/IHyBr94Na1TRSupWdlHqjWZtNSj9AuGFYGYMNvUijSvN1BVe4f8AOSv/ADnF/wA4p23/ADj1+Ycmm/mt5V/Mi+86+V9Q0fSfLGi30OoahfXOqWktvDDc2UTNNAlZf3hmVOIBH26KVVX/AJw71PRP+cVf+cLPyO0r/nIjzjpX5YX11FfTRxeZr2HT2ibVb++1O2s6Tuh9VbecM8f2kPJSBxxV8zf8+yPz+/5xs/K//nG+PRvN/wCankjyL501nzZrN5qFtqeo2dhfXCvOEtJZ/VdHYekqhC3Rem2Kv0Q/5zB8x+S/Kn/OMv5ya5+YflWfzx5Kg0B4NV0S2ufqc15DeSxWwWO5CsYmVpgwcAlSKjfFX4++XfL/AOS2mflQPPH5F/8AP03zn+WOh6TozXOi+RfNXmKKS506WCEsmmy6Mt3Zu1HT0wIbVgwFYxItKqvpPT9S/Oz/AJyq/wCfTvmWbVNOluPzM1zRLj6p+j7OOyk1mx0HWY5w0NpCkcfK4tbJo+MagSN9hfiAxV89f846/wDKjfMP5H+RtQv/APn5f+ZX5P6joWhWlhqvku8842OinR7qzgSKe0sLG5iSSS3RkIhMYaqcR9qqhV9k/wDPujQf+cdrLXP+cifMH5D/AJsefPzkvvMWr6TJ5u8x+b7Voba7vwdTnWazneytGmldrmVp2arbxtQBgWVfp/irsVdirsVeaed/yW/J38zL+01X8x/ym8m/mBqlhb/VLW88yaDp+q3EFvzZ/SjlvLeZlTk5PEGlSTiqH8y/kZ+SXnO08v2HnD8nfJHmyx8p2S6dodvrPl7Tb+LTLNVRVtrJLi2kWGICNQEQBaAbbDFWZWXlLyppvlmHyXp3ljSbDydb2X6Ni0K2soItMSzKlPqy2aRiERcTTgF402piry3yz/zjF/zjj5M8wQea/Kf5E+QvLnmS0lE9rqOn+X9Pt7i1kXYPbOluDCfePjir0Hzt+XP5ffmVp1to/wCY3kTy95/0myuBd29l5k0u01W2iuArIJo4ryGZFcK7DkBWhI74q81i/wCcU/8AnF2GSOaH/nG38rIpomDo6eTtEVlZTUMpFiCCCNjir1/zN5X8uedNA1Tyr5u0Ox8y+W9bgNtf6ZqUCXNpcxEg8JYpAysKgHcdRXFXic//ADiL/wA4r3N9DqM3/OOf5btdwceDDyzpir8AAXki2wRqAdwcVfQVra21lbW9nZW8VpZ2kawwQQosccUaAKiIigBVAFAANsVeIeZv+cXv+cb/ADn5guPNXmz8iPIXmHzHeyme71G/8v6fNcXUrAAyXLtATM1AN5ORxV7DomhaH5a0u00Ty5o1j5f0WwQR2thptvFaWsKDoscMKoij2AxVNcVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVfInmb/nPD/nFDyf+Zd1+UHmP82YtO/MOx1aLQ7jS/wBD61Ksd/OyIkLXcOmyW1eUgBPq8R3I3xV61+dX5+/lJ/zjv5a0/wA4fnH5uXyd5d1TUU0m0ujZX1+ZbySKWZYlisLa6l+xC5rxpt1xVIvK3/OUv5Aecfyo1D88dE/M3TF/KrS7meyuvMOpR3OlQR3NuVDwmLUILWbmS6hFCVckBORIxV4v5W/5+Tf84V+cPMdv5X0r87bO21G9mW3tptW0zVdLsZXbYf6bfWNvBGK7VkdQe1cVfV35h+TLf8y/InmHyW3mjXvKVt5osvq36c8qX36P1a1RmVxLZXYSUIx40rxIIJFKHFX5kf8AONvk/wAx/kv/AM/C/P8A+SUf5x/mL+Znki1/JX/EcUPnvXpdXdL+fWdIi9RU4QwgojMqsI+QDMK74q/Uzzd5u8s+QvLOtecvOWtW3l3yv5dtmvNS1K8bjDbwJSruQCepAAAqTsMVfhB59/5yg8if85Jf85Kfmno/mz/nNbzB+Q3/ADjr5FsNMtvJv+Br240mTzFdXECNeXE1zHA7SCOUSArIhFDGEpRmZV+tH/OI+ieW9F/JfSX8m/np5h/5yH8p6vqF9e6X5t8y3o1C79IS/V3s0nMcbmOGW3cUerBiw2XiqqvprFXYq7FXYq7FXYq/Kf8A5ywk8uf84+f85of84z/85Pa5Faaf5G86WGq/l153vpok9KGRYZLzSbl2bYSGV6lzuEgpUbEKvgi8/wCcvfzo1r8jP+cj/IP5y3La55x/5yN0TRvMn5P6NeenJImk+cdUuNIubSz5RrQR2x5QqvSjGo5EhV9H/wDOVn5YeWPInnX/AJ9of843+ep4ofyD0u7fTtcjkf0rDVdZ0+LTYEa/+wn76WYgsabTzdN8Vfof/wA5c/lv+TOrf84vfmnpP5geX9E0zyb5V8p6hdaZL9VggGj3VrayfUZdPoqiKVZeCxqlOZPp0IYqVWIf8+39b81eYP8AnCz8jtQ84S3FxqcenahZW010zNLJp1lqd7baeTy34rbRRqnigU98VeW+V/8A5LT+Y/8A7TvB/wB13R8VfpNqGnafq1jd6Zqtjb6npt/E0FzaXcSTQTROKMkkcgZWUjqCKYq/Jn/nHf8AK38sdR/5z6/5zp0DUPy58r32haFa+SjpmnXGkWUtpZGfTEeU20DwFI+bbtxAqdzir9YNH0XR/L2m2ujaBpNnoej2Klbax0+CO2toVZixEcMSoigsxJoOpxVMsVdirsVdirsVdir56/5yd/5xw8m/85T/AJV3n5Wedb690iyfUbPVrLU9OERurK8s2YLJEJkkT4opJI2qPsuab4q8584f84Ofk95v8+f842+epzf6dN/zjNY2OmaDYW/om3vrPSfRfTIrwtEWpbTQ814EV5MD12Ve0/nj+Q35Yf8AORfkW5/Lz81vLy69oMsy3lrJHI0F3Y3kaOkd3aTpRo5FWRhXdWBKurKSpVfGkP8Az7D/AC01NtH0r8xfz1/On82vy/8AL88M1h5L80+ajPowEClUjkihtoWCgGg9ExEDatMVfZ/5i/kzoHn78q3/ACi03zB5i/Kvy4lvY2dleeQL5dE1HT7XTniaG3s5lhmRI+MIjZOBBSo98VfEcf8Az6z/ACyh8xz+cYv+cj/+cg4vN11ZDTZtbTzhZLqUlkHWQWz3Y0X1TFyRW4FuNQDTbFX2Don5CWGifkZd/kSn5nfmHqNhd2N7Ynzjfa76vm+Nb2eScyR6oLdQrxmTjH+6oqAKQR1VfHdj/wA+r/yu0vWtY8yab/zkX/zkDp3mLzCIhquqW3m+yhvb4QLxiFzcJoiyS8FFF5saDpir9Afyx8hW/wCV/kPy35CtPM3mHzjb+W7d7dNZ81336S1i7DyyS8ru79OLmw9TitFAChVAoMVZ5irsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdiqySSOJHlldYoo1LO7kKqqNySTsBiqFOpacsVrO1/bLBfMq20hlQJMziqiNq0YkdKYqjcVUI7q2llmgiuIpZ7YgSxo6s8ZYVHJQaivviqvirsVdiqEF/YtdNYrewG9RebW4kX1QviUryp70xVERyRyxpLE6yxSKGR0IZWU7ggjYg4qoTX1lbTW9vcXkEFxdsVgikkVXlYbkIpILH5YqisVdirsVdirsVfiR/zlD/ziv8AlN+Yf/Pxf8kvLutWmq2+l/nd5c13WPOsdnqV1E2pSaNZymCEMZGMMTC0jDrHxBANKMeWKvlPyx/zhr5B8weXv+fhtt5g1vzBquhf84mjzHbflXYyancejpEttFqmqu7REmNvUNpAj0AD/vHK8ypVV9I+e/zZ/Nv81fyf/wCfdn/OOeief9U8m3f/ADknoUA87ebrWZ49Un03Sre2hlhiuuQJedDI0u9XYIrEq7hlXtf5u/8APs78ivy+/KrzH54/Ia78y/lV+cf5caLd6/o/m+18wag1zc3em273BS9Es5iVZvTIZoUi4k8t1BRlXz//AM5H69Zf85Tf8+qfKv8Azkd+Ytg9z+aXkJLaOyv7eaa2hF+3maz8v3901tFIkMhuIIefxIQjMfT49MVfXH/Ob/5Gfl1+bH/OHs/5g+d9OvNS80/lB+W17rXlm5jv7mCOC8m062keSaGOVY5qtbJ/eKe9Opqqzjy9bfmlrn/PtryTa/lFeXK/mnqH5HaCmhTQyBbt7ltGtAywSyMoWZ4+SxuWHFypqCK4q/Gc6N/zhHp35VW3k786/wDnHj8+vyi/PS201P8AEf5iXNhe3Ho648a/Wr54brVYkeB5asYzaKShorBzzxV+qGk/mlpf/OOn/PsC18//AJQ+ebX8zo/I3lBLDQfMkVnLDFJf3+qDTkleyuCZI/qs95Ro5Onp0YU2xV+b/wCWnlr/AJxU83/l7YeZv+chfyC/5yZ/PH85PPNhHqmteeE0fV5IhcXkYdTpckGqxRTQRqw9KSaOTmBy4hCEVV+i3/PsTzj+cd95c/N78ufzG0rz43kf8u9bth+XevfmFpd1p+r3miXrXypaTNcVV2t0tImKI7iL1eAbgIwFX6k4q7FXYq7FX5n/AJvf849/85i+bP8AnL7yN/zkD5I8xflHa+Ufyxim0ny9Zay2trqD6TqkHpal9ehgs5Ynn/fS+kUnRfscgPixVd5f/wCcSvz40fzn/wA5wWL+aPI3/Kp/+cqtM8yvphU6jJr1jrGrWktpZPcobWK3W3jS6l9UI8jMQhWm4xVhFt/zgR+cHmr/AJxx/KLyR54/MTyz5L/Pv/nGzWHuvyv84eS1vrmzttPSK0aO21L67Fau7yT2/J3SIBeEdFceojqo7zj+T/8Az8w/O3ypc/lB+Zf5k/k7+X/kLXIRpvmXzN5Ot9WuNc1PT5EKXEccNzHFD+9Bo6r9WqCVrxJUqvpzzl/ziD5J1n/nEDUf+cRfLF9LoHlkeX4dM03Up1E8qX9rdR6lFfXCDgHMl9F6soXjXkwXjtRV8Y+Yv+cYf+fj35m/klef849+e/zi/KnQfJOn6FHpMWqaPDqk+r68ljGgs7LULh7SFYoJDEgmmjj9Sg3jlDMrKveLT/nHb/nJC9/5wP8AMX/ONnmvzN5Li/M6y0K18seVtS8s3GpW+mrpGmLp6WiXl1PbpOZ3W2kWR0hCEMo4/aOKvO9O/L//AJ+o6T+Xq/lRd+bvyJ86WFzpLaNJ5t1v9NT6strLC0LfWB9UigmkVG4hntn50rJzZicVfQf5R/8AOGXlLyJ/ziJL/wA4o+bNZm82aTrunajDruqQqbZmvNSme5aa0Ri/AQSFDFyrUoGYbkYq+fPI/wCUf/Py78ivK1j+UP5a+ffyZ/Mb8vfLkP6N8s+YPOUGsW+s2GmxKFtoZYbQGICJfhRSbjioC8uIChV9d/8AONv5Zfn35BsvNmqf85Bfnofzi81+bp7WeKzstOh0/R9CW3WUPDYLHHE7iQyjkzImyL8HLkzKvprFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYqwL8vfzP8AIX5raZq+s/l75mtvNGmaBrV75e1C4tVkVYNT09lW5t29VIySnNTUAqQQVJBBxVnuKuxV2KuxV2KuxV2Kvzz/AOcwv+ckfze8rfmN+Vv/ADjB/wA406dpk356/nBbT6kusa2oksNB0eD1w160ZWRWb/RZmq8bqqxkCOR2VcVfL/5+6T/z9L/5x4/L7UfPXlz/AJyBs/zys5fTi1a00zyjpo1TR2mkWl3ZWosJPWjB+BtjxVuXo0BdFX6LfnX5p/NDS/8AnG+88z+TvOvk78rPzIfR9KuJfMH5gv8AUtE0uSY25vnuC0UwSRUaQRq0bL6nFWWmKvx482/85hfnJ+RdpoX5jaZ/zn55J/5yin0/WLSLzT+XMHl+y01bmwmYrM2nXsMCu5Wo3QRcftEOqlCq/VT/AJzN/wCcoL//AJxy/KTRPMPkjy/H5w/Mj8ytZs/LPkjSrgMbebUb9WdJrhEeJ2jRV+wrAs7IlVDFlVfKvmL8t/8An6v5R8k3n5nWf/OR/lvzv50sLN9QvPy4tvLem/VZEVS7WdndixiMsyKTxACF2Xj6j1+JV7d+TnnL/nM38yf+cMvyq84aA/lSz/5yL1m7vf0+/wCZum3umWj6db6lqlvHJ9T0q3gaOd4YrZk/dBChZqVZTir4L/5wRtv+c9X/AC9/NU/khqH5LQ6APzV8xDWh5yXXDdHXhFYfXTafUYmT6qR6fp8vjryr2xV+r/5l2/8AzkKf+cZ9UuLPz/5d/Lv8+dG0AapqevaFpf6W0ZbqwX6xeRWVrqoY8JUjZFaVSVry44q8O/IHzX/zkb/zkr/zg3+WHnLy9+cVn5C/OzzS15Le+cLjy/p+oRvDYazqFoyfo4xRWqs8NugLCOgIJC1NQqyj/n3t+b/5l/nZ/wA49r5t/NnzFH5q842PmnWtFm1OOztbD14bGZFiJgs4beEEBiKhBtStTvir4I8gf85J/nT/AM5G3v5heatV/wCc8/I//OH1/ofmXUNH8v8A5Z6tpWhSXUMVoyei99c6xPbSysxYxvwEi8lLBV/usVffn/ODuq/85O6v5S/MW4/5yY80aV52vk80vH5V1/Q30qXTNQ0gW0P7+xk0mKFGgaSpQyIJK8g4BFAq+3sVdir8yf8AnMb8rfzj8ofn3+T3/OaP5GeTJPzR1r8stIufLPmzyXaMVv8AUdDmN4wezVVkeSRfr8w4ojuG9NgkihwFUp1f/nOL/nIv8zzZeTv+cbv+cN/P2k+d9QlWG81z81NLl0by/ow5KsskrJKglKhieJmjfaqxyH4MVXf8/KPyz/NHzt5H/IDzPpnkC4/OPy3+WHnO0138wvIuhLO51m1VIRIYbdOUzxgJNFsGdVm5FSochV8bf85Rav53/wCclfyJ1Hyf/wA4+f8APvTzD+X3l3y99Q1DUNa1byra6Xq0ccFxEi2Xl3T7W3E05LEeo0JYiJXVowGqFX2L/wA5Dfl9+Zv/ADlt/wA40flP+ZX5Y/l75i/Lb85/yW80Wnmny/5S8+2K6TqV3NpDejJA0U0yiMSmNJoS7KH4BTx58gqtk/5+D/nTqGhReXfL3/OCX5ut+dc0SwPpmp6VPbeXLe6b4TM+ptEshgDAsC0UYYbeoteeKv0C/Jm4/Ni7/LHyjd/nlZaHpv5q3Vq82v2flz1P0bbSyTStDDEZJrglkgMayESMpkDFCVpir8pPyX/Mr83f+cGrv89vyn81f84ufmf+akXmf8ytY84eUde8i6S+qaXqFpq0duiRS3MSt6bBLVWKhXcFmVkXjuq/Qb8ofNn5zfnr+Vfn5vzi/Js/kNeeZFvtK0HSrrU01G9fTruz9Nbm7VIojE4kkYcWVW2rwAoWVfnd/wA4v/nr/wA5C/8AONn5Mad/zi5qH/OHH5l+bfzZ8h3eraf5f1GxsOPlW/F9f3V7DcXesOY4YoEkuypkUshVQS6sTRV7P/z6u/5WN5R/KXzV+Un5n/lD55/L7zL5a8wX2ty6t5l0eXTdM1L9Ky7x2M0wjMroYSX4JwoykMa4q+OPJn5gaB510i71n/nNv/n295483/mLd3l2w86eS/IU1s2sEyPVdQjjfS/30ZHpiT1JOYoTxoSyr6+/59nflD54/L2+/wCcivNs/wCXfmD8mfyb/MjzFZXvkDyL5ollbU7G3t/rgmuJredjJDzjlhT46u4QAs4jV3VfqtirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdir/AP/Z)

Figure 4.10 -- Lemmatization output

Here, we can already begin to see the advantages of using lemmatization
over stemming. Since the WordNet Lemmatizer is built on a database of
all the words in the English language, it knows that `mice` is
the plural version of `mouse`. We would not have been able to
reach this same root using stemming. Although lemmatization works better
in the majority of cases, because it relies on a built-in index of
words, it is not able to generalize to new or made-up words:

```
print(wordnet_lemmatizer.lemmatize('madeupwords'))
print(porter.stem('madeupwords'))
```


This results in the following output:


![Figure 4.11 -- Lemmatization output for made-up words
](data:application/octet-stream;base64,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)

Figure 4.11 -- Lemmatization output for made-up words

Here, we can see that, in this instance, our stemmer is able to
generalize better to previously unseen words. Therefore, using a
lemmatizer may be a problem if we\'re lemmatizing
sources where language doesn\'t necessarily match
up with *real* English language, such as social media sites where people
may frequently abbreviate language.

If we call our lemmatizer on two verbs, we will see that this doesn\'t
reduce them to their expected common lemma:

```
print(wordnet_lemmatizer.lemmatize('run'))
print(wordnet_lemmatizer.lemmatize('ran'))
```


This results in the following output:


![Figure 4.12 -- Running lemmatization on verbs
](data:application/octet-stream;base64,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)

Figure 4.12 -- Running lemmatization on verbs

This is because our lemmatizer relies on the context of words to be able
to return the lemmas. Recall from our POS analysis that we can easily
return the context of a word in a sentence and determine whether a given
word is a noun, verb, or adjective. For now, let\'s manually specify
that our words are verbs. We can see that this now correctly returns the
lemma:

```
print(wordnet_lemmatizer.lemmatize('ran', pos='v'))
print(wordnet_lemmatizer.lemmatize('run', pos='v'))
```


This results in the following output:


![Figure 4.13 -- Implementing POS in the function
](data:application/octet-stream;base64,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)

Figure 4.13 -- Implementing POS in the function

This means that in order to return the correct
lemmatization of any given sentence, we must first perform POS tagging
to obtain the context of the words in the sentence, then pass this
through the lemmatizer to obtain the lemmas of each of the words in the
sentence. We first create a function that will return our POS tagging
for each word in the sentence:

```
sentence = 'The cats and dogs are running'
def return_word_pos_tuples(sentence):
    return nltk.pos_tag(nltk.word_tokenize(sentence))
return_word_pos_tuples(sentence)
```


This results in the following output:


![Figure 4.14 -- Output of POS tagging on a sentence
](./images/B12365_04_14.jpg)

Figure 4.14 -- Output of POS tagging on a sentence

Note how this returns the NLTK POS tags for each of the words in the
sentence. Our WordNet lemmatizer requires a slightly different input for
POS. This means that we first create a function
that maps the NLTK POS tags to the required WordNet POS tags:

```
def get_pos_wordnet(pos_tag):
    pos_dict = {"N": wordnet.NOUN,
                "V": wordnet.VERB,
                "J": wordnet.ADJ,
                "R": wordnet.ADV}
    return pos_dict.get(pos_tag[0].upper(), wordnet.NOUN)
get_pos_wordnet('VBG')
```


This results in the following output:


![Figure 4.15 -- Mapping NTLK POS tags to WordNet POS tags
](data:application/octet-stream;base64,/9j/4AAQSkZJRgABAgEAnACcAAD/7QAsUGhvdG9zaG9wIDMuMAA4QklNA+0AAAAAABAAnAAAAAEAAQCcAAAAAQAB/+4AE0Fkb2JlAGSAAAAAAQUAAklE/9sAhAACAgICAgICAgICAwICAgMEAwMDAwQFBAQEBAQFBQUFBQUFBQUFBwgICAcFCQoKCgoJDAwMDAwMDAwMDAwMDAwMAQMCAgMDAwcFBQcNCwkLDQ8NDQ0NDw8MDAwMDA8PDAwMDAwMDwwODg4ODgwRERERERERERERERERERERERERERH/wAARCAAmADcDAREAAhEBAxEB/8QBogAAAAcBAQEBAQAAAAAAAAAABAUDAgYBAAcICQoLAQACAgMBAQEBAQAAAAAAAAABAAIDBAUGBwgJCgsQAAIBAwMCBAIGBwMEAgYCcwECAxEEAAUhEjFBUQYTYSJxgRQykaEHFbFCI8FS0eEzFmLwJHKC8SVDNFOSorJjc8I1RCeTo7M2F1RkdMPS4ggmgwkKGBmElEVGpLRW01UoGvLj88TU5PRldYWVpbXF1eX1ZnaGlqa2xtbm9jdHV2d3h5ent8fX5/c4SFhoeIiYqLjI2Oj4KTlJWWl5iZmpucnZ6fkqOkpaanqKmqq6ytrq+hEAAgIBAgMFBQQFBgQIAwNtAQACEQMEIRIxQQVRE2EiBnGBkTKhsfAUwdHhI0IVUmJy8TMkNEOCFpJTJaJjssIHc9I14kSDF1STCAkKGBkmNkUaJ2R0VTfyo7PDKCnT4/OElKS0xNTk9GV1hZWltcXV5fVGVmZ2hpamtsbW5vZHV2d3h5ent8fX5/c4SFhoeIiYqLjI2Oj4OUlZaXmJmam5ydnp+So6SlpqeoqaqrrK2ur6/9oADAMBAAIRAxEAPwD7+Yq0GUkqGBZaVFdxXpXFW8VdirsVa5KCFLAMdwK7mmKt4q7FX55/kV+VXm3y5/znx/zmh+Ymp+XdU0/yj5p0nybHoWsXNvLHY6k0unQm8W1nZRHKYJrUo4UkpUcqchVV9H/nl/zlB+RX/ONkflqT86vPaeSx5wa7XSAdO1LUXuTYiA3JCadZ3bKE+sx1LAD4hTFWM6h/zml/zjNpX5P6L+fd/wDmdFB+U/mLV20LTtcGlas5m1FPrHK3+pJYNdqw+qyGrQgUWtaEVVZf+R//ADkt+SX/ADkhYeYNS/JfzunnOz8rTwW2qMLDUNPa3kuVd4QY9RtLR2DCNqFQRsRWuKvmf82/ym82a7/z8T/5xT/NSw8v6nd+T/KXk3zVbatq8NvK9hZzi0v4baK4uFUxxvK2qDgrEF+JpUKaKv0MxV2KuxV8/wD/ADlD+Rmkf85F/kb+YH5V6jFAuoa9pkjaJezKCbLVrciexnDfaVRPEgk4kckLL0OKvyQm/wCcm9b81f8APvTyb+QemabE3/ORvm3zEP8AnHs6FcRqJ7e6tGjtrq7kjIqP9AmijeUfYnlLVPFsVfsz+SH5SeXPyL/KnyN+VfleGNdP8naTb2Elysaxve3SLW5vJQoH7yeZnkb3bwxV6tirsVdirsVdir4r0T/nBT8ndC/5yr1f/nLG0e/bzbqYnuo9Eb0f0VbatdwfVrnUo1EYkMsiF2IZiBI7v148VX2pirsVdirsVdirsVdirsVdirsVf//Z)

Figure 4.15 -- Mapping NTLK POS tags to WordNet POS tags

Finally, we combine these functions into one final function that will
perform lemmatization on the whole sentence:

```
def lemmatize_with_pos(sentence):
    new_sentence = []
    tuples = return_word_pos_tuples(sentence)
    for tup in tuples:
        pos = get_pos_wordnet(tup[1])
        lemma = wordnet_lemmatizer.lemmatize(tup[0], pos=pos)
        new_sentence.append(lemma)
    return new_sentence
lemmatize_with_pos(sentence)
```


This results in the following output:


![Figure 4.16 -- Output of the finalized lemmatization function
](data:application/octet-stream;base64,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)

Figure 4.16 -- Output of the finalized lemmatization function

Here, we can see that, in general, lemmas generally provide a better
representation of a word\'s true root compared to stems, with some
notable exceptions. When we might decide to use
stemming and lemmatization depends on the requirements of the task at
hand, some of which we will explore now.


Uses of stemming and lemmatization
==================================


Stemming and lemmatization are both a form of NLP
that can be used to extract information from text. This
is known as **text mining**. Text mining tasks
come in a variety of categories, including text
clustering, categorization, summarizing documents, and sentiment
analysis. Stemming and lemmatization can be used in conjunction with
deep learning to solve some of these tasks, as we will see later in this
book.

By performing preprocessing using stemming and lemmatization, coupled
with the removal of stop words, we can better reduce our sentences to
understand their core meaning. By removing words that do not
significantly contribute to the meaning of the sentence
and by reducing words to their roots or lemmas, we
can efficiently analyze sentences within our deep learning frameworks.
If we are able to reduce a 10-word sentence to
five words consisting of multiple core lemmas rather than multiple
variations of similar words, this means much less data that we need to
feed through our neural networks. If we use bag-of-words
representations, our corpus will be significantly smaller as multiple
words all reduce down to the same lemmas, whereas if we calculate
embedding representations, the dimensionality required to capture the
true representations of our words will be smaller for a reduced corpus
of words.



Differences in lemmatization and stemming
-----------------------------------------

Now that we have seen both lemmatization and stemming in action, the
question still remains as to under which
circumstances we should use both of these techniques. We saw that both
techniques attempt to reduce each word to its root. In stemming, this
may just be a reduced form of the target room,
whereas in lemmatization, it reduces to a true English language word
root.

Because lemmatization requires cross-referencing the target word within
the WordNet corpus, as well as performing part-of-speech analysis to
determine the form of the lemma, this may take a significant amount of
processing time if a large number of words have to be lemmatized. This
is in contrast to stemming, which uses a detailed but relatively fast
algorithm to stem words. Ultimately, as with many problems in computing,
it is a question of trading off speed versus detail. When choosing which
of these methods to incorporate in our deep learning pipeline, the
trade-off may be between speed and accuracy. If time is of the essence,
then stemming may be the way to go. On the other hand, if you need your
model to be as detailed and as accurate as possible, then lemmatization
will likely result in the superior model.


Summary
=======


In this chapter, we have covered both stemming and lemmatization in
detail by exploring the functionality of both methods, their use cases,
and how they can be implemented. Now that we have covered all of the
fundamentals of deep learning and NLP preprocessing, we are ready to
start training our own deep learning models from scratch.

In the next chapter, we will explore the fundamentals of NLP and
demonstrate how to build the most widely used models within the field of
deep NLP: recurrent neural networks.
