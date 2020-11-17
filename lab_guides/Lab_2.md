
*[]{#_idTextAnchor029}Chapter 2*: Getting Started with PyTorch 1.x for NLP {#_idParaDest-30}
==========================================================================

::: {#_idContainer081}
**PyTorch** is a Python-based machine learning library. It consists of
two main features: its ability to efficiently perform tensor operations
with hardware acceleration (using GPUs) and its ability to build deep
neural networks. PyTorch also uses dynamic computational graphs instead
of static ones, which sets it apart from similar libraries such as
TensorFlow. By demonstrating how language can be represented using
tensors and how neural networks can be used to learn from NLP, we will
show that both these features are particularly useful for natural
language processing.

In this chapter, we will show you how to get PyTorch up and running on
your computer, as well as demonstrate some of its key functionalities.
We will then compare PyTorch to some other deep learning frameworks,
before exploring some of the NLP functionality of PyTorch, such as its
ability to perform tensor operations, and finally demonstrate how to
build a simple neural network. In summary, this chapter will cover the
following topics:

-   Installing PyTorch
-   Comparing PyTorch to other deep learning frameworks
-   NLP functionality of PyTorch


Technical requirements {#_idParaDest-31}
======================

::: {#_idContainer081}
For this chapter, Python needs to be installed. It is recommended to use
the latest version of Python (3.6 or higher). It is also recommended to
use the Anaconda package manager to install PyTorch. A CUDA-compatible
GPU is required to run tensor operations on a GPU. All the code for this
chapter can be found at
<https://github.com/PacktPublishing/Hands-On-Natural-Language-Processing-with-PyTorch-1.x>.


Installing and using PyTorch 1.x {#_idParaDest-32}
================================

::: {#_idContainer081}
Like most Python packages, PyTorch is very simple to install. There are
two main ways of doing so. The []{#_idIndexMarker064}first is to simply
install it using `pip`{.literal} in the []{#_idIndexMarker065}command
line. Simply type the following command:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
pip install torch torchvision
```
:::

While this installation method is quick, it is recommended to install
using Anaconda instead, as this includes all the required dependencies
and binaries for PyTorch to run. Furthermore, Anaconda will be required
later to enable training models on a GPU using CUDA. PyTorch can be
installed through Anaconda by entering the following in the command
line:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
conda install torch torchvision -c pytorch
```
:::

To check that PyTorch is working correctly, we can open a Jupyter
Notebook and run a few simple commands:

1.  To define a Tensor in PyTorch, we can do the following:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    import torch
    x = torch.tensor([1.,2.])
    print(x)
    ```
    :::

    This results in the following output:

    ``{.literal}

    Figure 2.1 -- Tensor output

    This shows that tensors within PyTorch are saved as their own data
    type (not dissimilar to how arrays are saved within NumPy).

2.  We can perform basic []{#_idIndexMarker066}operations such as
    multiplication []{#_idIndexMarker067}using standard Python
    operators:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    x = torch.tensor([1., 2.])
    y = torch.tensor([3., 4.])
    print(x * y)
    ```
    :::

    This results in the following output:

    ::: {#_idContainer060 .IMG---Figure}
    ![Figure 2.2 -- Tensor multiplication output
    ](3_files/B12365_02_2.jpg)
    :::

    Figure 2.2 -- Tensor multiplication output

3.  We can also select individual elements from a tensor, as follows:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    x = torch.tensor([[1., 2.],[5., 3.],[0., 4.]])
    print(x[0][1])
    ```
    :::

    This results in the following output:

<div>

::: {#_idContainer061 .IMG---Figure}
![Figure 2.3 -- Tensor selection output ](3_files/B12365_02_3.jpg)
:::

</div>

Figure 2.3 -- Tensor selection output

However, note that unlike a NumPy array, selecting an
[]{#_idIndexMarker068}individual element from a tensor object
[]{#_idIndexMarker069}returns another tensor. In order to return an
individual value from a tensor, you can use the `.item()`{.literal}
function:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
print(x[0][1].item())
```
:::

This results in the following output:

<div>

::: {#_idContainer062 .IMG---Figure}
![Figure 2.4 -- Output of the .item() function
](3_files/B12365_02_4.jpg)
:::

</div>

Figure 2.4 -- Output of the .item() function

[]{#_idTextAnchor032}

Tensors {#_idParaDest-33}
-------

Before we continue, it is important that you are fully aware of the
properties of a tensor. Tensors have a []{#_idIndexMarker070}property
known []{#_idIndexMarker071}as an **order**, which essentially
determines the dimensionality of a tensor. An order one tensor is a
tensor with a single dimension, which is equivalent to a vector or list
of numbers. An order 2 tensor is a tensor with two dimensions,
equivalent to a matrix, whereas a tensor of order 3 consists of three
dimensions. There is no limit to the maximum order a tensor can have
within PyTorch:

<div>

::: {#_idContainer063 .IMG---Figure}
![Figure 2.5 -- Tensor matrix ](3_files/B12365_02_5.jpg)
:::

</div>

Figure 2.5 -- Tensor matrix

You can check the size of []{#_idIndexMarker072}any tensor by typing the
following:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
x.shape
```
:::

This results in the following output:

<div>

::: {#_idContainer064 .IMG---Figure}
![Figure 2.6 -- Tensor shape output ](3_files/B12365_02_6.jpg)
:::

</div>

Figure 2.6 -- Tensor shape output

This shows that this is a 3x2 tensor (order 2).


Enabling PyTorch acceleration using CUDA {#_idParaDest-34}
========================================

::: {#_idContainer081}
One of the main benefits of PyTorch is its ability to enable
acceleration through the use of a **graphics processing unit**
(**GPU**). Deep learning is a computational task that is easily
parallelizable, meaning that the calculations can be broken down into
smaller []{#_idIndexMarker073}tasks and calculated across many smaller
processors. This []{#_idIndexMarker074}means that instead of needing to
execute the task on a single CPU, it is more efficient to perform the
calculation on a GPU.

GPUs were originally created to efficiently render graphics, but since
deep learning has grown in []{#_idIndexMarker075}popularity, GPUs have
been frequently used for their ability to perform multiple calculations
simultaneously. While a traditional CPU may consist of around four or
eight cores, a GPU consists of hundreds of smaller cores. Because
calculations can be executed across all these cores simultaneously, GPUs
can rapidly reduce the time taken to perform deep learning tasks.

Consider a single pass within a neural network. We may take a small
batch of data, pass it through our network to obtain our loss, and then
backpropagate, adjusting our parameters according to the gradients. If
we have many batches of data to do this over, on a traditional CPU, we
must wait until batch 1 has completed before we can compute this for
batch 2:

<div>

::: {#_idContainer065 .IMG---Figure}
![Figure 2.7 -- One pass in a neural network
](data:application/octet-stream;base64,/9j/4AAQSkZJRgABAgEBLAEsAAD/7QAsUGhvdG9zaG9wIDMuMAA4QklNA+0AAAAAABABLAAAAAEAAQEsAAAAAQAB/+4AE0Fkb2JlAGSAAAAAAQUAAklE/9sAhAACAgICAgICAgICAwICAgMEAwMDAwQFBAQEBAQFBQUFBQUFBQUFBwgICAcFCQoKCgoJDAwMDAwMDAwMDAwMDAwMAQMCAgMDAwcFBQcNCwkLDQ8NDQ0NDw8MDAwMDA8PDAwMDAwMDwwODg4ODgwRERERERERERERERERERERERERERH/wAARCAIYARADAREAAhEBAxEB/8QBogAAAAcBAQEBAQAAAAAAAAAABAUDAgYBAAcICQoLAQACAgMBAQEBAQAAAAAAAAABAAIDBAUGBwgJCgsQAAIBAwMCBAIGBwMEAgYCcwECAxEEAAUhEjFBUQYTYSJxgRQykaEHFbFCI8FS0eEzFmLwJHKC8SVDNFOSorJjc8I1RCeTo7M2F1RkdMPS4ggmgwkKGBmElEVGpLRW01UoGvLj88TU5PRldYWVpbXF1eX1ZnaGlqa2xtbm9jdHV2d3h5ent8fX5/c4SFhoeIiYqLjI2Oj4KTlJWWl5iZmpucnZ6fkqOkpaanqKmqq6ytrq+hEAAgIBAgMFBQQFBgQIAwNtAQACEQMEIRIxQQVRE2EiBnGBkTKhsfAUwdHhI0IVUmJy8TMkNEOCFpJTJaJjssIHc9I14kSDF1STCAkKGBkmNkUaJ2R0VTfyo7PDKCnT4/OElKS0xNTk9GV1hZWltcXV5fVGVmZ2hpamtsbW5vZHV2d3h5ent8fX5/c4SFhoeIiYqLjI2Oj4OUlZaXmJmam5ydnp+So6SlpqeoqaqrrK2ur6/9oADAMBAAIRAxEAPwD7+Yq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq8W/OL8/vy1/I/Sfr/nXW0TUJ1H1HR7YrJf3bsQqrHEWFASacmIX3xV6N5O8zWPnXyl5Y846ZFNBpvmvSrPV7WO4AWVIb2FJ41kAJAYK4rQ9cVZHirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVfm9+e/8AznP57/Lr/nIe4/5x1/Kz/nH7Uvzh812Wg22uTCyv1t5GjmT1XKRek/wxqy1JPU9MVSby7/z8ck8u+fvKX5e/85LfkF5r/wCce7zztMltpesasyT6W00kixKJJOEZVOTjk4qF6tQb4q/TrFXYqslkSGOSaVgkcSl3Y9AqipP3Yq8c/I/8/Py0/wCciPK+pecPyt1afWND0jVrjRLqW4tpbV0vLZY3kXhKFJHGVSD0IOKvZsVdir88rX/nIP8AM6T/AJ+N3/8Azju+qWh/K6LyAuuJYfVE9cXoiWT1PrP29y246U7Yq/Q3FXYq7FXYq7FXYqxvzfp3mDVvLOtab5V15fLHmG8tmjsNVe3W6W1lNKOYWIDbVHtWuKv5hf8AnLP/AJxu/wCcoPJX5qaR5r/MnVb3zr5cn16yuG1aJne2kUXSN6jNUjYb8WoVHRRir9ofyp/PXzz+WX5Vflgv5k/lRfH8vF8saMunebvLUo1KGLTjaW4t5tRtQqyxP6ZBcKGAIOKvtzyt5s8t+dtEsvMflPWrTX9E1BBJBd2cgkjYHehI6HxB3GKshxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV+F/wCan5yfl5+RP/P2HV/Pv5n67/h3ytD+XFvYteejLPSe5s4REnCJXbfid6YqxX/nMr/nIz8rP+c7/MP5G/8AOOf5D6zBeamPOkOvaj5q1wppOm2NvBbz25iie6aOSV3FwW4KtWKKqhmb4VX7N/nDN+bujfldqkX5E6ZpXmL8zUitrLSB5luGhs1LskUt3cuoBYxpV+O3JhTvir88PzG/L7/n5R+W/knzP+cMf/OT3lnzVq/lXTJdbvfJ0Pl2KKwkgtU9a5iglcAErGrEVVeVNiDTFXuPlT80vzk/5y2/5xC8kfmX+Sfm7Rvyj/MLXHZtbuL+xOpWQWwF1baha26MXK85lVkY1Kgceprir85P+fankH/nLHzD+Wutat+Un53eXfI3kKw/MO5TWtH1LRhf3F3cxR2L3cscpX4RJEVULUbjFX3j/wA5R/8AOS/5ySfnx5F/5xE/5xeTSrf80/M+nHWvMnmfWIvXtPL+nUaVf3bKyFzDEztyB+1EqfG+yqbflp5Q/wCc/wD8rfzM8m2/nv8AMryx+f35T+Y7xrbzHMtkulanoiupZbmAUBkRWFCvI/6o6hV8T/nt5t/Njyn/AM/SdVP5IeS7fzt+ZnmL8u7TRNIhv5GisNPa7go+pXrKP7m2VebCor06kYq9E/Nn81/+c5f+cK/M35d/mN+cn5neXPzq/KDznrsGk+ZrSx0pdPOlPcDf6sVRGWicmjYGjFeLoAQcVfs7aXMV7a215AxaC7iSaMnYlJFDKfuOKvjv/nP382vPX5If84t+fvzH/LjVU0Xzdo11o0NpePDHcCNLvU7S3m/dyhlJMcjDcbVxV8reXdU/5+O/85Jfl9Z/m75L85eWvyF0G50+C/8AKvljUNPW51HXY0gDLd31w0TrAl03xIvHiFIPEDcqvo3/AJwH/wCcnPNX/OSn5Wa9N+Y2lwaR+aH5ba5L5b8zRW0fpRSzRKCk4jBIRmKuroNgy1GxoFX3RirsVSzWNF0jzDpt3o+uabbavpd8hjuLW7jWWKRT2ZWBGKrtP0jS9J0my0HTrCCz0bTrSOxtrKNAIYraJBGkKp04hAAB4Yq+V/O/5I+Yfy81S9/ND/nG/wBLRfMXL19a8luxXRPMES7uqRVCwXBFeLrSp2OKvXvyg/ODQPze0G61DTrW50TX9Dn+oa/oGoL6d9pd8o+KGZDQ0PVGpRhuMVet4q7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FX44T6Tpms/8/idUstX0y11ayP5XI5gvIUni5LZQUbhIrCorsaYqzj/AJ+h/lH+Q1l/zjJ5r89an5c0Hyn5/wDL89kfKmr2FvDZajJfyXUINrCYRG0iyRg81oeKj1NuFQqhfNn/ADlR+aP5F/8APuL8ofzWu7NL382vM+iaPo1hc6tGzxRvdh0t9QuQQA5+qxrIORo7EFiQTVVLfOn/ADjh+dDf84/+dPzS/OT/AJzi8/jUJ/Jl/rmo6fo9zZaN5bJuLJ5UsTGYypidnEWxXnX4QKgYq9C/59WxvL/zgv5PijUvJJqHmZVUdSx1G6AAxVgv/PnWWJ/+ce/zLhVw0tt+Z2rLKoO6k2WnMAfoOKsc8sapa+Rv+fx/5nx+cbgWa/mn+XVjaeUZLsemksi2OjEpAxop5NpF0g8Wqv2sVfrrq3mLQNBm0m31vW7DSLjXrxNO0yK8uIoJL27kBK29ssjKZJCATxWpoCegxV+UFl/8mK1P3/Kkf8mExVk3/P4gA/8AOIFSOnnfRKf8i73FX6YeUP8AlE/K/wD2yLL/AJMR4q+Df+fq/wD6xF+aX/Md5e/7rNjir6y/5x33/wCcfvyMrvX8vfLH/dKtMVfnZ/z6/AX8wv8AnO5FFEX83LgBR0H+l6v0H0Yq/XXFXYq7FXYq7FXy/wDnD+XeveXtfT8+fylg4efdCtxH5h0aM8YPNGjx/FJbTKNvXjA5Qv1qKYq9r/Lvz95e/M7yfo3nTyxcmfStYh5hHHGaCVfhlgmQ7rJGwKsDirNcVdirsVdirsVdirsVdirsVdirsVdirsVfBX57/wDPvf8AK78+fzan/OjU/wAw/wAwfIfnK50q20mZ/KOp2dlE0NsrIrVn0+6kDMhCsA4U0Hw1qSqwzyt/z60/5x40vzNp3mfzz5j8+/nTNpMizWlj561mO9sUdCGXlDbWtn6i1G6OWRujKRtir7P/ADh/JP8ALz88/wAtdW/Kjz/oovPKGqRxKsNq31eS0kt6G3mtXQUjeIgcdqdiCtRir4u8t/8APsn8s7O1sNA8+fnF+aX5seQNBWRdG8m+ZdcWTQrNmQrHL9Uht4lZ4ieSDZKj4kYbYq+uf+cdfyF8rf8AONX5WaP+Uvk7VdU1rQtGur27iu9ZaB7tnv7iS5kDG2gto6BpCFAQbYq+TW/59mflXpn5lat578i/mf8AmL+W2ieZNai1zWfKHlzVY7XR7qaGYXAh4/VzIITICSjM1ASEKClFXu//ADkx/wA4dflF/wA5S2+gT+eU1XQPNnlNv9w3mny5cJZ6vZx8xIYVlkinjeMsKgOh4mpQqSTiryz8of8An3p+W/5b/mJoP5recPzF8+fnZ538otz8v3HnTVfrVtpbgFUlghjjjJdVO3NmUGhCggHFXtcX/OL/AJJi/wCcnLj/AJyoXXNbPne48uDy0dLMlt+ihbhVT1Qn1b1+dF/37T2xVH/85Nf844eT/wDnKb8tP+VXed9Z1nQtE/S1prH1rQ3t47r1rQSqiVube5TiRKa/DXpvir3jTbGLS9OsNMgZ3g062itY2kILlIUCKWIAFaLvQYq8g/5yH/Iryx/zkj+U/mL8ofOGqapoug+Y5bKae80d4UvI2sbqG7j4G4huI6FoQGqh2JpQ74q9G8leVbDyL5N8peSdLnuLnTPJ2jWGh2c10VaeSDT7eO2ieUokalysQLEKBXoBirxP8gv+cYvJX/OPGt/m/rvlLW9b1e6/ObzLJ5n1ZNXktnjtrmSS5lMVqILaAhK3TfbLGgG/Wqr6SxV2KuxV2KuxV2KvizXTL/zjL+a58128cv8AypL83tQSHXYU/ufL/mCduMd6qj7MNx0emwbfFX2ijrIqujB0cBlZTUEHcEEYquxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KsU88eTdD/MHylr/AJM8x2q3ej+YbOS0uEYAleY+GRa/tI1GX3GKvB/+cb/N+t2kXmH8j/PlyZfPv5TSJaJcSkctU0SQf7j79B1NY6Kx8R74q+o8VdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirzn82PzJ0z8o/IeseftY0281bT9GlsopLWw9L6w5vryCzQr60kSfC1wCat0Bpvtir0bFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq+T/+cjdMuPJGq+S/+ch9Bhcah+Xd0ll5mSD7V75ZvXEd0rqCOXoMwkX5eAxV9TWN9a6nZWeo2M6XNlfwx3EE0ZDK8cihkZSKggg4qisVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVfLP/ADml/wCs4+ef+YzQP+65p2KvqbFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYql2saTYa9pOp6Jqtut3pmr2s1ndQt0khnQxup+ascVfN//ONOqahoFn5v/I3zFcNNr35OagLGyll+1d6Bd8ptKuBUmoEX7s+HHfFX1DirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdir5Z/5zS/9Zx88/wDMZoH/AHXNOxV9TYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq+V/wA2/wDkHH5y/lV+cEREGieYXP5f+a3GyiG/b1dLuJKf77uVKlj0UgYq+qMVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVfLP/ADml/wCs4+ef+YzQP+65p2KvqbFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXnX5teQrX8zvy583eR7khH1ywdLSU7ejeRUltZgaGnCZEb6MVSL8hPPd1+Yn5V+WNd1RTH5itIn0nXYX+3HqmnOba6DjehZ4+dPBhir2LFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq+Wf+c0v/WcfPP/ADGaB/3XNOxV9TYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXy7+Wo/wD+f35vfly/7nSvPcdv+YmhIdl9W4P1TVkXsWM6K9B0G+KvqLFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq+Vf+c0JY2/5xy88BZFY/XNA2BB/6XenYq+pxJGW4h1LdKVFfuxVfirsVdirsVdirsVdirsVdirsVdirsVdirsVdir5c/5yIH+Dte/KL864f3cXkLzCmk65IB/wBKPX+NncM/Y+nIY2UHoTXbFX1HirsVdirsVdirsVdirsVdirsVdirsVdirsVdirRAIIIqDsQe+Kv56P+ckvK+u/l1+b3nnyZNquoy6Gb79IaZDNdTvCbC8IubZQruVPp8uFafaQ4q+tv8An3t5V1zzB5m82fmLreqahf6d5et10uwS7uZ5Y2vLr4pZFDuykpEtDt+2MVfrFirsVdirsVdirsVdirsVdirsVdirsVdirsVdirBfzN8oW3n78vvOPk26A9PzDpNzaIx/3XM8Z9GQe6SBWHuMVYb/AM46+cbnzx+TPkXWdQJGs21h+itURvtpe6azWkwk/wAomHkfnir2zFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FX5hf8/Gfy6Nxp3kj80rGDlJp8reXdUZRUmGflPZO3gFkEq18ZFGKvrn/nFv8u/+Vafkl5M0WeH0dV1S3/Tepggq31rUAsvFwejJFwQ+64q+hMVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdir5T/ACHceUPzR/P78qHIjt7XXovOWkR9OVpr0SvclB2VLhCtOlemKvqzFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FWJeePJWgfmH5avvKfma1+uaPqD28ksXQlraeO4jIPajxDFWW4q7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq+TfzEJ8j/85Pfk352B9DTPzG0y/wDIepvT7dwtb3TE7DeQP92KvrLFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXy3/AM5f6fOfygk822Hwar+W+uaT5otZx1hWyuo/rDj/AJ4PIMVfSul6jb6vpmnataNztNUtYbuFvGOdFkQ/cwxVH4q7FXYq7FXYq7FXYq/L/wDPj/nNH82db/OfUf8AnF//AJw58hWfnr80dBJHmnzNrBB0XQwqj1U+GRF5xswV2lNA9Y1jkY7KsVufOX/P1P8AJCKLzh588peQv+cjfKMTrJq2h+TVki1u2ta1kNokdlYu8gGwCRXB78D1Cr9ZLeUzwQTmJ4DNGrmOUUdOQB4sKmhFaHFVbFX5u/8AP1bzf5s8k/8AOJmp615N8zar5T1g+aNFt/r+j3k1jdelI8pdBNA8bhW4iorvir728hXNxeeRfJd3dzvc3V1oOnTTTSsWeSR7WJmdmNSSSak4qyzFXYq7FXYq/LD/AJ9jed/OfnRf+cqT5w82ax5qOj/mneWth+l76e9+qwUl/cweu7+mnwj4VoPbFX6n4q7FXyrf/wDOUuk2H/OWuif84ot5Ru5NV1ryq3mdPMIukFugVbt/q5tvS5H4bU/Fz6n7PfFX1VirsVdirsVdirsVdirsVdirsVdirCvzI8twecfy/wDOnlW6XlB5g0W9sXFK19aF16fM4q83/wCcXvMlx5n/ACG/Li9vm/3J2GmDSr6OtTFPpztatG3uBEMVe+4q7FXYq7FXYq7FUk8y6t+gPLmv676Rn/Qum3V/6agkv9WheXiANzXjTFX5Hf8APnXR7PUfy3/PP80L1Bd+b/OX5hXNrfajK5kupLa2tre4ijmZiTUS3srb/wA1cVfsZir5a/5yQ/IXz3+fE/k7R9J/PzzN+TP5eaWL2TzVY+UONpqmtmX6uLWMalzVoI41WXkvF1fkOSGgIVfl/wD85N/k/qn/ADgTZ/l/+d3/ADj/AP8AOQPn3UVt/M9lpnmXyz5q15NWs9QtLqtXa3jit1apSjB0bryQoV3VfSH/AD9m1GPWP+cIxq0KGOLVPMfly7RSalVnEsgBNB0DYq9M/wCctf8AnIvzB/zjT/zhz5a84eS41bz55j0/QvLfl2V4451tLy8sg5ujBIGWT044W4qQRzK8gVqCq8b8tf8APt/znr/k/SPO/nT/AJy6/N6z/wCcg9Qs01GXXbDW2bS9PvpwJTDFaHjO8UZbj8NzGGpUKg+HFUZ/z8V1H80/y3/5wY8lwav+YN7cfmTpmreV9N8weZdDml01tRvI43W6mX0DCQssicitAP8AJHQKv0t8rXNxP+Xnly8mnkmu5vLtnNJO7FpGka0Ri7OSSWJNScVfnJ/z6v8ANfmnzV5F/PWbzR5l1XzJLY/mTqENs+qXk940MZFSkZndyq13oNsVYv8A8+oPsf8AOXf/AJtq8/5nYq9O8x/84DXv5l675g81f85Bf85Xfmb5kvdSv7i50rTPLeqReXNG0i1Z2MMMFmyXqFo141kUR8iKspO+KvNP+cIfNn5g/l1/zlP/AM5D/wDOHPmb80NX/NjyX5H0W38y+U9X1q6N5f2lrMbFjCLqrE1j1aIMteKvGSipyYYq+O9X/wCcP9Mtf+fi3lf8ih+eH5qzWmqeQ5taPm6XXlPmeJvR1Fvq0d/9WoIT6NCvDcM3jir9JPz9853n/OAf/OH97F5P81+YfzE84LejR/Ler+db1dTvjqOqzMyyzyNHGjpAvJlSgBoAdq4q8H8uf84YaZ5x8i6d5z/MP/nNr8xP+hg9f05dSOt2PnKO1sNL1C8j9ZLaCyU81hjZwrKkqVp8HpiihV9A/wDPub/nIHzx+d35Veb9C/M3VE8w+ffye8zXHlTUNcXgDqkMQrb3LhFUFqAqX/bpyO5OKv0LxV2KuxV2KuxV2KuxV2KuIBBBFQcVfJ3/ADjIf8P61+fX5ZsCo8m+fLu/tw3/ACza6i36cK/sqWIFOnTFX1jirsVdirsVdirsVUbm3hu7ee0uYxNb3UbRSxt0ZHBVlPzBxV+Dn/OIf5raZ/zgN+ev50f84ufn/fr5S8n+avMD+Y/KXmy8iMGmztIixiSacsVSOa3jioTsjoyuQSMVfph+ZH/OdX/OLP5Z+XW1/Uvze0LzFLKitZaT5buotW1K9aT+7WC2tWkYhiQORooruQMVfEf/ADnh5/v/ADb/AM5Hf84x/kV50/MPWPyi/wCcePzH059W8xX9neHSDqc8rSRrYXd1VSiqBGhVjwUzcmUlVKqvk/8A5+E/k9/zgv8Akv8Al95b8vfklpelR/nLf6xYXLPpus6hq9xHpSt++muw99cW8QkLoFLKCx+z3xV9r/8AP0T/ANYD0D/tp+Uv+TLYql//AD8p8l6/rv8AzhX+VXnXy9Yy6jL+Vl55b16+iTdY7E2Qhed17qkjx12NASegOKvs7yZ/zmZ/zjZrf5SaL+Z8/wCbnlvRtCbS4J7yG/v4Iby0n9NedrLbF/UEwf4QgUkn7NajFXyt/wA/INXsfzs/5wEH5oeQEu9V8sTXnl3zhbu1vJFO2mSzrH6zwuA6hVuQzVGygk7b4q9w0b/nNT/nHjQ/+cZfKf5l3/5jaTcwyeVrG3XRbK5jm1eXVFtEjfTYrIN6pnWUFCtKbcq8d8VfNn/Pn68bUPyv/O+/e1lsZL78xLq4a2nHGWEyxBzHIOzLWh98VUv+fV009tp//OY1xbWxvbmD809Qkit1IUyuqTlYwTsCxFK4q+fv+cUPKf5Jf85gz/m5+Zn/ADmh+Y8vnTz/AKD5ovraPyfr/mGfR9N0PSohEVkt7GG5tSicyyVRgg40I58mKqc/84Fwfknbf8/IP+cgbX/nHa2itvyftPy3kttC+ryXU9vI0F35bivZYJr2SWWSN7tJWR+RVlIKfDTFXq35reY9C/Lv/n7r+UvmDzpq1r5b0DzF+WTWdrqWoSrb2nryprcEcbTSFUDNKgQCvVlHcYqz7/n6D5ftfzq/5xHuPOH5eXdr560z8tvM1trt8dIuVuYmtbT1bW+q8LEH0VnJcVqoBPbFWOfkv/zjx/z67/N38s/LfnvS/Jvk9GvNLt5tWtbnzLqdtcWF56Sm5guYZdWV0ZHqNxuKEVBBxV9sf84u+Tv+cWvKPlvzVB/zivHoB8tS6yYtcm0DUJ9UibUoIkUxvcT3FyarGw+ENxFa03xV9QYq7FXYq7FXYq7FXYq7FXYq+UNIX/DX/OYfmq2r6Vn+Y3kKzvoohsHvNKuWilk9z6cijFX1firsVdirsVdirsVdiry/80fyV/Kn86tJt9D/ADV8h6T5302zdpLZNRh5SQO4oxhmQpKnIdeLCtBXoMVeSfl9/wA4P/8AOKP5Xa9B5o8l/knoOn69aPzt7y6+sai0DV5K0S309wqMp+yygMKDeuKvVfzb/I78p/z20G38tfm15G07ztpFlP8AWbVL1XSW2lIALwXELxSxkgDlxcBqfFXFXk+kf84Of84o6F5N1vyDpv5LaLB5Z8yTWc+qwl7pp7uSwlWa3Ml2bg3HFZFDcBIFJ6rir178yPyY/LH83fJEf5b/AJjeUrfzP5JhktpY9LmlnhjV7MUt2V7eWGQcBsPi+eKs9j0bSo9HTy9+j4JNDSzGnfUZUEsDWgj9L0XR+QZCnwkGtR1xV8ij/n3r/wA4ajzCvmYfkRof6RWb6yI/VvDaetz58zafWfQ6/s8OP+Tir63k0HQ5tEbyzLo1jL5bey/RraU9vE1ibP0/S+rG2K+n6XD4eHHjx2pTFXzN5L/5wa/5xO/L3zZb+d/KX5JaFpvmSyl9e0uZTcXaW0orSSCC6nmiRhWoIWo/Zpir2f8ALr8ovy5/KaPzNF+XnleDyyvnHWLjX9Z9GWeY3epXTFprh2nllILE9AQo7AYqhfyz/JX8r/yc/wAVD8tPKNv5UHnbVX1vW/q8txL9bv5K8pm9eaXj1PwrRfbFXkXn3/nBz/nFL8zPN935786fkxo2reaNSl9e+u0kurQXctKepPDa3EMbsepYrVj9onFXqflT8hvye8jedZfzE8n+QNL8tecZvL8PlY3+no8AGj27QtFZpAjiFUU28Z+FAfhFTiqXfnL/AM44/kn/AM5BWmlWn5v/AJf6f5y/QTStp087TQXNr6wAkEc9vJDJxbiCVLFSQDSoxVlv5dflX+X35TeSrL8uvy98rWflryXp4nEOlw85Yv8ASXZ5i5maV3Ls5LFia/LFXzd5g/594/8AOGvmXWrjX9S/IvRo7+8uHubhbKe9sreSSQ8m/wBHtrqKJRXsiqMVfTH5efln5A/Kby1a+T/y28pab5N8tWbM8dhpkIijMjmrSOd2dz3ZiT74qznFXYq7FXYq7FXYq7FXYq7FXyf+cJXy9/zkP/zjX5uOw1a41ryex7E6hbLcID9Nttir6wxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV8o/85cq+m+TPInnWBC115G8+aBqCsOqRTXS2szV7USY4q+rFZXVXQhlcAgjoQehxVdirsVdirsVdirRIAqTQYq+B/wA0/wDn5f8A84j/AJU6/d+WNR8+XPmzWtOlaC8i8rWT6lFBKp+JGuA0ULEd+DtTod9sVekfkV/zm5/zjb/zkVqaeXvy4/MCKTzZLHJNHoGqwyafqMkcQLOYY5gElIUFisbswUFiABXFX1jirsVdirsVdir5I/PX/nKiD8k/zr/5x0/J+XyZJ5hf8/dXm0pdTS9W3GmGO4srcSGEwServeBiOS7LTvsqyb8+P+cqfyr/AOcc9a/LXQfzFbVxf/mpqh0rR/0XZfWkSRXhjaS4YyR8UDXCD4Qzb7LQHFX0jirsVdirsVdiqSx+ZPLs2uT+WYte06XzJa263U2kpdQtfRwMaLK9sH9QIT0YrTFU6xV2KuxV2KuxV2KuxV2KuxV2KuxV4L/zk/oz67+QP5p2cK1u4tCuLy1aleE9rSaNh8imKvRvy41hfMH5f+SdbVuY1TQ7C4J61Z7eMt+NcVZpirsVdirsVdir8qf+foX5xeb9I8tflf8A843flzqH6I82f85Ha5Dod3qCuySW2mtcW8BUFFLBZpZwHZSCEVh0bFX2J/zj7/zir+T3/OOvkfSPKfkzyhpz6lb26fpXXLqBJ9R1K7Kj1p5riRS9GavFBRVFFUADFWXap/zj5+TGsfmF5V/Na6/LvR4fzE8mXElzpevWcAtLxGkgltmWaS39P1k9OZqJJyUE8gK74q+c/wDnKv8A5zg0L/nFP8y/yq8q+bvLLah5R896brGoalq0Ezm8tDpyxCGG2tViYSvK8gUcnUVpuOuKof8AIH/nJ7/nIT8xtV80ax+a/wDzi9q35U/ljD5ZufNPl7UhO15qF1DAVZLSW2KxsbiWJuSIUjJp9mm+KvNj/wA5a/8AOcHmyxbzl+W//OCN8nkERvcW6eZ9ftbHXby2SrCRNOke3mjZl3CenJX9gvUYq+nv+cTP+cpPLP8Azlb+XN3500bQrzyjrnl/UpNF8xeXr9/Vm07UIlVygmEcXqIysCrcFPUMoIOKvnX8y/8AnPDz1ffnL5q/Ij/nFf8AIW8/Pnzf+X9U806pLfx6bpWn3MchSW2EkoRCykFOTypV1YIrheWKvhP82v8AnI67/Pn/AJzM/wCcGNI81flnrn5Qfmf+Wnnr6j5p8r60PUEDX1/pEtncWtyEi9aKaOJmB4CnbkvF2VfqD/zlr/zkTp/5Geaf+cfNIvfy203z6/5lecY9Hiur+VY30kn01+s23K3nrIDKKbr0xV6v/wA5I/8AOR3kT/nGP8vZPPnnWO71Oa9uk0zQ9D01VfUNX1KZWMNrbozAb8as2/FamhNAVXxh5g/5za/5yz/Lby+PzW/Nn/nCS88u/kwvoT39xp3mS0vNb0mznZR9YurMKHNA26vFEFO0jR1xV9mebv8AnIPy5Yf84267/wA5H+R4k84eXbXypJ5o0uBpDbfWoxFzSKVuLmNgTRxQkEEYq+IPKn/PxH83Pzu8r6ddf84zf84t6l+aeuaZpNldecb6fU49O0PSdSmiSafS7Se7W2a7ljV6HiysD0R1IYqvoz/nFj/nMez/AOcnfJn5hT2PkC+8n/m1+Vk0lh5g8j6jcBZkvQJxAiTyxQlRJJbvGwkjVonBVgdmZV+T/lb86P8AnJyD/n4N54892v8Azi7Je/m1e+S4rO+/L9fMNohtLNYbel39fb90wPEHiN98Vfrf+Yf/ADldffkh/wA43aP+dP52flle+VfPWrSJp8XkCwvYb+6k1W4llS3tEu4lZKOsYdmCtxBpxLfDirzbyX/zkh/zmhJ5l8nzfmT/AM4ZNov5dec7+zs21DQvMNtqWpaLFeuoS5vrSP1HZEDVkoicBUsVI4lV+iOKuxV2KuxV2KuxV2KuxV2KsX876emq+TPNmmOnqLqGj31vx8fUgkUfrxV5D/zijfyah/zjx+VLTv6l1ZaJHY3B/wCLbVnhYf8ACYq+hsVdirsVdirsVfi9/wA/SIZfy9/N7/nD/wD5yHuYLh/LXkfzUljrdzHA88Vrbpd212SyqCObxCbgOpK7Yq/SP85Pyb/LP/nLD8rdM8r+aNT1K68latcWXmGxvfL1/wDVJZSkUhgdZlWRWRknNQQR07jFX5Af9C9eSv8AnFf/AJ+Yf84reR/yu1TzCdC8zaFd6tfLrGom7keaWLXLZ1DLHCOBS3X4SDvir3T/AJzp0nTNb/5zs/5wC0zWLCDVNNudYuvWtbqNZYZAl7YOA8bAqwqo2Ipir9WPzC89eXvyx8j+a/zC82XD2vlvybplxq2oyRoZHEFshdgiClWNKAeJxV+df5c/85A/858f85J6BB+ZP5I/lT+Uv5afldrUkh0Gf8yr7WL3UtQtopHj+sBdIMYRWKHYx+PFmHxYq8y/59R/4jg1z/nLq18yfo//ABFF595ajFpRlXTvrxFwJjbCUcxGXHw8hyp1xVGf8+d1srn8tvz/ANYuvTk833/5n3i6nM5rctAtnaPAJK/Fx9WWele5bFW/+c7rawT/AJzo/wCfd95FbwLqdx5oeK4mVVE7wRappRgSRgORVWkkKg7As1OpxVMP+fnP/kw/+cKf/Nlxf8nLXFWE/wDPzf8Axvef85I/84T6R5RuNEtr241m8l0qTzOsz6Iuri6sVt2vkgVpDGKivEE4q9083fl3/wA/PPO/lXzN5N13zh/zjpLovmzSrzR75Y7bzGsn1e9heCQxsbRgHCuSpINDQ0OKpdL+Rnmz/nHH/n2T+Zn5Sedtb03zB5h8veVPMUs91pDTvZhby5muY44nuIYJGCLKBUovyxV7R/z7gsLGx/5wv/I4WVnBZi70q5uZ/RjWP1Jpb25Mkj8QKsx3JO5xV8y/84iItj/z8d/5zt0+0Ho2c8dpeSRLsrTPLayM59y07n6cVb8k7f8AP3n8z67V/LiCn/IizxV9s/8AOW//ADjRpX/OU/5TXP5fXPmCfylrun30Os+X9cgT1TZaja8vTaSMMhaNgxVuLKw6qajdV8Q6L/zlN/zlZ/ziP5q/L/8ALX/nMzyfo3nf8vfNWoQeXtH/ADS8py0cysUjRr+ArEHKhl5AwwPxDMPVINVX6+I6SIkkbBkkUMrDoQRUHFV2KuxV2KuxV2KuxV2KuxVZIiyxvE45JIpVh4gihxV8rf8AOHskkP5Xa3oc5pN5Z86+ZNM4fyxR6hK8Q/4Bxir6sxV2KuxV2KuxV5t+bn5S+Rvzw/L/AMw/ln+YukDWPK3mOERzxq3pzwyIQ0VxbygExyxsAysPkQVJBVfmP5d/5w5/5zr/AOceXuPLP/ONX/OTmi6t+VEMzy6X5f8APNs8s9ksv+6kP1O7CqlOkUsaMat6YJOKs4/KH/nCX8/L/wD5yH8m/wDOTP8AzlL+eem+e/OX5f281roej+XLAQ2cdvNDdokLztBZ8Vja8dqCJixO74q9n/5yA/5xf84/m1/zkv8A84xfnVoevaNp3l38kb25utYsr5rgXt0ss1tKi2ixQSRk0iIPN1pt17KvqX80vy70T82vy586fln5kaWPQ/O+k3OkXjwECVI7hCvNCajkpoR8sVfmp+Wf/OLv/PwP8p/Jy/kZ5Q/5yI8hWH5SWTyWul+ZH0q6l8zaZpksjF47OFohAsvF2ZQ8jhSfhlHZV7t/zhX/AM4ieYf+cVNS/OyLVPN8PnDRvP3mCHUdFu5Jpp9Ua2ijYNJqbyQQp67u5LcC478sVeOa9/zhV/zkN+Un53+fPzb/AOcOfzZ8t+UNF/NS5+veZPKXm+2mk09buWV5He3FrbSgojSuyU4MgJSrDqqkB/595/nDq353/kV/zkP59/O20/MP8zfKfmyHW/O1xfrNa2K6XYy20thpnl6zht2WNIys5YuUDM4agbkWVfSn/OXn/OMHnH/nIXzT/wA4+a55W17RtGtfyl83x6/qqaq1wslxbK0TFLUQQTAyfuqUcqN/tYqzX/nLb/nFby1/zlV5AsPLd/rNx5Q83+Vr9NY8reZrNPUn02/jFByTkhaN9uShgQQGU1UYq+brD8uf+fp9jpEHlJvzy/KK7gggWAea7jTb6XVuKjiHaP6msDS0FfiioT1J3xV9P+b/AMl/zB82f84reZPyQ1/8w4vOf5jeYfKdxot15t1S2+pwXV/OprPLBbLIUjBNKKCaDxxVPf8AnFn8pNc/Ir8gfy1/KfzLqNjq2u+TNNa0vLvTDK1pJI88stYjNHC5UCSlSg+WKvHvyW/5xf8AOH5af85Z/wDORn5/6vr+jX3lf84re1h0nT7M3Bv7Yw/Vi5uhJAkQ3hNODt9GKvNv+ciP+cR/zy1L/nI/Qf8AnKb/AJxj/MLy75U8+QaJ+gda0rzTBNJY3cCq8auphhn5FkkoysF48VZHrtir1Lzr+VP/ADmDr35QflnF5c/PzQ/LP5+eTr6a+17UY9OZ/LmupO0g+pzRG3LrEiMoVvR5VFaKfiCrwXVv+cTf+csv+cjPOX5b3X/OWn5n+R4Py0/LXV4tdj8s+QbK6WXVr63ZTH9blvEoqMFoSGb4SwEas3MKv1QRFjRY0UKiAKqjoANgBiq7FXYq7FXYq7FXYq7FXYq7FXyl/wA45E6f54/5yY8t8fTi0r8wnu7dCf8AdOoWcE/ID/W5Yq+rcVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVfKv5Uj6h/zkh/zkfpf2fraeXtV4+Pr2zpX/hMVfVWKuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KvH/wA+fzJ1H8o/yt8w+ftK0y21i+0afTYo7S7d44XF9qFrZuWaMFhxW4LD3GKvYMVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVfKflMG0/5zC/N2NjRdX8h+W7hB4tbT3cbH7mGKvqzFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXyz/AM5pf+s4+ef+YzQP+65p2KvqbFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXypP/ALj/APnMiwXp/iH8tZX/ANb6nfgfhzxV9V4q7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq+Wf+c0v/WcfPP/ADGaB/3XNOxV9TYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq+TvNQKf85mflM4NBN+XPmOI+/G8tHGKvrHFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXyz/zml/6zj55/5jNA/wC65p2KvqbFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXyl51/d/wDOXn5Ky/798l+ZYf8Akpbt/DFX1birsVdirsVdirsVdirsVdirsVdirsVdir5Z/wCc0v8A1nHzz/zGaB/3XNOxV9TYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq+U/wAw/wB3/wA5U/kJL09XQPMcP/CwtT8MVfVmKuxV2KuxV2KuxV2KuxV2KuxV2KuxVokAEnoNzir4e/5y5/ND8ufMP5Cec9G0Pzto2q6tPd6IY7O2u45Jn9HWbCSSiKSfhRGY+ABxV9UaJ+Z35eeZNTi0by/500fWtVmDtHa2V3HNKwjBZyFRiaACpOKs6xV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KvlP8zPj/wCcn/8AnHuNP7xNM8xSN/qejEP1nFX1ZirsVdirsVdirsVdirsVdirsVdirsVdir+fX/nLH8to/y1/PPzhYW1qtvpPmGX/EOmcUCr6Gos7yIlBQBJxKgHgoxV9cf8+5fy2jEnnT81Lu1UcAvl/S3KClTwnu3UncED01qOzEYq/VLFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq+UfOjCf/nL78lrU7/VvJXmS7p/z1to6/wDDYq+rsVdirsVdirsVdirsVdirsVdirsVdirsVfnT/AM/D/wAt31vyV5U/MbT7YyX/AJSvv0XfFFqTY6myrGzEdeFwqKv/ABkbFX1r+Qf5eJ+V35R+SvJ7RelqFnYJc6lUAMb67/f3AY9+DuUB8FGKvYcVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdir5R8wIJv8AnM38u3G/1H8staJ9jPqFuP8AjTFX1dirsVdirsVdirsVdirsVdirsVdirsVdiqWaxo2l6/p8+k6zZRajp1yUaW3mHJGMTrIhI9mQEfLFUzxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV8qN/pX/ADmQhHxfon8tirf5P1i/qPv44q+q8VdirsVdirsVdirsVYX+Yf5ieS/yo8naz5//ADC1+Dyx5P8ALyxPqGp3KyNHAJ5o7eKqxJI5LSyqoAU7nFU48s+ZdD85eXdC82+WNSi1jy75lsLfU9Mvoa+ncWl1GssMq8gpoyMDuAcVYT+Zf51/ld+Tz+VU/MrzhaeU287amuj6ILpJn+t3r0IiX0o5OPUfE1F364q9SxV2KuxV2Kvmf8wP+cofJ/5d/n7+VP8Azj3qmgaze+ZfzZtp7jT9StFtzY2oiLqBPznSTcofsoae+KvpjFXYqtZlQcmYKPEmgxVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdir5S8n8r3/nMD8452HKPR/I3lq0jPg9xLdTOPuAxV9W4q7FXYq7FXYq7FXYq+B/8An59/6w1+eP8Axj0D/wASDScVe3f84h/+ss/848f+a88u/wDdPgxV8Bf8/Zf96v8AnEX/AM2fb/8AErfFX3z/AM5Qf85G6F/zjB+XumfmH5h0K617TtQ8xadoDRWs0MBh+vmStxJJMyqERYyT/Ab4q+TPM/8Az8w0wnV/MX5R/wDOPX5i/nH+U3lWWWLW/PukWDw6TH9XXlcNau8beosfRmcxjaoJWhKr7W/5x/8Az+/L3/nJT8ttJ/M/8t72abR793trqzvEEV7p17EF9a0u41ZwsicgaqxVgQysykHFXyd+ZX/PxLQNE/NHzL+UP5Lfk150/wCch/NnkWRovMz+U7b1LPT5YZWiuYPUVZXeSJkIaiBOXwh+Qair4j1f/nIvyR/zkj/z8D/5w+80eUrLVtB1DQI7/SPMHl/XrVrTUtJ1CN52ME6VZTVWDKyMQRtswZQq/Ub/AJyJ/wCcsNH/ACH13yt5G0z8t/OH5ufmX51tLi+0ny35RsDdSm2tjwee4lJARA23whm78ab4q8L8m/8APxBIPzI8n/lj+f8A+QXnT/nHnVvzBuVs/L2oa+iyadc3DusccbTFIGHJ3VaqrAMQGp1xV4D/AM/efzd8zWf5V3f5UWv5Y+boNCuNR0bVH8/wxFPL/qVua6e1wjV9WoB4mnTFX2N/zin/AM5JedPzbl0vyZ5h/wCcdfzD/K/T9G8qQ3sfmjzRZGDTL6S3NpbrDDMwBaSVZjIg7qjHtirGvzF/5z30rT/zG8z/AJS/kV+TvnD/AJyN88eRnaDzJ/haFV0zS7mOQxyW0944b94jKVbipUNVOXIMAq9E/wCcdP8AnMjyR+f3mXzT+XF35W8w/lV+b/kpBNq/kvzZbC2v1t/3dZ4GUsroDKoIPFxUNx4kHFX1/irsVdirsVdirsVdirsVdirsVdirsVfKX5OH9I/85Af85M6v9r6nf6Jo/Lw+r2Zk4/8AJTFX1birsVdirsVdirsVdir4R/5+ZWN1qH/OEH56wWkRmljtNHuWUdorbW9Mnlb/AGKRsfoxV6t/zhnqNnqv/OKP/OPV1YzLPAnkPRbUspqPVtbVLeZdv5ZImB+WKvhH/n6oDqvmn/nDjyxY0n1vUfzJiuLe2B+J445LVHYDwBcYqzD/AJ/EAH/nEuzDbqfPOjA/L0rzFX6N/lv5d0Ly9+XHkvy3oejWej6Bp+g2Vrb6bawrFbRReglY1iApQ1NfGpr1xV+R/wDz7TuLnyhrv/Oemh+UkNx5S8n+bZrry7bxvyha4j/SqcUU1FTHbQCvcU8MVZ7/AM+bdK0sf841edPNakXfmfzR+YWp/pq/kPO5la2tLAwxyyGrMFE7SCp6yMf2sVY5/wA5I+XtD0v/AJ+jf84k61puj2un6t5j06WTVLu3jEct69sbiKGScrTmyJ8AY78QFrRQAq+vP+ckP+cvtO/Jfzx5T/KLyF+XepfnJ+fPni1a40ryzpTpAILQFiJ766dW9KJjGx2U7KWNAMVfmX/zn152/wCcnvNOk/kNdfnZ+Q3l/wDKnSNM/MTTp9J1LTPMsGtXn1pylYWiiRCg4rUt4gDFX13/AM/cXZ/+cMXd2LM/mnQSxPUkpcE4q++tI1DUNL/JPTtU0uE3Wqab5KjurOLvJPDpweJN/FlAxV+Hv/PuHzR/zl1of5Ga7qH5Gfkt5M/MPR/MPnDUbvV/MWv+YlstUudREFoHinVjzKopDKWPV2b9o1VfUvkr8lP+cv8Azx/zmz+Wf/OSn5qfln5S/LTS/K2g3Wg6u+i65HqDXdq0F2sYaNSXZy1zxFRQClTsMVfrjirsVdirsVdirsVdirsVdirsVdirsVfJ/wDzjaoufOn/ADk5rKksmqfmRMiE/wAtrZ20IA+RU4q+sMVdirsVdirsVdirsVYr558l+XfzG8m+Z/IXm2xGpeWfN+mXOk6nbcipktrqNopArDdWAaqsNwaEYq/Jn8uPyK/5+Gf84h219+W35H6n5M/Ov8m4Lua40GLzLP8AUL7TxdSM7xhS6cF5NzdVdlLEsvGpGKvTfyf/AOcSfz48/wD58aB/zkt/zmV5r0XV/MvkNXj8m+S/LvJ9K0uTqly0nwjkrEOAORZwGZqALiqWf8/h1Vv+cTLNWrxbzzowNOtDFeVpirFtH8o/8/OPy7/Ly2/KX8t9c8mfmV5XuNKih8s/mFrlybXW9NsJ4AUhuIpGbnLAJOMcjB/sg7/ZCr7B/wCcLv8AnFey/wCcVPylk8oXmrR+Z/OvmjUJdc816wisI7m/nVU9OISVb0o0QKK05Hk5ALEYq+N/KP8AzjN/zmP/AM4f/mL+Y1t/zinH5P8AP/5HfmPqra1beXvM921lLot1NzQKpBBIiTihdWPqIkfJQy4qu0T/AJwy/wCcn7v/AJyl/IP/AJyT/NPzxpfn3zDYXV7cedYrWYWmmaBapGF0+w0a3ZS8qj1ZC5oKtvv1Kr1H/nJ3/nGz8/Yv+ckfI/8Azlv/AM4z3Gg61530LRT5d1ryz5kl9C3urECYcoJSAKskzKw5KQeLKTuMVeKfn9/zjX/znZ/zlLZ+TvMn5iyeTPKMfkrzTpd7pH5e6HemSD0Ayte6lfalMDzlQKFjiWooWI3+0q+7/wDnMT/nHm//AOclP+cdfM35UaVqNvpXmWUWWoaPc3Vfqwv7B1dUmKgsFkTmnID4SwbelCqwL/nE+L/nNbTbrTPJ3/ORnljyZpfkDyh5T/RNtqekXn1rU9V1O3ksorWWREJRY/qyTeoeI5OV264q+ePLP/ONf/OXH/OHfnjz83/OJ6eVPzJ/I7z5qk2vJ5K8zXT6fPo905AEVtLy+ICOkYYN8aInMclqVX1D+Sdz/wA5yeY/zFTXvz10/wAg/l7+WNrYzKPLPl+STUtSuruRSIna8YkRqhIJFd+lPBV9pYq7FXYq7FXYq7FXYq7FXYq7FXYqtdlRWdzxRAWYnsBuTir5T/5w/V7r8ufNPmOdf3/mnz15k1Dl/NEL6SKE/wDARjFX1dirsVdirsVdirsVdirsVdirsVfKv/OYX/ONC/8AOVv5TQflg3mtvJvo69Y619fW2F1UWglVovTLx/aEpoa7EdDir6Z0jTxpOk6XpaymYabaQWokIoXEMapyIHjxxVMcVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirGPOuoJpPk7zXqjv6a6fo99ccj29OCRh+rFXj/8Azihp0mnf848/lYJ4/Tu9Q0dNQuR/xdeO8z/8TxV9D4q7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXg3/ADlBq0ui/wDOP35s31u1Lr/D11BbitOUtwBEij5l8Vei/lvpUeh/l95I0iNOC6doWnwcfArbxhvxrirNMVdirsVULqRora4lSnKKJ3FelVUkYq/Bf/nGHSf+cwf+cvJPzk80WH/OavmT8sIfJXnO80a30uPR01KB0LPKnFl1CxEaqKKFCNsOuKvfvyI/5yL/AOcj/wAnP+cqbL/nD7/nKbX9P/MwecbJrzyX52sLVLS4mCxSzRpcJFHErI6QSK3NeaSLTnIrCir9UvMPmvyv5RsxqPmvzJpflnTy3EXOq3kNnDy609Sd0Wv04qraH5i8v+Z7Ian5b1zT/MGnM3EXWm3MV1CWABoJIXda0I74qiLjV9Js7+x0u71O0tdT1QSGztJZkSe4EQBkMUbMGfiDvQGmKpVpHnTyfr+qapoehea9H1rWtDPHUdPsb63uLqzNeNLiGKRnj32+IDFUBrX5k/l35b1KLRvMPn3y7oOrz0MdjqGqWltcvXpxillRz9AxVmMUsU8cc0MizQyqHR0IZWU7ggioIOKpXH5h0CZdVeLXLCVNDkaHUmS5iYWciCrJcEN+7YDchqYqoeXvNXljzbZvqHlXzHpfmawikML3OlXcN5Csi0JQvA7qCK9K4qh7Hzt5M1PW7zy1p3m3Rr/zHp211pVtfW8t7BtX97bpIZF28VxVk+KsU1Pz55H0XVbbQtY85aHpOt3ppb6feahbQXUv+pDJIrt9AxVkF5qFhp9pJqF/ewWNhCoaS5uJFihRWIALSOQoBJHfFXwf/wA4xf8AOZjfnV+Yf/OSPlLzm3lXyppH5QebP0J5evLe8MT6lZGe/iE0rXFwyO1LRTyjAX4unTFX3uZohF65lQQcefqchw40ry5dKU74qxnRvPXknzHfXel+X/OGia7qdgxW5tNP1C3uZ4WHUSRxSOyn5jFU+1DULDSbK51LVL6303T7NDJPdXUiwwxIOrPI5VVHuTiqQan578kaLcaLaaz5x0TSbvzGVGkw3moW0El+XoF+qrJIplrUU4VxVF+YvNvlXyhaLqHmzzLpXliwclVudWvIbKIkdQHndFrv44qj9J1jSNesLfVdD1S01nS7teUF5Yzx3EEi9KpJGzKR8jiqY4q7FXYq7FXYq7FXYq+U/wDnL+aW6/Lfy95StW/0vzx5y0DSFT+eFrxJbgU/4xxnFX1RDFHBFFBEvCKFFjRR2VRQD7hiqpirsVdiqFv/APeG8/4wSf8AETir+c7/AJwV/wCcz/yw/wCcZrP8+vLfnbQPNmvaz5j/ADAu7+xh8t6Z9fDxr6kRQuZYgHLLsD18cVfVP5L+VvzY/wCcvf8AnMzQP+cufN/5b6x+Uv5PflRpT6f5KsvMcfo6nqszpPEJPR2+HlcySMwqiniiM5DEKvoH88/+cXf+cM5vzG1785/+cqPOEerXfmd4YNLsvPHmhtM0fS4oY0QQadBFcWRAJQsys7KSSeNd8VfEH5LX35K/k7/z8n/Lbyd/ziR55stW/JX84/KF/H5l0bRNXfV9Nt9TtbXVrhEDSTTkOrafBIpZi6iSRVIR+OKs5/5+UaH5x80/85Y/84beVPIPmd/JXmrzZHrGiWeuxVEtguoT2dvNPGQOQZYpG4lfiB6EHfFX3L+TH/OBv5EfkEdU1P8ALS31zRfOGueV7nyxqPmNtTnmvJ1uuLSXoR2aKOcOoZTGqgeHfFXyrrP/ADib/wA+ufyxtb3yb+bPnHy7f+fLiNm1TWvN/nWSLzFNcTVLXUiRX1tGsnJi20IFftBsVRH/AD6U84Xt15J/O38sYvNU3nDyd+WPnSS08pX0kwnjXSrhW9NIHHWMmPmANhyNNsVfM3/OOf8Azjhp/wDzkv8A85U/85m+XPzD80akfyX8l/mde6vq/kjT55bOHzDql1f6ilo99cW7RSejbrbOQoblyYFSnxEqpZ+cv/ON9/8A84+/85s/l3+TH/OLn5g6x+R/k3/nKDy0uma0tndT3jWMCXFyl8LVp5Hl5+lAGgdpPUjkd+MqK2yr0T/nOX/nCT8mf+cYfyG078+PyGh1ryL+Z35Ua5o1wNdTV724u9SNzdxWpluGkmKpKJJhJyhWMH4hxoaBV9p/85Vf85LebPyx/wCcFLT86dCkWz8+eePLXl+KyuouMf1O/wDMNtC8lzEpVhWISOyDsQPDFXwv+V35Z/8APrWb8rdLi/Of82PLv5gfm15t06PUPNXmzVfMd8NUGq30QlmMTJchEMMkhA5KxYj94XxV6h/z76u/Lv5//lR/zkN/zif+ZmtS/nP+VH5feYYLHRNRnvblTe6A9y09jEl1BLHMEjmsA8fGQcVIQAIAMVeEf84Sf84Y/wDOOH5u/m7/AM5j+V/zA/L467on5Veel0byxb/pTUrb6lZG71eL0udvdxNJ8NrGOUhY7ddzir3j/n4P+ZGhL+bv/OPP/OHutefz+T35F61pceteftRhvGtDNoMDT2trppuDyKxuthKlHDh3ZCwPGjKvBv8AnI7yr/z7k8tflNqXnP8A5xa/M3Qvy8/PX8uEi1nypqHlrzBfyaheXFs4LWxE91LyaRWajCjA03K1Qqvsz88fzM1P85P+fVev/mZraKmtebfIFjdahxHFXuo7y3hmkAAAHN4i1AKCu22KvMP+cQf+cCPy6/Nn8ofy3/PL/nIHU9W/MX8w/Mun6TqegypqNxaWugaVYFG0ywtYYDGhCpGOYYFdyFA+0VXk3n//AJUpP/z8C/NvSf8AnPi2uJPKN1p9lb/lNNrlxdweV7XTyATWS3kijQsi7sx9MSCX1P3hU4q/Wz/nHX8ivyR/JTQvMM35CR/VvJX5g30OuC2tdTk1PTFlFukPqWLyyzlVkVAW+NqmlKAABV9E4q7FXYq7FXYq7FXYq+TvztH+IPz1/wCcZvJ6GpsdZ1PzbMn80emWhjHIeAa4B+eKvrHFXYq7FXYqhb7/AHhvP+MEn/ETir8iP+fRD8vK3/OS6hqhfzOuTSuwJjb+FMVfsDir+f8A8va/+RWjf859f85P3v8AznXFanVY9QjT8tZvOlvLcaJDoscl00YhSRGg4m2aD0uSkVL/ALZbFVPy957/ACg/Mb/n51/zjL5j/IfyFD5S/LK00rXdFs9VsdEi0TTdfvrPTNbmu7uyjihh9VIxdRxGQrUldvh4nFX0V/zmqQP+c+v+fflTSusXg/6fLLFX6F/85KyefYvyA/OCT8r/AKz/AMrATyrqR0L6nvc/W/RbiYP+LKV4960pir8cP+cOvzT/AOfd/wCXv/OPFhqX5oaN5f1r864GuP8AFmn+ZNF/TXmLUdWeeQrHYw3cE5cMGVV4FQGrzINTir2n/n05IZde/wCcs5j5Xm8k/WPO8U66BcwLazaakqzulrJAgVY2jUhSoFBSmKp7/wA+7GU/85P/APPxQBgSPzFirTt/uR8wD9YxVv8A5yqIH/PzD/nBsV3OmX//ACcvcVeu/wDP1z/1iL8zv+2l5c/7rNjirGv+ci/yW8yfnv8A8+3PI/lPyZafpLzTpXkXyf5h02zRect2+m6daySW8ABH7ySIsF8TirzX/nH7/nKD/n37rv5X+WrX81PKX5d/lr+ZPlnTLfTPMujeZfLFnb3K39nEsUrxu9h+8DleXiCaMARir7u/5xj/ADO/5x7/ADU0DzbrX/OO3l+w0jyxpOtfou/u9N0JNDtr67jgjkMkQWCD1lVZAvMj5bUxV+YH/OM/59fln/zif/zlR/znb5V/PjXH8gzebPOZ1/Qri+tpvTv7NLzVZg0ZRG3kivYXjH7YJp9k4qzb/nPPRrXyh+f/APzjN/zmte+Rk/Mz8ltP0mPy/wCcbSTTU1FbbS7s3U0F3NZ3EboQYtUlaMuBxkRQSrMpxV7Hff8AOUH/AD67svLkfmNYPyy1BJo1dNMs/KVrNqbFwKR/Uxp/qBt6UIHvir0D/nNTUvLusf8APvj8z9Z8paX+g/K+r+TNPvtKsDZ/o829pcz2ckMZtOKekQrgcKCh2xV63/zhCQf+cRv+ceCDUHyRpe//ADyGKvFvzL/5yk/5wU/M298+/lD+f97otlfeQtVv9Hv9K88afJbSCW3laF7jTbgoWAkEYKvE6sVpirwX/n1LY3VtrX/OUsvkGTWm/wCcY38226fludZWZVkpJftcPaCUBqeg1t6hIBNUr8XLFX7E4q7FXYq7FXYq7FXYq+TYSPMf/OZl5y+OP8t/y6j9JuoWbWrsh0B7HhACcVfWWKuxV2KuxVogEEEVB6jFUi0Lyt5Y8rR3kPlny5pflyLULhrq6TS7OGzWed92llEKIGc92NTiqfYqw3zT+XX5f+eJLSbzp5H0DzbNYUFtJrGm2188IDc6RtPFIVFd6DbFUdD5N8oW9zod7b+VdHgvPLEUkGjzx2NusunxTKVkjtHEYMSspIYIQCOuKq+oeV/LOr6ppWuar5d0zU9b0EsdM1C7tIZrqyL0Lm2ndGeOvEV4kVpiqe4qwBfyo/K5PMR83J+XHlhfNTS+udXGk2YvjNy5+r9Y9H1OfLflWvvirJtN8u+X9Gu9Wv8AR9C0/Sb7Xrj63qdxZ2sUEt7cU4+tcvGitI9BTkxJxVZpXljy1oV5q+o6J5e0zR9Q1+f6zql1Y2kNvNezVY+rcyRIrSNVj8TEnc4q698seWtS1jTPMOo+XtMv9f0VXXT9TuLSGW8tFkrzFvcOhkjBrvxIriqtrmgaF5n02fRfMui2HmHR7oqZ7HU7aK7tpCjB15wzK6GjAEVGxxVH2trbWVtb2VnbxWlnaRJDBBCgjjijjAVERFACqoAAAFAMVefa9+Tn5R+adSbWPMv5X+VNf1Z5PWe91DR7K4uHfpyeSSFmY/MnFWcaVpGk6FYQaXommWmjaZagrDaWMMdvBGCakJFEqqNz2GKsX8y/ll+XHnO+ttT83+QPLvmjUrMKsF3q2l2t5PGqElVWSeJ2ABYmlaYqy+eys7qzm066tIbnT7iFreW2ljV4XhZeLRtGwKlSpoQRSmKvOLL8kPyZ03U/0zp35TeTrHVeSuLuDQ7COZWXoyusAIPuMVZ/q2j6Rr+m3mja7pdprWkahH6V1Y38EdzbTxkg8JYZVZGFR0IxVV0/TtP0mxtdM0qxt9M02xjWG2tLSJIYIY1FFSONAqqoHQAUxVhfmX8p/wArvOeoR6r5v/Lnyz5o1SLjxvNV0q0u56IKKDJNC7EAdiaYqzLTdM03RrC10rR9PttJ0yxQRW1nZxJBBCg6LHFGqqo9gMVR2KuxV2KuxV2KuxV2Kvkv/nHMHzL5/wD+ci/zLaktvrXnH/D+nSHcpb6DCttLGD4etU4q+tMVdirsVdirsVdirsVdirsVdirsVdirsVdiryT85PzIu/yw8uaPrlnpkWqyapr+naI0UzsgQahL6IlBX+RiDTuK4q9bxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxVjHnbX7byt5P80eZLuT0rbQtKu76R+lBBE71r9GKvGf+cTtAudC/IbyNLqEXpax5khn8wal/l3WpzSXDP47q64q+jcVdirsVdirsVdirsVdirsVdirsVdirsVdir5b/AOctf/Jf+Uv/AAPPLf8A1Grir6kxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV8xf85eapPb/AJKa15dsQZdT/MC/0/yrbQL9qUancxxTqvyhLn6MVfRGg6TBoGh6NoVr/vLotjb2EPb93bRLEv4Liqa4q7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXy3/wA5a/8Akv8Ayl/4Hnlv/qNXFX1JirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdir5Y/NJD5y/5yD/InyKAZbDyj+kPPupAH+7kso/qunkjvWWduvhir6nxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV8t/85a/+S/8AKX/geeW/+o1cVfUmKuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2Kvl38n2Tzp+c355/mcB6tjpl1aeQdHmrWsWkqZ78D2+tT9vDFX1FirsVdirsVdirsVdirsVdirsVdirsVdirsVdir5b/AOctf/Jf+Uv/AAPPLf8A1Grir6kxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KvP/wA1fOkH5d/lx5086z0P+HtJuLmFG6SXHErbxfN5WVR88VYh/wA45eTp/I/5M+R9Kv6vrN/ZfpjVZX/vJL7U2a7mMn+UDKFPyxV7firsVdirsVdirsVdirsVdirsVdirsVdirsVWu6RI8kjrHHGpZmYgKqgVJJPQDFXx7/zlR5s8qal5E8qw6f5n0i+mj88eXZXSC+t5GWNLxSzkLISFA3J6Dvir6n07zR5Z1e4NppPmLTNUugpcw2l3DPIFHU8I3Y0GKp7irsVdirsVdirsVdirsVdirsVdirsVdirsVdir5J/5yOL+ffN/5QfkLbFntvOOsf4h8yqhIC6HohE7JIewmmACn+ZcVfWqqFAVQFVRQAbAAdhireKuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxVa6JIjxyIJI5AVZWFQwOxBB6g4q/nA/PX8tYvyr/Nvzv5Nis1t7DT9Qkl034Rvp12BPa703pFIFJ8QcVfoN/z7m/LOCx0nzl+adzZJHcapKug6ZIFCn6vDxmuj06M5jAP+SwxV+neKuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxVpmVFZmYKqgkkmgAHUk4q+TvyIf/lZX5k/mr+fUil9Hv518meUHbodJ0lybm4jqPsz3PxD5EYq+ssVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdir8t/+fiX5ZT3d15A/MnSbRpri5c+VtR9NeTM7s0+nmi79TMpPugxV9+/k75Ch/LH8svJnkiJFWbQ9NjS8KEENeS1lumBHUGaRqe1MVel4q7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXzL/zlH541TRPJNl+X/lCTl+YX5vXi+WdEiQ/HHHcUW8ujuCFjiYgtXYsD2OKvbPIXk7S/y+8meWvJWjJx07y1p8NlG1KNIY1+OVuvxO5LN7k4qy7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYqx3zN5Y0rzZYWmnavAtxb2Wp6dqsYZQ1J9Nu4byE7g/twgH2JGKsixV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KoDVdU07Q9Nv9Y1e8i0/S9Lt5Lq7uZjxjihiUs7sfAAYq+RPyO07U/wA4PzG13/nJPzLZy2uhmF9D/LqwnrWHSlJE2ocD0a4NeJ8C1PhYYq+ysVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVYJ52/M78vfy4gjuPPPnHSvLCzxtJBHf3KRzTqmx9GEnnIa7UVSa4q+ZDc+cP+crtRis/wBFap5G/wCcdbKZZrqS+ja11Lze0T8kiRD8UVnVaserdOv2VX2dZ2lrp9rbWNjbx2dlZxJDBBCoSOONAFVEUAAAAUAGKojFXYq7FXYq7FXYq7FXYq/G/wD5zb1/84PNP/OaH/OPP5CeRfzp8x/lH5e8/aLP9ZudAY1SfncP6zwrLB6hIgC/E+w6e6rEfz/8of8AOZX/ADgx5Qs/zx8r/wDOVGq/nf5S0XU7SDzH5c826coWSC4cxoysZrtgnIgMUkjYEqRyFRir9fvyo8/2P5q/ll5B/MrTbaSysfPegWGuQW8tC8SX0CTBGpXdedMVQmofnP8AlHpOvx+VdT/M3yvp/mSUsF0241W0juaqaMDG0oIIOxBxV6QJY2jEyyKYSvMOCCpWleVelKd8VYPP+aP5bWvlq585XHn3QIfKdpNJbTaw+o24skmiNJIjP6nDmp6rWuKpr5V86+UPPWmrrHkzzPpfmrSmoPrWlXcV3ECdwC0TsAT4HFU7v9QsNKs7jUNTvbfTrC0XnPc3Uiwwxr05PI5VQN+pOKsI8qfm1+V/nq+udM8mfmF5d806jZu6TWul6lbXUyNH9sGOORm2pvtirOrq7tbG3mvL65is7S3XnLPO6xxoo6lnYgAfM4q8w0b89/yV8w62fLWhfmv5T1bX1k9L9H2urWklwZKkBAiykk1HQYqrfnH+ZWkflV+XvmrzXqGs6VpWo6fpGoXOkRatcJBHd31tayzRQKGdC5LIKqu9MVfNP/OFf/OWtr+f35K+T/Nf5k+ZfKWifmV5g1C+sH0XT7pLZpGt52jh9K0nuJZgXQA0qa9sVfc+Kvx8/wCc+PM/5vav/wA5V/8AOLf5Gfl9+cWv/lLon5m2txBfXehMeSzy3LoJ3jWSEyFViAClwBvir0vRP+cEv+cgdK1nSdUuv+c/vzF1O2068guZbOTTwEuEidXaJi2qyCjgUNVPXocVfpdeXtnptrPfajdw2FlaoZJri4kWKKNR1Z3chQPcnFXn3l/85vyk82avNoHln8zPLGva3BJ6T2FjqlrPcep/KI0lLE+wGKs81TVdN0TT7zV9Y1C30vS9Piaa6u7uRYYIY16vJI5CqB4k4qwbW/zi/Kjy3D5cuNf/ADI8t6PB5wSOXQ5LvU7aJdQjlp6b2xaQCRWqKMuxxV6OCCAQag7g4q3irsVYrrXkbyX5k1Ox1nzD5T0jXtV0yIw2d1qNnDdS26MwciJpUfjVlB2xVlIAAAAoBsAO2Kt4q7FXYq7FXYq7FXYq7FXYq/DX/nOnWfPnl7/n4X/zi7rH5ZeU7bzx56sdClfSdDvLn6pBeSl7xWR56jgAhZq+2KvOPz4/Nz/nIj/nJL8z/wAvf+cS/wDnJ/SNO/5xJ8iefdQt7prqGKbUW12WGULb2tvfl2hBeQlVrRfU4c+ylV9zf8/BfzHvv+cT/wDnDG18v/lLI/lu5vptK/L3Q7qCQrcabYm2naWaFuvP6vZNGGG6l+Q3AxV8heUtD/59C6f+WFn5O83eddC82+bbzTOGt+cLo6r+mbnU50JuL2Kf0yI29V2ZFAKiihg9CSq+gP8An1x+amu+dvyD/NnyBrHmW4866b+UHmG+0by/rtz6jPc6NcQySWqc5KkhTG7KCSVR1TYKuKvlf/n2z/ziZ5H/AD/8geYPNf5z3t5538h+S/N+rWXlryHLPNBo9pfXEUDXuozxwtEZJJF9NU+KgC712GKvSL78v9G/5we/5+Ifkpo/5PSz+X/yr/5yNspNO1nyy080tlBKsjxcow7sTwkMbx8qlauAaNTFVT/nOr82/Ifnf/nMD8vP+cd/zv8AP0nkT/nHTyPpEXmXzfbxyzxpreozoZ7S0uGt0LcKelQEGg5sKMVKqvB/+csNc/59+aH+Wtr+YX/OIPnXSPIH59/lxqen6j5efyqNQt7i/RriK3uIZfXXg3CKVpSzbkIytyDEYq/YHzV5O8h/85Zf84y/lhq/5yane+WPJ/mnQ/LvnjXIbHUjpFvIbnT47lrS9men+jc7r4lYjdV3BFcVflt/zlb5Z/59gwfkn55tvyZ8xeTvLv5weV7Q33li48tXt5JfS6jZOrLbGXlMj+pxK1JqDurDFX2Pp3lHyd/zlB/z7z/L/wA7/nX5ei88eYvLn5d6hrOnXl7LOk0Op2Vlc2q3ZaGWMu7C3UtzqGO5FcVeN/8APsP/AJxa/IHzh/zj7+W35z+Zfy10/V/zO0zX9RurbXZprr1kmsbx1tn9JbhYSYwopVDuK9cVftDir8PP+fiuvecvLH/Obn/OHnmD8vfKS+e/Ouk2M1xo+gNN9XXULlbybjCZf2agk19sVfTnkb/nI/8A5zw1zzp5T0Xzb/zhZa+WPK2ravZ2mr6wPMCSGwsZpkS4uQlDyMcZLAd6UxV41/zl6+s/85M/85r/AJQf84YXfmK+0P8AKOz0NvOHnK20u4aGTVSi3MwtZyg2Cx2yhK1AMpbjyVSFWaf85Qf8+6P+ceLL8kPOXmb8nfJA/LH8yfy80mfzDoOt6Le3kU3r6ZGbgpOzzSlgyxt8Qo6tRg21CqxqH849e/Pj/n015388ebp/0h5lTyhf6Nql29C13caXdpbfWJBQDnIqKzeJqcVYb/zhR/zgV+VH5v8A5AeSfzU/5yFS7/NPzR5w0OztdES5vLiG20DQNPH1fTrG0SF4viWOPk5NRViKVqzKv2tRFjRI0FFjUKB7AUGKrsVdirsVdirsVdirsVdirsVdirsVdirsVfjl/wA5MMo/5+k/84dKWAY6TPQfP6/T9RxV75/z84/JOX82f+cZte8x6HbyHzx+Tsw84aLc2wYXUcVrT6+kTqylf3A9Ukb1iXFXzt+dln5l/wCc/v8An2v5M83eSLb9M/mL5amsdcvdKgT9/eanokdzp2qW0ALVDMs7zRjq9FA3YYqzb8sP+c9v+cHtZ8k6bP8AmnpGgfld+Y+l2kdv5j8uar5XImg1OFAl1HB6VjICnqA8eXFgNmAIOKvs78ifzc/K787Pyr80edPyh8s3nlzycl7f6bC93o40ZdQkgtIJWvLaIBfUhYThVk7lWXqpGKvxp/59zf8AOYvlX/nHTyZ5q8p/nJoGpeWvy080ebtRutE8+wWs11p/6VSOEXWm3npK5RhGqPGQKn4qim4Ve4eXvOLf856f858flf8AmZ+Wei6h/wAqK/5xqs5ZJfM9/ZvDBqGouXk9GMSEULysnAbtwQuyrUYqyD/nMLQrr/nHj/nMzyB/zmD5g8gzef8A8kvMGgjyx55a309NRbSGjjMKXMkLhlA4iJkduIPF46hitVXsOqf853/8+87TSba+0caP5w1a94pb6BoflFp9UllfZYliksoU5E/5eKsO/wCfrkHmjUf+cfPyu1DR9I1Z/wAprLzZp1/5803TI3guI9GW3YwrcRx7JEjEhgw4rJ6Z/Zrirxj/AJyA/wCcjf8AnCzXv+cavOn5Z/8AOMf5b6f5z81a95SvI4LXy/5a+qt5bsUiLXGo6lez2q+kLdATXmSzUHIVrir7L/5xK0W88y/8+4fIXl7TY2n1DXPy31iwtY03Z55/r8caD3LMBir5N/59rf8AOWX5Qflp+UHlP/nHDz7qWo+XPzitPOV/oieX5tOuTNJcX927R7qhVeBJWTmQVI6Yq/brFX48f85ikf8ARR7/AJwPFd6t/wBRk+Kv2HxV+Nn/ADmLD5o/5xn/AOc0Pyn/AOc1Y/Kd/wCafyobQW8qec30q3MsumKVuYWuZuLD7UVyGjLUXlFxZqsuKsg/5yH/AOfjf5O/mB+U2v8A5bf846XGr/mt+bn5q6ZNoGjaNpel3Ye2bUEMEslyJI0+wkh+Fa1PUgVIVTPXPyR1L/nHf/n1b56/LbzA0a+Y7HyZealrSRsGjhv9RuUuZ4VcUDCMycK96eGKvp3/AJwA/wDWOfyB/wDAXg/4m+KvsPFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYql82k6Vc31tqlxplpPqdkpS3u5IUaeJW+0I5SpZQe9DiqNkjjmjeKVFlilUo6OAysrChBB2IIxVCadpemaPbLZaTp1rpdmrMwgtIUgiDMasQkaqKk9dsVY/qX5f8AkTWL6PU9W8l6HqeoxOXS6utPtppgzGpbm8ZapPviqcX9nHFol/Y2FqkSfU5o4YIVCKCyMAqqKAVJxV+af/Ps78m/MvlP/nHzzz5R/OP8up9Hk1Hz9ql7HpPmSyQ+tA0NsizCGYOCpKsFam9NjTFX6ZaZpGlaLapY6NplrpNlH9i3s4UgiXYDZI1UdB4Yqiri3t7uGS2uoI7m3mHGSKVQ6MPBlYEHFWLaf+X/AJD0i+l1PSvJehabqUzc3urbT7aKYt4mRIw1fpxVlM9vBdQy211BHc286lJIpVDo6nqGVgQQfA4qkOl+TfKOiRXkGjeVtI0mDUQ63UdnZQQJOJPtiRY41DA9wcVTu0s7TT7aGysLWGys7ZQkUFuixxRqOioigAD2AxVJf8HeUf00PMn+F9J/xCtaan9Sg+uAnqfX4c6+9cVZHiqX3Gk6Xd3lpqN3ptrc6hp9fqtzLCjzQcuvpyMpZa+xxVMMVUbi3t7uGS2uoI7m3mXjJFKodHXwZWBBGKpBpPkzyhoNxNeaH5W0jR7u4cySzWVlBbyO7ChYvHGpJIxVPrq1tb23mtL22iu7W4UpLDMiyRup6qyMCCPY4q1aWdpp9tBZWFrDZWdsgjhggRY4o0HRURQAAPADFURirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVfnX/zlV/z8T8jf84qfmr5Y/LTzN5D1nzFHrGnQ6pqGqWE0KLZW08zxApC9WlKhCxFV8BXFX335e8waP5r0HR/M3l+/i1TQ9fs4b+wu4W5RzW9wgkjdSPFWGKvm38ov+cpNL/Nr89/z3/I608pXejXn5Hz20E+qzXEcsWoGd5Y2KRKoaMAxbVJr7Yq+q8VdirsVdiryX8zPzz/K78ntU8g6N+YnmdfL+pfmfrCaB5ahNtc3Bvb93hjWKtvDKEHKdAWcqor1xV61irTMFVmPRQSfoxV+UOn/APPzHzZ5t8x+eNI/LH/nED8xPzO0vyLrt1ol7qfl+t4okt5ZI1MiQWkwRnEZYKW+nFX0J/zjl/znT+Wv/OQPnTW/yrn8s+ZPyq/NzQIpJ7jyn5utVtbuWKEK0rW7K78igYFkYI9PiClQSFX23irsVed/mp+a3kL8lPJOqfmL+ZevL5b8n6M9vHd37QzXHB7qZIIVEVvHLIxaSRRsp+7FWX6Drel+ZtD0bzJod2t/ovmCxt9S0+5QFVmtbuJZoZAGAIDI4O4riqa4q7FXwH/zgb/zkf8AmP8A85E6f+el1+Yh0sy/l/5+uvL2k/oy1a1Asok5KJQ0svJge+KvvzFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FX4p/8AOSn5eeW/za/5+VeS/wAs/N1qLry952/KXUtLugAC8Xqw3xjmjJ6NHIquKEdKV3xVnv8Az7d/MzzN+XuufmR/zg5+bVy0fnn8l724n8ryzk/6foTvyKwFmPJYzIsiU29OQAfZOKqf/OGn/wAkA/5z+/7aOn/9RN1ir0783f8AnJP88vzE/wCchdW/5xZ/5xKtfL1nr/kbTU1Pz5568zLJcafo/rKvCzt4YvtzVmjqfiPLkvCiOyqsJf8A5yA/5yp/5xQ/Nb8qvJv/ADlXqnlX8zvym/N7URoVj558vWsmnXOk6o7qEW+hZVQrR1JUIKpyZXJRkxVl/wDzkn/zkt+b3/OLX/OSP5Yat52ubDVP+cUfzPdNEup47L07vQdU4lS8typYsK0lFR8SCRQAVqVUy/N//nJX8y/N/wDzlF+XP/OMP/ONeoaZ9Z0+OPzH+Zfmaa3TUbbTNIIilS2jHNU9SSJhvyrWSMCnxHFXxJ/z8n0X/nJRPzz/AOccJtT84eTpvLWrfmciflbb29pcpdaXem40lY31lmjKyIJDGT6ZOwbbemKv1j/5x60H/nKTRE81/wDQynnbyV5wa5ktToH+ELO5tfq6qJvrX1k3EcXIMTHwAG1GqdwAq+jpv7qX/Ub9WKvwK/59/f8AOTf5GfkLq3/OVOnfmz+YNh5Ov9c/Mi5urC3ullZ54Ypb5XdBHG+ysQD88VZ5+WfmbQv+cw/+fkOmfnl+T0jRflj+TfluPT9T1+dBaT6vdGK5WOKK2lKTMhM/HkVFFUk7FQVX6Jf85F6j/wA5ZT6n5S8p/wDOM+geVLO11mK4k8wedPNk7PBpIUhYYobGJhLI71LcgrgUoV74q+GPzI/Or/nOL/nCzV/y784fn35x8mfnT+Tvm7zDBoOtz6TYPp97pbXFXDRt6UAqI1kdKq4bgVbhUNiqP/5+26f+d95+Rmu67oPmby1b/kElloS67olxbznX59WfVgYprecK0QhAe3qpYH4X61GKvef+cM/Ln/OX+naB+WWqfmr+YHkHWvyZl8h6YdJ0nRbG7i1iKN7G1OniaaSKOOscW0hqanp44q8iv/8AnI//AJyn/wCcpvzi/M78tv8AnEO68reQvy4/KG8bSNZ8+eZIZbt73UlZkMdrEI5AF5xSBaRuCqhy4DKMVfUf/ONep/8AOXum+ZvN/kH/AJyb0fy3rml6PY2175d8/eWWMVvqjPIY5rWe2YhllSnKpjj27N1xV+SP/ODfnT/nJe41X/nIb8pf+cbPKmj2mq6z+Y2o61rv5geafVfRdEtAZLeK2SCJeUt1K6EqBXioqUIqVVfbv5F/85If85K/l3/zlTp//OJP/OVraB5o1Dztosur+TvNvl6B7dbs28VxOwkTjGpjZLSZamNGR46fEHFFX6q4q7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FX5GfmN/8lu/Jf8A813df8m77FW/+fj/AOWvmP8ALXzJ+Wv/ADnT+VFpTzn+TV/bW3m2CEBf0hocknpq0xCNVVErQuTWiSKQPgxVh3/PvPz15e/M7/nMf/nNH8wvKl0bzy75yj0bVrCQji3pXMty4VlNCGUkgjxGKvGPyf8Aym86eef+c8v+c2vI9h/zkF5n/InzfP5hm8wWkehQW7TazpMt7czRc/rBU8beG9gKca1Dk9Bir7C84/8APt/zN+Zb+WYPzV/5y9/MD8wNE8r6zb61a6fqNnZKq3MFVDJIrVRirstQD1xVP/8An6F558n2X/OP3/Kob7y9H51/Mb86dSttF8laGhJuU1BJoyNRQKGYCAsANqMzBDsTir5k/wCfZ8f/AELp+dv5yf8AOMH5yaPDpP55a1HY61Ya9LK051nToLZT9TguJBVliDF0oaNR/wCTFXqH/Pztlh/Mz/nAi5lYR28H5uW7SStsqD67orVJ7bKT9GKv10xVTm/upf8AUb9WKvxR/wCfY35c/l953v8A/nLS486eQ/Lvm6ey/MyeO2l1rS7S/eJGkviyxtcwyFQSKkDFWI/854/l/wDl9/zjn/zkR/zix+YX/OPVhafl/wDm15l80Jaahonl1Rbx6hp5ntovVlsovgCvzaFgFCuCag8dlX1v/wA5af8AOQn5zRfn3+Uv/OJf5AappXkvzh+ZthPqeq+cNWg+t/o2zRZyEtbdgFaTjbs1fi/ZWg+Jgq+Df+fmH5Lfm5+Wv5A+XtX/ADK/5yv8wfnGmpeb7C3i8tarpun6fbGYWd8z3kAt29QmIfDsKfvN96Yq+9P+foaO/wDzgf59ZVLLFN5YZyOig6rYLU/SwGKvrv8AIdvrf/OOX5MtaOJPrP5b+XTC6bhuekWvAj51GKvzq/59Eana2fk3/nIHyFqs6Dz/AOWPzFvJtchk+C4YSr6AkaM70E1tKD4HbFX64rq2lNqj6IupWra1FbC8ewEyG6W2ZyizNDy5hCwoGpSu2KvyZ/59Lk/oX/nKkVNB+bd8QO1THv8AqxVZ/wA5LAL/AM/Tf+cKZE+GRvLOoqWGxK8Nd2r4fEfvxV+vOKuxV2KuxV2KuxV2KuxV2KuxV2KuxV2KuxV8u6//AM4u+X9f/wCcofKP/OUM3mnUrfXvKOgS6DFoaRQmymSRZlErSkeoCBOdh4DpvVV9EeY/L2i+bdA1ryt5j0+LVtA8xWM+najZzisc9tcxtFLG1KGjKxFRv4Yq+N/+cTP+cFPy/wD+cRPMn5ieYPJXmvWvMCefY7a3W01VYQlhb2ss8qRxvEqtIf31OTb0HjXFUy/5yL/5wj/L38/fN+g/mfZeZ9f/ACn/ADd8vJHBb+cPKc4t7yW2i58IbhDRZOPKgbZuPwVK7YqwTyj/AM4K+ZIfOXlHzf8Amt/zlV+Z35sx+StSt9V0/Rry9+oWD3FpIssH1hYHLSKGQclJAYVB2OKvVp/+cSvL2tf85RQ/85QecvOOq+cNY0DThp/lPy3ew266XoH7sIZrfivN5KtI4Ztwz1qeKcVUP/zkP/ziB5X/AD589/lb+alr5u1X8tfzL/Km9W403zDoccT3E9usiyi1nWX4SoYGh8GZTVWpirJf+coP+cXPI/8AzlT+Xtl5H87X9/o19ol/Fqmja/pZRL2wvI1KM6BgVKupIZT7EEFQcVQ//ONv/OPHmH8h4PM58zfnd5t/Oe/8xrYwxTeZpAY7C309Zlijtow70LCb4zXei+GKvp9lDKynowIP04q/KuP/AJ9dado/mHzprfkb/nJr8y/IFr521q61q807RJoIIfVuJZZVVioUv6YlKqzb0xV7B+Rv/Pvf8o/yf8+W35ra95i8zfnJ+Z+nFjp3mDzpefXJLEno8EIAQOtSFc1K1PGmKsw/5yf/AOcNPJP/ADkrqnk7zi/mnXPy1/M/yASuh+bfLkvC7hgZjIYJEYgMockqQQwqwrRiMVfPfnT/AJ9feWPzV8v3qfm/+fHn38y/zAme1Sx816zLFK+l2kLcprWysjWFRMac3NW2FPdV9+fmn+UvlH84/wArvM35R+d7aS+8q+atOTT7sRt6cymFo5YJ42ANJIpYUkU0pyUbYq+Yf+caf+cLdU/5x18x6bqB/wCchfPP5h+U/Lel3GlaF5U1iRI9MsorkpV/TRmDFApCAAKtdh0oqx384P8An3r5S88fmpqX52flb+aPmr8gvzH15XOs3flWUC21GZ+AaaaEslGbjV+LcWb4ivLfFXqn/ONv/OIXlX/nHjWPNXnSXzt5l/NH8zPOsCWer+avNF0ZrmS0ik9VLeKIErGgffqT03oMVTX/AJxi/wCcW/L/APzjDafmXaaB5q1HzQn5leaJvM9y2oRQxG2lmXj6MfogcgPE/dirvzC/5xb8vfmF/wA5IflD/wA5IXvmnUtP1z8odPuNPs9Hgiha0vEuFvBzlkYeopH1w9PAe+KvqTFXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq7FXYq//Z)
:::

</div>

Figure 2.7 -- One pass in a neural network

However, on a GPU, we can perform all these steps simultaneously,
meaning there is no requirement []{#_idIndexMarker076}for batch 1 to
finish before batch 2 can be started. We can calculate the parameter
updates for all batches simultaneously []{#_idIndexMarker077}and then
perform all the parameter updates in one go (as the results are
independent of one another). The parallel approach can vastly speed up
the machine learning process:

<div>

::: {#_idContainer066 .IMG---Figure}
![Figure 2.8 -- Parallel approach to perform passes
](data:application/octet-stream;base64,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SW8uYLie39P09gAEavVeuKvrLFXYq7FXYq7FXYqhfqNkLs34s4PrxQRm59NfVKDovOnKm/SuKorFUJDYWNtPc3VvZQQXV4QbiaONVklK1pzYAFqV2riqJVEQEIoQEliAKbk1J+nFXKiJyCIqBmLHiAKk9SadziqHvLGy1G3e01CzgvrWQgtDcRrLG1DUVVwQaHFVeOOOGNIokWKKJQqIgCqqgUAAGwAGKqVzZ2l4sa3drDdLDIs0YmRXCSIaq6hgaMD0OKqxRGZWZQzJUqSKkVFDQ/LFXMiPxLorlG5LUA0I7j33xVtlVlKsAysKEHcEHscVQtrYWNlbm0srKC0tSWPowxrHHVySx4KANyd8VRQAAAAoBsAO2KoX9H2H1w6j9St/0gU9L616a+tw/l9SnKntXFUX12O4OKpfZaTpemGZtN02109rlucxtoUiMjbmr8FFTv3xVMMVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVdirsVfm9+b/APz8r/LDyZ53ufyw/JzyJ5m/5yY/MPT+YvrDyLCbmytXjcI8TXcMV00jqWAb0YZEU/CzhqrirCrT/n5+vk/UtPt/+cj/APnFn8z/APnH3RdTnW3i17UrG4vdOR2/akeWy06QqB19JJG/ycVfqfa3Nve21veWkq3FrdxJNDKhqrxyKGVgfAg1xVXxV2KuxV2KvMvzr84ar+Xn5Nfm35/0KO3l1vyN5M17zBp6XaNJbtdaZp1xdQLMiPGzIXiHIBgSO4xV4b/zh3+e/nX88f8AnFHyh+dnnTSrS/8AOWrW+vS3FhoMDW8dy+lalf2sEVvFLLMQ8iWqjdqFj2G2Kso/5xV/Pfzf/wA5DflxqPnjzp+Tetfkhqllr93o8Wia280k1xBbRW8i3kbXFlp78C07Rn91TnG1CegVfS+KuxV2KuxVjvm7zZ5d8ieV9f8A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)
:::

</div>

Figure 2.8 -- Parallel approach to perform passes

**Compute Unified Device Architecture** (**CUDA**) is the
[]{#_idIndexMarker078}technology specific to Nvidia GPUs that enables
hardware acceleration on PyTorch. In order to enable CUDA, we
[]{#_idIndexMarker079}must first make sure the graphics card on our
system is CUDA-compatible. A list of
[]{#_idIndexMarker080}CUDA-compatible GPUs []{#_idIndexMarker081}can be
found here: <https://developer.nvidia.com/cuda-gpus>. If you have a
CUDA-compatible GPU, then CUDA can be installed from
[]{#_idIndexMarker082}this link:
<https://developer.nvidia.com/cuda-downloads>. We will activate it using
the following steps:

1.  Firstly, in order to actually enable CUDA support on PyTorch, you
    will have to build PyTorch from source. Details about how this can
    be done can be found here:
    https://github.com/pytorch/pytorch\#from-source.

2.  Then, to actually CUDA within our PyTorch code, we must type the
    following into our Python code:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    cuda = torch.device('cuda') 
    ```
    :::

    This sets our default CUDA device\'s name to `'cuda'`{.literal}.

3.  We can then execute operations on this device by manually specifying
    the device argument in any tensor operations:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    x = torch.tensor([5., 3.], device=cuda)
    ```
    :::

    Alternatively, we can do this by calling the `cuda`{.literal}
    method:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    y = torch.tensor([4., 2.]).cuda()
    ```
    :::

4.  We can then run a []{#_idIndexMarker083}simple operation to
    []{#_idIndexMarker084}ensure this is working correctly:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    x*y
    ```
    :::

    This results in the following output:

<div>

::: {#_idContainer067 .IMG---Figure}
![Figure 2.9 -- Tensor multiplication output using CUDA
](data:application/octet-stream;base64,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)
:::

</div>

Figure 2.9 -- Tensor multiplication output using CUDA

The changes in speed will not be noticeable at this stage as we are just
creating a tensor, but when we begin training models at scale later, we
will see the speed benefits of parallelizing our computations using
CUDA. By training our models in parallel, we will be able to reduce the
time this takes by a considerable amount.


Comparing PyTorch to other deep learning frameworks {#_idParaDest-35}
===================================================

::: {#_idContainer081}
PyTorch is one of the main frameworks used in deep learning today. There
are other widely used frameworks []{#_idIndexMarker085}available too,
such []{#_idIndexMarker086}as TensorFlow, Theano, and Caffe. While these
are very similar in many ways, there are some key differences in how
they operate. These include the following:

-   How the models are computed
-   The way in which the computational graphs are compiled
-   The ability to create dynamic computational graphs with variable
    layers
-   Differences in syntax 

Arguably, the main difference between PyTorch and other frameworks is in
the way that the models themselves are computed. PyTorch uses an
automatic differentiation []{#_idIndexMarker087}method called
**autograd**, which allows computational graphs to be defined and
executed dynamically. This is in contrast to other frameworks such as
TensorFlow, which is a static framework. In these static frameworks,
computational graphs must be defined and compiled before finally being
executed. While using pre-compiled models may lead to efficient
implementations in production, they do not offer the same level of
flexibility in research and explorational projects.

Frameworks such as PyTorch do not need to pre-compile computational
graphs before the model can be trained. The dynamic computational graphs
used by PyTorch mean that graphs are compiled as they are executed,
which allows graphs to be defined on the go. The dynamic approach to
model construction is particularly useful in the field of NLP. Let\'s
consider two []{#_idIndexMarker088}sentences that we wish to perform
sentiment analysis on:

<div>

::: {#_idContainer068 .IMG---Figure}
![Figure 2.10 -- Model construction in PyTorch
](data:application/octet-stream;base64,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)
:::

</div>

Figure 2.10 -- Model construction in PyTorch

We can represent each of these sentences as a sequence of individual
word vectors, which would then form our input to our neural network.
However, as we can see, each of our inputs is of a
[]{#_idIndexMarker089}different size. Within a
[]{#_idIndexMarker090}fixed computation graph, these varying input sizes
could be a problem, but for frameworks like PyTorch, models are able to
adjust dynamically to account for the variation in input structure. This
is one reason why PyTorch is often preferred for NLP-related deep
learning.

Another major difference between PyTorch and other deep learning
frameworks is syntax. PyTorch is often preferred by developers with
experience in Python as it is considered to be very Pythonic in nature.
PyTorch integrates well with other aspects of the Python ecosystem and
it is very easy to learn if you have prior knowledge of Python. We will
demonstrate PyTorch syntax now by coding up our own neural network from
scratch.


Building a simple neural network in PyTorch {#_idParaDest-36}
===========================================

::: {#_idContainer081}
We will now walk through building a neural network from scratch in
PyTorch. Here, we have a small `.csv`{.literal} file containing several
examples of images from the MNIST dataset. The MNIST dataset
[]{#_idIndexMarker091}consists of a collection of hand-drawn digits
between 0 and 9 that we want to attempt to classify. The following is an
example from the MNIST dataset, consisting of a hand-drawn digit 1:

<div>

::: {#_idContainer069 .IMG---Figure}
![Figure 2.11 -- Sample image from the MNIST dataset
](6_files/B12365_02_11.jpg)
:::

</div>

Figure 2.11 -- Sample image from the MNIST dataset

These images are 28x28 in size: 784 pixels in total. Our dataset in
`train.csv`{.literal} consists of 1,000 of these images, with each
consisting of 784 pixel values, as well as the correct classification of
the digit (in this case, 1).

[]{#_idTextAnchor036}

Loading the data {#_idParaDest-37}
----------------

We will begin by loading the data, as follows:

1.  First, we []{#_idIndexMarker092}need to load our training dataset,
    as follows:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    train = pd.read_csv("train.csv")
    train_labels = train['label'].values
    train = train.drop("label",axis=1).values.reshape(len(train),1,28,28)
    ```
    :::

    Notice that we reshaped our input to (`1,`{.literal} `1,`{.literal}
    `28,`{.literal} `28`{.literal}), which is a tensor of 1,000 images,
    each consisting of 28x28 pixels.

2.  Next, we convert our training data and training labels into PyTorch
    tensors so they can be fed into the neural network:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    X = torch.Tensor(train.astype(float))
    y = torch.Tensor(train_labels).long()
    ```
    :::

Note the data types of these two tensors. A float tensor comprises
32-bit floating-point numbers, while a long tensor consists of 64-bit
integers. Our `X`{.literal} features must be floats in order for PyTorch
to be able to compute gradients, while our labels must be integers
within this classification model (as we\'re trying to predict values of
1, 2, 3, and so on), so a prediction of 1.5 wouldn\'t make sense.

[]{#_idTextAnchor037}

Building the classifier {#_idParaDest-38}
-----------------------

Next, we can start to []{#_idIndexMarker093}construct our actual neural
network classifier:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 98)
        self.fc4 = nn.Linear(98, 10)
```
:::

We build our classifier as if we were building a normal class in Python,
inheriting from `nn.Module`{.literal} in PyTorch. Within our
`init`{.literal} method, we define each of the layers of our neural
network. Here, we define fully connected linear layers of varying sizes.

Our first layer takes **784** inputs as this is the size of each of our
images to classify (28x28). We then see that the output of one layer
must have the same value as the input of the next one, which means our
first fully connected layer outputs **392** units and our second layer
takes **392** units []{#_idIndexMarker094}as input. This is repeated for
each layer, with them having half the number of units each time until we
reach our final fully connected layer, which outputs **10** units. This
is the length of our classification layer.

Our network now looks something like this:

<div>

::: {#_idContainer070 .IMG---Figure}
![Figure 2.12 -- Our neural network ](6_files/B12365_02_12.jpg)
:::

</div>

Figure 2.12 -- Our neural network

Here, we can see that our final layer outputs **10** units. This is
because we wish to predict whether each image is a digit between 0 and
9, which is 10 different possible classifications in total. Our output
is a vector of length **10** and contains predictions for each of the 10
possible []{#_idIndexMarker095}values of the image. When making a final
classification, we take the digit classification that has the highest
value as the model\'s final prediction. For example, for a given
prediction, our model might predict the image is type 1 with a
probability of 10%, type 2 with a probability of 10%, and type 3 with a
probability of 80%. We would, therefore, take type 3 as the prediction
as it was predicted with the highest probability.

[]{#_idTextAnchor038}

Implementing dropout {#_idParaDest-39}
--------------------

Within the `init`{.literal} method []{#_idIndexMarker096}of our
`MNISTClassifier`{.literal} class, we also define a dropout method in
order to help regularize the network:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
self.dropout = nn.Dropout(p=0.2)
```
:::

Dropout is a way of regularizing our neural networks to prevent
overfitting. On each training epoch, for each node in a layer that has
dropout applied, there is a probability (here, defined as *p* = 20%)
that each node within the layer will not be used in
training/backpropagation. This means that when training, our network
becomes robust toward []{#_idIndexMarker097}overfitting since each node
will not be used in every iteration of the training process. This
prevents our network from becoming too reliant on predictions from
specific nodes within our network.

[]{#_idTextAnchor039}

Defining the forward pass {#_idParaDest-40}
-------------------------

Next, we define []{#_idIndexMarker098}the forward pass within our
classifier:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
```
:::

The `forward()`{.literal} method within our classifier is where we apply
our activation functions and define where dropout is applied within our
network. Our `forward`{.literal} method defines the path our input will
take through the network. It first takes our input, `x,`{.literal} and
reshapes it for use within the network, transforming it into a
one-dimensional vector. We then pass it through our first fully
connected layer and wrap it in a `ReLU`{.literal} activation function to
make it non-linear. We also wrap it in our dropout, as defined in our
`init`{.literal} method. We repeat this process for all the other layers
in the network.

For our final prediction layer, we wrap it in a log `softmax`{.literal}
layer. We will use this to easily calculate our
[]{#_idIndexMarker099}loss function, as we will see next.

[]{#_idTextAnchor040}

Setting the model parameters {#_idParaDest-41}
----------------------------

Next, we define []{#_idIndexMarker100}our model parameters:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
model = MNISTClassifier()
loss_function = nn.NLLLoss()
opt = optim.Adam(model.parameters(), lr=0.001)
```
:::

We initialize an instance of our `MNISTClassifier`{.literal} class as a
model. We also define []{#_idIndexMarker101}our loss as a **Negative Log
Likelihood Loss**:

Loss(y) = -log(y)

Let\'s assume our image is of a number 7. If we predict class 7 with
probability 1, our loss will be *-log(1) = 0*, but if we only predict
class 7 with probability 0.7, our loss will be *-log(0.7) = 0.3*. This
means that our loss approaches infinity the further away from the
correct prediction we are:

<div>

::: {#_idContainer071 .IMG---Figure}
![Figure 2.13 -- Representation of loss for our network
](6_files/B12365_02_13.jpg)
:::

</div>

Figure 2.13 -- Representation of loss for our network

This is then summed over all the correct classes in our dataset to
compute the total loss. Note that we defined a log
[]{#_idIndexMarker102}softmax when building the classifier as this
already applies a softmax function (restricting the predicted output to
be between 0 and 1) and takes the log. This means that *log(y)* is
already calculated, so all we need to do to compute the total loss on
the network is calculate the negative sum of the outputs.

We will also define our optimizer as an Adam optimizer. An optimizer
controls the **learning rate** within our model. The learning rate
[]{#_idIndexMarker103}of a model defines how big the parameter updates
are during each epoch of training. The larger the size of the learning
rate, the larger the size of the parameter updates during gradient
descent. An optimizer dynamically controls this learning rate so that
when a model is initialized, the parameter updates are large. However,
as the model learns and moves closer to the point where loss is
minimized, the optimizer controls the learning rate, so the parameter
updates become smaller and the local minimum can be located more
precisely.

[]{#_idTextAnchor041}

Training our network {#_idParaDest-42}
--------------------

Finally, we can actually start training our network:

1.  First, create a loop that runs once for each epoch of our training.
    Here, we will run our training []{#_idIndexMarker104}loop for 50
    epochs. We first take our input tensor of images and our output
    tensor of labels and transform them into PyTorch variables. A
    `variable`{.literal} is a PyTorch object that contains a
    `backward()`{.literal} method that we can use to perform
    backpropagation through our network:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    for epoch in range(50): 
        images = Variable(X)
        labels = Variable(y)
    ```
    :::
2.  Next, we call `zero_grad()`{.literal} on our optimizer to set our
    calculated gradients to zero. Within PyTorch, gradients are
    calculated cumulatively on each backpropagation. While this is
    useful in some models, such as when training RNNs, for our example,
    we wish to calculate the gradients from scratch after each epoch, so
    we make sure to reset the gradients to zero after each pass:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    opt.zero_grad()
    ```
    :::
3.  Next, we use our model\'s current state to make predictions on our
    dataset. This is effectively our forward pass as we then use these
    predictions to calculate our loss:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    outputs = model(images)
    ```
    :::
4.  Using the outputs and the true labels of our dataset, we calculate
    the total loss of our model using the defined loss function, which
    in this case is the negative log likelihood. On calculating this
    loss, we can then make a `backward()`{.literal} call to
    backpropagate our loss through the network. We then use
    `step()`{.literal} using our optimizer []{#_idIndexMarker105}in
    order to update our model parameters accordingly:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    loss = loss_function(outputs, labels)
    loss.backward()
    opt.step()
    ```
    :::
5.  Finally, after each epoch is complete, we print the total loss. We
    can observe this to make sure our model is learning:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    print ('Epoch [%d/%d] Loss: %.4f' %(epoch+1, 50,         loss.data.item()))
    ```
    :::

In general, we would expect the loss to decrease after every epoch. Our
output will look something like this:

<div>

::: {#_idContainer072 .IMG---Figure}
![Figure 2.14 -- Training epochs ](6_files/B12365_02_14.jpg)
:::

</div>

Figure 2.14 -- Training epochs

[]{#_idTextAnchor042}

Making predictions {#_idParaDest-43}
------------------

Now that our model has been trained, we can use this to make predictions
on unseen data. We begin by []{#_idIndexMarker106}reading in our test
set of data (which was not used to train our model):

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
test = pd.read_csv("test.csv")
test_labels = test['label'].values
test = test.drop("label",axis=1).values.reshape(len(test),                  1,28,28)
X_test = torch.Tensor(test.astype(float))
y_test = torch.Tensor(test_labels).long()
```
:::

Here, we perform the same steps we performed when we loaded our training
set of data: we reshape our test data and transform it into PyTorch
tensors. Next, to predict using our trained model, we simply run the
following command:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
preds = model(X_test)
```
:::

In the same way that we calculated our outputs on the forward pass of
our training data in our model, we now pass our test data through the
model and obtain predictions. We can view the predictions for one of the
images like so:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
print(preds[0])
```
:::

This results in the following output:

<div>

::: {#_idContainer073 .IMG---Figure}
![Figure 2.15 -- Prediction outputs ](6_files/B12365_02_15.jpg)
:::

</div>

Figure 2.15 -- Prediction outputs

Here, we can see that our prediction is a vector of length 10, with a
prediction for each of the possible classes (digits between 0 and 9).
The one with the highest predicted value is the one our model chooses as
its prediction. In this case, it is the 10[th]{.superscript} unit of our
vector, which []{#_idIndexMarker107}equates to the digit 9. Note that
since we used log softmax earlier, our predictions are logs and not raw
probabilities. To convert these back into probabilities, we can just
transform them using *x*.

We can now construct a summary DataFrame containing our true test data
labels, as well as the labels our model predicted:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
_, predictionlabel = torch.max(preds.data, 1)
predictionlabel = predictionlabel.tolist()
predictionlabel = pd.Series(predictionlabel)
test_labels = pd.Series(test_labels)
pred_table = pd.concat([predictionlabel, test_labels], axis=1)
pred_table.columns =['Predicted Value', 'True Value']
display(pred_table.head())
```
:::

This results in the following output:

<div>

::: {#_idContainer074 .IMG---Figure}
![Figure 2.16 -- Prediction table ](6_files/B12365_02_16.jpg)
:::

</div>

Figure 2.16 -- Prediction table

Note how the `torch.max()`{.literal} function automatically selects the
prediction with the highest value. We can see here that, based
[]{#_idIndexMarker108}on a small selection of our data, our model
appears to be making some good predictions!

[]{#_idTextAnchor043}

Evaluating our model {#_idParaDest-44}
--------------------

Now that we have some predictions from our model, we can use these
predictions to evaluate how good our model is. One
[]{#_idIndexMarker109}rudimentary way of evaluating model performance is
**accuracy**, as discussed in the previous chapter. Here, we simply
calculate our correct predictions (where the predicted image label is
equal to the actual image label) as a percentage of the total number of
predictions our model made:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
preds = len(predictionlabel)
correct = len([1 for x,y in zip(predictionlabel, test_labels)               if x==y])
print((correct/preds)*100)
```
:::

This results in the following output:

<div>

::: {#_idContainer075 .IMG---Figure}
![Figure 2.17 -- Accuracy score ](6_files/B12365_02_17.jpg)
:::

</div>

Figure 2.17 -- Accuracy score

Congratulations! Your first neural network was able to correctly
identify almost 90% of unseen digit images. As we progress, we will see
that there are more sophisticated models that []{#_idIndexMarker110}may
lead to improved performance. However, for now, we have demonstrated
that creating a simple deep neural network is very simple using PyTorch.
This can be coded up in just a few lines and leads to performance above
and beyond what is possible with basic machine learning models such as
regression.


NLP for PyTorch {#_idParaDest-45}
===============

::: {#_idContainer081}
Now that we have learned how to build neural networks, we will see how
it is possible to []{#_idIndexMarker111}build models for NLP using
PyTorch. In this example, we will create a basic bag-of-words classifier
in order to classify the language of a given sentence.

[]{#_idTextAnchor045}

Setting up the classifier {#_idParaDest-46}
-------------------------

For this example, we\'ll take a selection of sentences in Spanish and
English:

1.  First, we split each []{#_idIndexMarker112}sentence into a list of
    words and take the language of each sentence as a label. We take a
    section of sentences to train our model on and keep a small section
    to one side as our test set. We do this so that we can evaluate the
    performance of our model after it has been trained:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    ("This is my favourite chapter".lower().split(),\
     "English"),
    ("Estoy en la biblioteca".lower().split(), "Spanish")
    ```
    :::

    Note that we also transform each word into lowercase, which stops
    words being double counted in our bag-of-words. If we have the word
    `book`{.literal} and the word `Book`{.literal}, we want these to be
    counted as the same word, so we transform these into lowercase.

2.  Next, we build our word index, which is simply a dictionary of all
    the words in our corpus, and []{#_idIndexMarker113}then create a
    unique index value for each word. This can be easily done with a
    short `for`{.literal} loop:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    word_dict = {}
    i = 0
    for words, language in training_data + test_data:
        for word in words:
            if word not in word_dict:
                word_dict[word] = i
                i += 1
    print(word_dict)
    ```
    :::

    This results in the following output:

    ::: {#_idContainer076 .IMG---Figure}
    ![Figure 2.18 -- Setting up the classifier
    ](7_files/B12365_02_18.jpg)
    :::

    Figure 2.18 -- Setting up the classifier

    Note that here, we looped through all our training data and test
    data. If we just created our word index on training data, when it
    came to evaluating our test set, we would have new words that were
    not seen in the original training, so we wouldn\'t be able to create
    a true bag-of-words representation for these words.

3.  Now, we build our classifier in a similar fashion to how we built
    our neural network in the previous section; that is, by building a
    new class that inherits from `nn.Module`{.literal}.

    Here, we define our classifier so that it consists of a single
    linear layer with log softmax activation functions approximating a
    logistic regression. We could easily extend this to
    []{#_idIndexMarker114}operate as a neural network by adding extra
    linear layers here, but a single layer of parameters will serve our
    purpose. Pay close attention to the input and output sizes of our
    linear layer:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    corpus_size = len(word_dict)
    languages = 2
    label_index = {"Spanish": 0, "English": 1}
    class BagofWordsClassifier(nn.Module):  
        def __init__(self, languages, corpus_size):
            super(BagofWordsClassifier, self).__init__()
            self.linear = nn.Linear(corpus_size, languages)
        def forward(self, bow_vec):
            return F.log_softmax(self.linear(bow_vec), dim=1)
    ```
    :::

    The input is of length `corpus_size`{.literal}, which is just the
    total count of unique words in our corpus. This is because each
    input to our model will be a bag-of-words representation, consisting
    of the counts of words in each sentence, with a count of 0 if a
    given word does not appear in our sentence. Our output is of size 2,
    which is our number of languages to predict. Our final predictions
    will consist of a probability that our sentence is English versus
    the probability that our sentence is Spanish, with our final
    prediction being the one with the highest probability.

4.  Next, we define some utility functions. We first define
    `make_bow_vector`{.literal}, which takes the sentence and transforms
    it into a bag-of-words representation. We first create a vector
    consisting of all zeros. We then loop through them and for each
    []{#_idIndexMarker115}word in the sentence, we increment the count
    of that index within the bag-of-words vector by one. We finally
    reshape this vector using `with .view()`{.literal} for entry into
    our classifier:
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def make_bow_vector(sentence, word_index):
        word_vec = torch.zeros(corpus_size)
        for word in sentence:
            word_vec[word_dict[word]] += 1
        return word_vec.view(1, -1)
    ```
    :::

5.  Similarly, we define `make_target`{.literal}, which simply takes the
    label of the sentence (Spanish or English) and returns its relevant
    index (`0`{.literal} or `1`{.literal}):
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def make_target(label, label_index):
        return torch.LongTensor([label_index[label]])
    ```
    :::

6.  We can now create an instance of our model, ready for training. We
    also define our loss function as Negative Log Likelihood as we are
    using a log softmax function, and then define our optimizer in order
    to use standard **stochastic** **gradient** **descent** (**SGD**):
    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    model = BagofWordsClassifier(languages, corpus_size)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    ```
    :::

Now, we are ready to train our model.

[]{#_idTextAnchor046}

Training the classifier {#_idParaDest-47}
-----------------------

First, we set up a loop []{#_idIndexMarker116}consisting of the number
of epochs we wish our model to run for. In this instance, we will select
100 epochs.

Within this loop, we first zero our gradients (as otherwise, PyTorch
calculates gradients cumulatively) and then for each sentence/label
pair, we transform each into a bag-of-words vector and target,
respectively. We then calculate the predicted output of this particular
sentence pair by making a forward pass of our data through the current
state of our model.

Using this prediction, we then take our predicted and actual labels and
call our defined `loss_function`{.literal} on the two to obtain a
measure of loss for this sentence. By calling `backward()`{.literal}, we
then backpropagate this loss through our model and by calling
`step()`{.literal} on our optimizer, we update our model parameters.
Finally, we print our loss after every 10 training steps:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
for epoch in range(100):
    for sentence, label in training_data:
        model.zero_grad()
        bow_vec = make_bow_vector(sentence, word_dict)
        target = make_target(label, label_index)
        log_probs = model(bow_vec)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        
    if epoch % 10 == 0:
        print('Epoch: ',str(epoch+1),', Loss: ' +                         str(loss.item()))
```
:::

This results in the following output:

<div>

::: {#_idContainer077 .IMG---Figure}
![Figure 2.19 -- Training loss ](7_files/B12365_02_19.jpg)
:::

</div>

Figure 2.19 -- Training loss

Here, we can see that our []{#_idIndexMarker117}loss is decreasing over
time as our model learns. Although our training set in this example is
very small, we can still demonstrate that our model has learned
something useful, as follows:

1.  We evaluate our model on a couple of sentences from our test data
    that our model was not trained on. Here, we first set
    `torch.no_grad()`{.literal}, which deactivates the
    `autograd`{.literal} engine as there is no longer any need to
    calculate gradients as we are no longer training our model. Next, we
    take our test sentence and transform it into a bag-of-words vector
    and feed it into our model to obtain predictions.

2.  We then simply print the sentence, the true label of the sentence,
    and then the predicted probabilities. Note that we transform the
    predicted values from log probabilities back into probabilities. We
    obtain two probabilities for each prediction, but if
    []{#_idIndexMarker118}we refer back to the label index, we can see
    that the first probability (index 0) corresponds to Spanish, whereas
    the other one corresponds to English:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def make_predictions(data):
        with torch.no_grad():
            sentence = data[0]
            label = data[1]
            bow_vec = make_bow_vector(sentence, word_dict)
            log_probs = model(bow_vec)
            print(sentence)
            print(label + ':')
            print(np.exp(log_probs))
            
    make_predictions(test_data[0])
    make_predictions(test_data[1])
    ```
    :::

    This results in the following output:

    ::: {#_idContainer078 .IMG---Figure}
    ![Figure 2.20 -- Predicted output ](7_files/B12365_02_20.jpg)
    :::

    Figure 2.20 -- Predicted output

    Here, we can see that for both our predictions, our model predicts
    the correct answer, but why is this? What exactly has our model
    learned? We can see that our first test sentence contains the word
    `estoy`{.literal}, which was previously seen in a Spanish sentence
    within our training set. Similarly, we can see that the word
    `book`{.literal} was seen within our training set in an English
    sentence. Since our model []{#_idIndexMarker119}consists of a single
    layer, the parameters on each of our nodes are easy to interpret.

3.  Here, we define a function that takes a word as input and returns
    the weights on each of the parameters within the layer. For a given
    word, we get the index of this word from our dictionary and then
    select these parameters from the same index within the model. Note
    that our model returns two parameters as we are making two
    predictions; that is, the model\'s contribution to the Spanish
    prediction and the model\'s contribution to the English prediction:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.language-markup}
    def return_params(word): 
        index = word_dict[word]
        for p in model.parameters():
            dims = len(p.size())
            if dims == 2:
                print(word + ':')
                print('Spanish Parameter = ' +                    str(p[0][index].item()))
                print('English Parameter = ' +                    str(p[1][index].item()))
                print('\n')
                
    return_params('estoy')
    return_params('book')
    ```
    :::

    This results in the following output:

<div>

::: {#_idContainer079 .IMG---Figure}
![Figure 2.21 -- Predicted output for the updated function
](7_files/B12365_02_21.jpg)
:::

</div>

Figure 2.21 -- Predicted output for the updated function

Here, we can see that for []{#_idIndexMarker120}the word
`estoy`{.literal}, this parameter is positive for the Spanish prediction
and negative for the English one. This means that for each count of the
word \"`estoy`{.literal}\" in our sentence, the sentence becomes more
likely to be a Spanish sentence. Similarly, for the word
`book`{.literal}, we can see that this contributes positively to the
prediction that the sentence is English.

We can show that our model has only learned based on what it has been
trained on. If we try to predict a word the model hasn\'t been trained
on, we can see it is unable to make an accurate decision. In
[]{#_idIndexMarker121}this case, our model thinks that the English word
\"`not"`{.literal} is Spanish:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.language-markup}
new_sentence = (["not"],"English")
make_predictions(new_sentence)
```
:::

This results in the following output:

<div>

::: {#_idContainer080 .IMG---Figure}
![Figure 2.22 -- Final output ](7_files/B12365_02_22.jpg)
:::

</div>

Figure 2.22 --[]{#_idTextAnchor047} Final output

[]{#_idTextAnchor048}


Summary {#_idParaDest-48}
=======

::: {#_idContainer081}
In this chapter, we introduced PyTorch and some of its key features.
Hopefully, you now have a better understanding of how PyTorch differs
from other deep learning frameworks and how it can be used to build
basic neural networks. While these simple examples are just the tip of
the iceberg, we have illustrated that PyTorch is an immensely powerful
tool for NLP analysis and learning.

In future chapters, we will demonstrate how the unique properties of
PyTorch can be utilized to build highly sophisticated models for solving
very complex machine learning tasks.
