==============
Design Choices
==============

This document contains some additional reasoning and explanations of some concepts related to the
network architecture.

Explanation Training Step
=========================

By default, the network architecture implements attention-like explanations for edges as well as nodes.
These are directly returned by the network as tensors called ``node_importances`` and ``edge_importances``.
They contain values between 0 and 1 for each edge/node and for each channel.

Theoretically, this would already be enough to create explanations. Training the network on any kind of
graph classification or regression task should cause these importance values to naturally sparsify forming
a kind of priority mask, which can be interpreted as an explanation.

This network however introduces an additional *explanation training step* to the training process of the
model, which is executed for every batch right before the "default" training step is performed:

.. code-block:: text

    def train_step():
        explanation_train_step()
        prediction_train_step()

Why do we need this additional train step ?
-------------------------------------------

As mentioned, the presence of the attention mechanism should itself be enough for the attention tensors to
naturally evolve into priority masks. This usually works, but we face a problem due to the multiple
explanation channels: We want each channel to have a very specific meaning. In the classfication example
we want each channel to provide an explanation about what parts of the graph speak *specifically for that
class*. For regression a common example would be that we would like two channels, where one channel explains
which parts of the graph speak for a high output value and which parts speak for a low output value
(relatively to the center of the expected value range).

With the presented, general network architecture the problem is that the network is in no way guided to
adopt this desired behavior. By adding an additional dense MLP on top of these pooled graph embeddings the
network can learn any nonlinear mapping between the separate importance channels and the final output value.
Due to this additional output mapping, the network is able to allocate any kind of explanation to an
arbitrary channel.

This is in fact the behavior that we repeatedly observed when working without the additional explanation
training step: Explanations were extremely inconsistent. Sometimes the channels contained principally the
correct explanations but the order was permutated relative to the expected labels. Sometimes explanations
for different classes were combined into one channel, while another did not contain anything. But sometimes
the explanations were actually correct. Nonetheless, such inconsistent behavior would be of no use for
unknown tasks, where correctness of explanations could not be checked by comparing to some known ground
truth.

Why not remove the final MLP then?
----------------------------------

As previously motivated, the final MLP layer is most likely the main reason for this inconsistency of
explanations. However, it is not the only one. Even without any additional final mapping whatsoever,
especially the flipping of the order of explanations can still occur due to other reasons. We suspect for
example the presence of bias weights for the node importance layer or certain configurations of negative
values in the final node embeddings to be among possible reasons for this flipping to occur.

By carefully constructing the final layers of the network to take all those considerations into account, we
were able to create a network which delivered consistent explanations which were mostly in line with
expectations. But all these measures represent severe limitations to the expressiveness of the network and
have a large negative impact on the prediction performance of the model to such a degree that the network
becomes essentially unviable for any real application.

Thus, we opted to keep the architecture as general as possible and to address the problem of the
explanations by directly incentivizing the network to adhere to the expectations regarding the explanations
via a parallel training objective.

How the explanation train step works
------------------------------------

The idea of the additional explanation train step is to use a very simplified and restricted version of the
network to train the attention part. Afterwards, the actual prediction training step can make full use of
the final MLP to predict the target values.

For this simplified version we perform essentially the same weighted global pooling step, only under the
assumption that the GAT part of the network always outputs a constant value of 1 as the node embedding of
every node. This is basically the same as pooling just the importance values themselves without any
additional information from the preceding GAT layers. Doing it this way the network is forced to
approximately solve the task at hand using *only the attention mechanism*. For example it has to do it's
best to make a classification for or against a certain class solely on the basis of selecting nodes which
are important for this class association versus de-prioritizing those that are not.

How exactly this explanation training step is structured depends on whether the model is supposed to be
doing classification or regression.

**CLASSIFICATION:**

For classification the same one-hot encoded vectors of ground truth class labels are used as the target.
The prediction is created by applying a shifted sigmoid activation on the previously described pooled
graph embeddings of each channel. A BCE loss is then used for training.

**REGRESSION:**

For regression we first need a-priori knowledge about the range of values to be expected from the dataset.
From that we can deduce the center of that value range. By default we map a regression problem to be
composed of 2 channels: One which explains high values and one which explains low values (relative to the
center of the expected value range). We then set up two separate regression problems by only considering the
absolute distance from this center value. One of the channels is then only trained on all the samples which
have values below the center value and the other is only trained with the samples which have original values
above this center value.

.. code-block:: text

    lo_true = abs(y_true - center) where y_true < center
    hi_true = abs(y_true - center) where y_true >= center

    reg_true = concat(lo_true, hi_true)

The predictions of the network are computed by applying a relu activation on the previously described
simplified graph embeddings of each channel and then using MSE loss to train on the vector of
constructed regression values ``reg_true``.

Why do we use this shifted sigmoid for the classification training step instead of softmax?
-------------------------------------------------------------------------------------------

During development we noticed that the softmax activation itself was also a possible problem that decreased
the quality of the explanations. The softmax operation is also a point that introduces a cross dependency
between the individual channels, which may lead to certain explanations appearing in a channel to which the
should not belong according to expectations. You can imagine this is the case, because the same output of
a softmax operation can equally be achieved by increasing the value of one component as well as by reducing
the values of all other components.

Instead we use a shifted version of sigmoid activation:

.. code-block:: text

    def shifted_sigmoid(x, multiplier=10, shift=10):
        return sigmoid(multiplier * x - shift)

    shifted_sigmoid(0) ~= 0.03
    shifted_sigmoid(2) ~= 0.97

Since the important values are between 0 and 1, the global sum pooling will always be a positive value. This
is why we cannot use a normal sigmoid here because sigmoid(0) is only 0.5 and we would not be able to use
the full value range.
