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

The idea of the additional explanation train step is to