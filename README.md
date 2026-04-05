# Message-Embedding

This repository contains my experiments on generating proper message embeddings to be
able to search them semantically later.  

Personally, I am dissatisfied with how message search functionality is implemented with
messaging apps. So I tried building my own.

My initial plan was to embed each message with a [Sentence Transfomer](https://arxiv.org/abs/1908.10084),
however, that doesn't work well in practice since there are a lot of messages that does not contain
any meaningful semantics.

Instead, I opted to aggregate eight messages into a `context`, joined with special markers around
each message, and pulled their embeddings close to the individual messages they contain. This approach
yielded significantly better results.

Specifically, a `context` looks like this:
```
<user0>This is a message</user>
<user1>This is another message</user>
<user2>Yay another message :D</user>
<user3>I love talking</user>
<user4>Me too!</user>
<user5>Me three!</user>
<user6>I am running out of examples</user>
<user7>Shut up!</user>
```

Note: The `<userX>` tokens shown above are static, they don't contain any user related information.
That's just what I call them.

This works under the assumption is that a search query is semantically close to a message contained
by a context (thus close to the context itself) which is what a sentence transformer tries to do.

This repository contains the scripts to preprocess the data, train the model and test it. Training
script has Huggingface Accelerate support, so it should be able to utilize multiple GPUs, however I
only worked with single-GPU setups so it is not tested well. It also has MLFlow support to log training
metrics.

I tested Triplet loss as well as [InfoNCE](https://arxiv.org/abs/2407.00143) and [CLIP](https://arxiv.org/abs/2103.00020)
losses, triplet works the best. The final model I got has Top-1 score of around 66% and Top-8 score
of 90%.
