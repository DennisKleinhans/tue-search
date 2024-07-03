### Model architecture
 - learned and shared embeddings
 - Fastformer transformer encoder
 - scaled softmax (softmax temperature also controls loss tau)
 - siamese dual encoder structure for query and document
 - batch padding with random documents
 - loss function: contrastive loss
 - evaluation metric: Mean Reciprocal Rank