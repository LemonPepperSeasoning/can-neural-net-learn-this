# can-neural-net-learn-this

Tldr: Teaching MLP to learn Logic-Gate-Functions seems really difficult. I need to figure out other strategies.

UPDATE: The network just needs to be wide. For some reason, if network is wider than input layer, it performs significantly better.

Below is the loss for Sha256 Decryption dataset. The loss is going down. It still very close to 0.25, which means its still very similar to making a random guess but... the results show there seems to be a pattern that Network is learning.

```
0.25084131956100464
Epoch 1/500, Loss: 64.9381
0.2508728504180908
Epoch 2/500, Loss: 64.1592
0.2504434585571289
Epoch 3/500, Loss: 64.1227
0.25021111965179443
Epoch 4/500, Loss: 64.0881
0.25017598271369934
Epoch 5/500, Loss: 64.0699
0.2502148449420929
Epoch 6/500, Loss: 64.0595
0.250180184841156
Epoch 7/500, Loss: 64.0541
0.25015050172805786
Epoch 8/500, Loss: 64.0506
```

| Algorithm         | Loss  | Comment                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Sha256 Encryption | 0.25  | Not learning. Its just predicting 0.5 for each bit                                                                                                                                                                                                                                                                                                                                                           |
| Sha256 Decryption | 0.25  | Same as Encryption. Not learning & predicting 0.5                                                                                                                                                                                                                                                                                                                                                            |
| Identity function | 0.007 | Learns if the network is wide.                                                                                                                                                                                                                                                                                                                                                                               |
| AND Gate          | 0.001 | Was barely learning with Deep network. When i decrease it to 3 layers ([2048, 2048, 2048]), the loss improvement rate significantly improved. The Loss improvement rate & the local minima significantly depends on the number of layer & size of each layer. For example, very thin & deep layer is very bad & wont reach 0.12 loss. Thin & shallow layer also is bad. [ 64 ]\*3 reaches 0.09 but thats it. |

Removing dropout significantly improved the performance of the model. I think it makes sense. The aim is just to make NN learn simply polynimical function. There is no need to include drop-out to teach complex idea/correlation.

## Sigma0 function

The loss is definitly decreasing over time but its very minimal.
