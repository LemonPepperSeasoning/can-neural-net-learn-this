# can-neural-net-learn-this

| Algorithm         | Loss  | Comment                                                                                                                     |
| ----------------- | ----- | --------------------------------------------------------------------------------------------------------------------------- |
| Sha256 Encryption | 0.25  | Not learning. Its just predicting 0.5 for each bit                                                                          |
| Sha256 Decryption | 0.25  | Same as Encryption. Not learning & predicting 0.5                                                                           |
| Identity function | 0.193 | After 0.208ish lr dramatically decreased. I feel it is learning but need some help to get over local minima                 |
| AND Gate          | 0.115 | Not any better. Loss is only lower because using Default mask bits, max loss is 0.25. ie, even at random loss will be 0.125 |

Removing dropout significantly improved the performance of the model. I think it makes sense. The aim is just to make NN learn simply polynimical function. There is no need to include drop-out to teach complex idea/correlation.
