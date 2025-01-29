# can-neural-net-learn-this

## Summary

| Algorithm             |  Success/Failure   | NN Description |  Loss   | Accuracy |
| :-------------------- | :----------------: | :------------: | :-----: | -------: |
| Identity function     | :white_check_mark: |   Defail MLP   |   TBD   |     100% |
| AND Gate              | :white_check_mark: |   Defail MLP   |   TBD   |     100% |
| OR Gate               | :white_check_mark: |   Defail MLP   |   TBD   |     100% |
| NOT Gate              | :white_check_mark: |   Defail MLP   |   TBD   |     100% |
| XOR Gate              | :white_check_mark: |   Defail MLP   |   TBD   |     100% |
| Shift right           | :white_check_mark: |   Defail MLP   |   TBD   |     100% |
| Sha256 Encryption     |        :x:         |   Defail MLP   | 57.7833 |      50% |
| Sha256 Decryption     |        :x:         |   Defail MLP   |   TBD   |      50% |
| Sha256 Step 1         | :white_check_mark: |   Defail MLP   |   TBD   |     100% |
| Sha256 Step 1 reverse | :white_check_mark: |   Defail MLP   |   TBD   |     100% |
| Sigma0Dataset         |        :x:         |   Defail MLP   |   TBD   |      50% |

## Description:

Loss: MSELoss

$$
\text{Accuracy} = \frac{\text{Number of correctly predicted bits}}{\text{Number of output bits}}
$$

## Appendix

```
Defail MLP:
MLP{
    hidden_sizes = [2048, 2048, 2048]
    activation = "relu"
    dropout = 0
    optimizer = 'adam'
    learning_rate = 0.001
}
```
