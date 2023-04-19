## Bilateral-Knee-Network

- **_News (2023-4-15)_**: We release the [**Demo**](https://www.kaggle.com/calvchen/bilateral-knee-network-demo) for BikNet.
- **_News (2023-4-15)_**: We release the training codes of BikNet for radiographic osteoarthritis progression prediction.

### Network architectures

- BikNet

<img src="figures/architecture.jpg" alt="architecture" style="zoom:67%;" />

##### Reproduce

Run the training

```
python main.py --opt options/training/Bilateral.json
```

[Test Demo](https://www.kaggle.com/calvchen/bilateral-knee-network-demo) is also available in Kaggle. `<img src="figures/reader_test.jpg" alt="reader_test" style="zoom:80%;" />`

##### Results

![fig 3. model performance](figures/performance.jpg)
