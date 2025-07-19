# Food Category Recognition using Deep Learning (20-Class Subset)

This project is a deep learning-based food classification system built on a curated 20-class subset of the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/food-101/). It leverages **transfer learning** with a **ResNet18** model pretrained on **ImageNet**, and achieves a **test accuracy of 86.2%**. The goal is to build a fast, accurate, and robust model suitable for deployment and inclusion in machine learning portfolios.

---

## Project Highlights

- **20-Class Custom Dataset**: Filtered and extracted a high-quality subset from Food-101 for 20 diverse food categories.
- **Transfer Learning with ResNet18**: Used PyTorch to fine-tune a ResNet18 model pretrained on ImageNet for faster convergence and improved generalization.
- **Robust Training Pipeline**: Included custom data augmentations, image corruption filtering, class balancing, and real-time loss/accuracy tracking.
- **High Accuracy**: Achieved **86.2% test accuracy** with good generalization across unseen data.
- **GPU Acceleration**: All training was done using GPU for efficiency.

---

## Dataset

We used a **filtered 20-class subset** from Food-101, with structured directories:

```
food-20/
│
├── train/
│   ├── pizza/
│   ├── sushi/
│   └── ...
├── test/
│   ├── pizza/
│   ├── sushi/
│   └── ...
```

Selected classes include: `pizza`, `sushi`, `waffles`, `fried_rice`, `ice_cream`, `steak`, `spaghetti_bolognese`, `guacamole`, `ramen`, `grilled_cheese_sandwich`, `churros`, `macaroni_and_cheese`, `hot_dog`, `falafel`, `chicken_wings`, `french_fries`, `donuts`, `pancakes`, `tacos`, `caesar_salad`.

---

## Results

| Metric        | Value     |
|---------------|-----------|
| **Test Accuracy** | **86.2%** |
| **Test Loss**     | 0.4572    |
| **Model Used**    | ResNet18 (ImageNet Pretrained) |
| **Framework**     | PyTorch  |

---

## Model Weights

You can download the trained model (`best_model.pth`) using the link below:

[Download best_model.pth](https://drive.google.com/file/d/1AmPDZCmvrd5jL4lMfV8JYKPSfHhLQM_c/view?usp=sharing)

> Place the file in the project root directory before running predictions.

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements_food_recognition.txt
```

Main libraries:
- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `Pillow`
- `tqdm`

---

## Skills Demonstrated

- Data curation and cleaning for custom multi-class classification
- Transfer learning using pretrained CNNs
- GPU-accelerated model training and validation
- Image preprocessing and augmentation
- Performance monitoring and evaluation
- PyTorch-based deep learning model development
