
# Drug Inventory and Supply Chain Management System

## Project Overview

This project aims to develop a software solution to effectively manage and track drug inventories and the supply chain. It ensures accurate and efficient drug distribution to guarantee that the right drugs reach the right people on time and under the correct conditions.

## Features
- **Tampering Detection:** Detects tampering or smudging on drug labels or packaging using a Convolutional Neural Network (CNN).
- **Text Extraction:** Uses Optical Character Recognition (OCR) to extract crucial information such as expiry date, quantity, and more from drug labels.
- **Expiry and Quantity Validation:** Ensures the medicine is not expired and its quantity falls within acceptable limits.

## Requirements

Install the following dependencies before running the project:

```
tensorflow==2.x
opencv-python
pytesseract
numpy
```

## File Structure

- `train/` : Training dataset (contains 'tampered' and 'not_tampered' image subfolders).
- `validation/` : Validation dataset (same structure as `train/`).
- `tampering_detection_model.h5` : Pre-trained tampering detection model.
- `main.py` : Main script to process medicine images and check for tampering, expiry, and quantity.

## How to Run

1. Install the dependencies listed above.
2. Organize your training data under `data/train/` and validation data under `data/validation/` directories, with subfolders `tampered/` and `not_tampered/` for each category.
3. Train the model by running the following command:

```python
python main.py
```

4. After training, the script will save the best-performing model as `tampering_detection_model.h5`.
5. To process a new image for tampering, expiry, and quantity validation, use:

```python
python main.py
```

## Example

```
image_path = 'medicine_lot_image.jpg'
max_quantity = 100
result = process_medicine(image_path, max_quantity)

if result == 1:
    print("Medicine is good")
elif result == 0:
    print("Medicine is not good (Expiry or quantity issue)")
else:
    print("Defective medicine (tampered or smudged)")
```

## Contributing

Feel free to contribute by forking this project and submitting pull requests.

## License

This project is licensed under the MIT License.
