import cv2
import numpy as np

def extract_digits(area_img, margin=4):
    #find contours in the binary image
    contours, _ = cv2.findContours(area_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_imgs = []
    # sort contours by their x-coordinate
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    h_img, w_img = area_img.shape

    for x, y, w, h in bounding_boxes:
        if w > 6 and h > 8:  # Filter out small contours
            x0 = max(x - margin, 0)
            y0 = max(y - margin, 0)
            x1 = min(x + w + margin, w_img)
            y1 = min(y + h + margin, h_img)
            digit = area_img[y0:y1, x0:x1]
            digit_imgs.append(digit)
    return digit_imgs

def main():
    image = cv2.imread('img_data/cheque.jpg')
    
    if image is None:
        print("Error: Could not read the image.")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    area_numbers = binary[89:120, 555:696]
    gray_area_numbers = gray[89:120, 555:696]
    

    cv2.imshow('Area of Numbers', gray_area_numbers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extract digits from the area of numbers
    digits = extract_digits(area_numbers)
    #resize the digits to 28x28 pixels
    resized_digits = [cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA) for digit in digits]    
    #apply a morphological close operation to the digits
    
    kernel = np.ones((3, 3), np.uint8)
    resized_digits = [cv2.morphologyEx(digit, cv2.MORPH_CLOSE, kernel) for digit in resized_digits]

    
    for i, digit in enumerate(resized_digits):
        cv2.imshow(f'Digit {i+1}', digit)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    # Load the pre-trained CNN model for digit recognition
    from tensorflow.keras.models import load_model
    model = load_model('mnist_cnn_model.h5')

    # Prepare the digits for prediction
    digit_array = np.array([digit.reshape(28, 28, 1).astype('float32') / 255.0 for digit in resized_digits])
    predictions = model.predict(digit_array)
    predicted_classes = np.argmax(predictions, axis=1)

    # show the predicted classes
    date_str = ''.join(str(d) for d in predicted_classes)
    # Format the date string
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    print(f"This payment is from the date: {formatted_date}")

if __name__ == "__main__":
    main()
