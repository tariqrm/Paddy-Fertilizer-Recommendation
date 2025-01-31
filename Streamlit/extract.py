import fitz  # PyMuPDF
import easyocr
import numpy as np
import cv2

# Initialize the easyocr reader for Sinhala
reader = easyocr.Reader(['si'])  # 'si' is the language code for Sinhala

# Function to extract text using easyocr
def extract_text_with_easyocr(image):
    result = reader.readtext(image)
    extracted_text = []
    for detection in result:
        extracted_text.append(detection[1])
    return " ".join(extracted_text)

# Function to extract numbers and hyphens from text
def extract_numbers_and_hyphens(text):
    parts = text.split()
    result = []
    for part in parts:
        if part == '-':
            result.append(part)
        else:
            try:
                result.append(float(part))
            except ValueError:
                continue
    return result

# Path to the PDF file
pdf_path = "Badulla_Book.pdf"

# Open the PDF file
document = fitz.open(pdf_path)

# Iterate through each page
for page_number in range(len(document)):
    # Extract the page
    page = document.load_page(page_number)
    
    # Convert the page to an image
    page_image = page.get_pixmap()
    
    # Convert the image to a numpy array
    page_image_array = np.frombuffer(page_image.samples, dtype=np.uint8).reshape(page_image.height, page_image.width, page_image.n)
    
    # Convert the numpy array to an image format compatible with OpenCV
    page_image_cv2 = cv2.cvtColor(page_image_array, cv2.COLOR_BGR2RGB)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(page_image_cv2, cv2.COLOR_BGR2GRAY)
    
    # Use thresholding to binarize the image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Detect contours (tables)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    extracted_values = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:  # Adjust based on expected table cell size
            cell_image = page_image_cv2[y:y+h, x:x+w]
            cell_text = extract_text_with_easyocr(cell_image)
            extracted_values.extend(extract_numbers_and_hyphens(cell_text))
    
    # Print the page number and the extracted numbers and hyphens
    print(f"Page {page_number + 1}: {extracted_values}")
