from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import os
import openai

app = FastAPI()

# Securely load the OpenAI API key from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the schema for input data
class ImageBase64Request(BaseModel):
    image_base64: str

# Helper function to decode a Base64 image
def decode_base64_image(base64_string):
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 image: {e}")

# Helper function to process the image (edge detection and contour counting)
def detect_edges_and_contours(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Perform Sobel Edge Detection
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = cv2.convertScaleAbs(sobel_edges)
    
    # Apply Otsu's Thresholding
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Perform Canny Edge Detection
    canny_edges = cv2.Canny(blurred, 100, 200)
    
    # Count contours using Adaptive Thresholding
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    
    # Convert processed images to Base64
    _, sobel_buffer = cv2.imencode('.jpg', sobel_edges)
    sobel_base64 = base64.b64encode(sobel_buffer).decode('utf-8')
    
    _, otsu_buffer = cv2.imencode('.jpg', otsu_thresh)
    otsu_base64 = base64.b64encode(otsu_buffer).decode('utf-8')
    
    _, adaptive_buffer = cv2.imencode('.jpg', adaptive_thresh)
    adaptive_base64 = base64.b64encode(adaptive_buffer).decode('utf-8')
    
    _, canny_buffer = cv2.imencode('.jpg', canny_edges)
    canny_base64 = base64.b64encode(canny_buffer).decode('utf-8')
    
    return {
        "contour_count": contour_count,
        "sobel_base64": sobel_base64,
        "otsu_base64": otsu_base64,
        "adaptive_base64": adaptive_base64,
        "canny_base64": canny_base64
    }

# Function to send the results to OpenAI for analysis
def send_to_openai(contour_count, sobel_base64, otsu_base64, adaptive_base64, canny_base64):
    try:
        # Define the prompt for OpenAI
        prompt = f"""
        Analyze the following data:
        - Contour Count: {contour_count}
        - Sobel Edge Image (Base64): {sobel_base64[:100]}... (truncated for brevity)
        - Otsu Threshold Image (Base64): {otsu_base64[:100]}... (truncated for brevity)
        - Adaptive Threshold Image (Base64): {adaptive_base64[:100]}... (truncated for brevity)
        - Canny Edge Image (Base64): {canny_base64[:100]}... (truncated for brevity)
        
        Based on the processed images and contour count, provide insights about the image, 
        including the type of objects detected, their possible characteristics, and any 
        suggestions for further analysis.
        """

        # Send the request to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}")

@app.post("/process-image/")
async def process_image_endpoint(data: ImageBase64Request):
    try:
        # Decode the Base64 image
        image = decode_base64_image(data.image_base64)
        
        # Process the image
        results = detect_edges_and_contours(image)
        
        # Send the results to OpenAI for analysis
        openai_analysis = send_to_openai(
            results["contour_count"],
            results["sobel_base64"],
            results["otsu_base64"],
            results["adaptive_base64"],
            results["canny_base64"]
        )
        
        # Return the results along with the OpenAI analysis
        return {
            "status": "success",
            "contour_count": results["contour_count"],
            "processed_images": {
                "sobel": results["sobel_base64"],
                "otsu": results["otsu_base64"],
                "adaptive": results["adaptive_base64"],
                "canny": results["canny_base64"],
            },
            "openai_analysis": openai_analysis
        }
    except HTTPException as e:
        return {"status": "error", "message": str(e.detail)}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}

@app.get("/")
async def root():
    return {"message": "Image Processing API with OpenAI Integration is live!"}
