import fitz  # PyMuPDF
import os
import shutil
import numpy as np
from tkinter import Tk, filedialog
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Hide the root window
Tk().withdraw()

def extract_and_save_images(pdf_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    document = fitz.open(pdf_path)
    image_count = 0
    image_paths = []
    
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image to disk
            image_path = os.path.join(output_dir, f"image_{page_num+1}_{img_index+1}.{image_ext}")
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            image_paths.append(image_path)
            image_count += 1
    
    print(f"Number of images extracted: {image_count}")
    print(f"Extracted image paths: {image_paths}")
    return image_paths, image_count

def generate_image_embeddings(image_paths, model, processor):
    embeddings = []
    for img_path in image_paths:
        try:
            print(f"Processing image for embedding: {img_path}")
            image = Image.open(img_path)
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.get_image_features(**inputs)
            embeddings.append(outputs.detach().numpy().flatten())
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    print(f"Total embeddings generated: {len(embeddings)}")
    return embeddings

def generate_text_embedding(query, model, processor):
    inputs = processor(text=query, return_tensors="pt")
    outputs = model.get_text_features(**inputs)
    text_embedding = outputs.detach().numpy().flatten()
    return text_embedding

def compare_text_and_image_embeddings(text_embedding, image_embeddings):
    # Compute cosine similarity between text embedding and each image embedding
    similarities = cosine_similarity([text_embedding], image_embeddings)
    return similarities

def classify_and_rename_images_based_on_keywords(output_dir, pdf_name, property_index, image_paths, image_embeddings, model, processor, keywords):
    matched_images = set()
    
    for keyword in keywords:
        text_embedding = generate_text_embedding(keyword, model, processor)
        similarities = compare_text_and_image_embeddings(text_embedding, image_embeddings)

        top_image_idx = np.argmax(similarities)
        top_image_path = image_paths[top_image_idx]

        # Check if this image is already matched with another keyword
        if top_image_path in matched_images:
            continue

        # Rename the top matching image based on the keyword
        new_name = f"{pdf_name}_property_{property_index}_{keyword.replace(' ', '_').lower()}.jpg"
        new_path = os.path.join(output_dir, new_name)
        shutil.move(top_image_path, new_path)
        image_paths[top_image_idx] = new_path  # Update the path in the list
        matched_images.add(new_path)

        print(f"Keyword '{keyword}': Top matching image renamed to {new_path}")

    # Remove unmatched images
    for img_path in image_paths:
        if img_path not in matched_images:
            os.remove(img_path)
            print(f"Removed unmatched image: {img_path}")

def process_pdf_and_update_db(pdf_path, destination_folder):
    output_dir = f"extracted_images_{os.path.splitext(os.path.basename(pdf_path))[0]}"
    output_dir = os.path.join(destination_folder, output_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Extract and save images
    image_paths, image_count = extract_and_save_images(pdf_path, output_dir)
    
    # Initialize CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Generate image embeddings
    image_embeddings = generate_image_embeddings(image_paths, model, processor)
    
    # Ensure number of embeddings matches number of images
    assert len(image_embeddings) == len(image_paths), f"Number of embeddings ({len(image_embeddings)}) does not match number of images ({len(image_paths)})"
    
    # Keywords to classify images
    keywords = ["main property image", "location map", "external view", "internal view", "floor plan"]
    
    # Classify and rename images based on keywords
    classify_and_rename_images_based_on_keywords(output_dir, pdf_name, 1, image_paths, image_embeddings, model, processor, keywords)

# Prompt the user to select a PDF file
pdf_path = filedialog.askopenfilename(
    title="Select a PDF file",
    filetypes=[("PDF files", "*.pdf")]
)

# Define destination folder
destination_folder = "/Users/razataiab/Desktop/syndx"

# Ensure destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Process the selected PDF
if pdf_path:
    process_pdf_and_update_db(pdf_path, destination_folder)
else:
    print("No PDF file selected.")
