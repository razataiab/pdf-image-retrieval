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

def classify_and_rename_images(output_dir, pdf_name, property_index, image_paths):
    for idx, image_path in enumerate(image_paths):
        image = os.path.basename(image_path)
        
        # Placeholder for image classification logic
        # Example logic (replace with actual image classification)
        if "main" in image.lower():
            new_name = f"{pdf_name}_property_{property_index}_main.jpg"
        elif "location" in image.lower():
            new_name = f"{pdf_name}_property_{property_index}_location.jpg"
        elif "external" in image.lower():
            new_name = f"{pdf_name}_property_{property_index}_external_{idx+1}.jpg"
        elif "internal" in image.lower():
            new_name = f"{pdf_name}_property_{property_index}_internal_{idx+1}.jpg"
        elif "floor" in image.lower() or "blueprint" in image.lower():
            new_name = f"{pdf_name}_property_{property_index}_floorplan_{idx+1}.jpg"
        else:
            new_name = f"{pdf_name}_property_{property_index}_unclassified_{idx+1}.jpg"
        
        new_path = os.path.join(output_dir, new_name)
        shutil.move(image_path, new_path)
        image_paths[idx] = new_path  # Update the path in the list
    print(f"Renamed image paths: {image_paths}")

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
    print(f"Image Embeddings: {embeddings}")
    
    return embeddings

def generate_text_embedding(query, model, processor):
    inputs = processor(text=query, return_tensors="pt")
    outputs = model.get_text_features(**inputs)
    text_embedding = outputs.detach().numpy().flatten()
    print(f"Text Embedding: {text_embedding}")
    return text_embedding

def process_pdf_and_update_db(pdf_path, destination_folder):
    output_dir = f"extracted_images_{os.path.splitext(os.path.basename(pdf_path))[0]}"
    output_dir = os.path.join(destination_folder, output_dir)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Extract and save images
    image_paths, image_count = extract_and_save_images(pdf_path, output_dir)
    classify_and_rename_images(output_dir, pdf_name, 1, image_paths)
    
    # Initialize CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Generate and print image embeddings
    embeddings = generate_image_embeddings(image_paths, model, processor)
    
    # Ensure number of embeddings matches number of images
    assert len(embeddings) == len(image_paths), f"Number of embeddings ({len(embeddings)}) does not match number of images ({len(image_paths)})"
    
    for idx, embedding in enumerate(embeddings):
        print(f"Embedding for image {idx + 1}:\n", embedding)
    
    return embeddings, image_paths, model, processor

def compare_text_and_image_embeddings(text_embedding, image_embeddings):
    # Compute cosine similarity between text embedding and each image embedding
    similarities = cosine_similarity([text_embedding], image_embeddings)
    return similarities

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
    embeddings, image_paths, model, processor = process_pdf_and_update_db(pdf_path, destination_folder)
    
    # Generate text embedding for the sample query
    sample_query = "Kitchen"
    text_embedding = generate_text_embedding(sample_query, model, processor)

    # Compute cosine similarity
    similarities = compare_text_and_image_embeddings(text_embedding, embeddings)

    # Find the index of the image with the highest similarity
    top_image_idx = np.argmax(similarities)
    top_image_path = image_paths[top_image_idx]

    # Print the similarity and top matching image
    print(f"Top matching image: {top_image_path} with similarity: {similarities[0][top_image_idx]}")

    # Save the top matching image to a separate folder
    top_image_dir = os.path.join(destination_folder, "top_matching_image")
    if not os.path.exists(top_image_dir):
        os.makedirs(top_image_dir)

    shutil.copy(top_image_path, top_image_dir)
    print(f"Top matching image saved to: {os.path.join(top_image_dir, os.path.basename(top_image_path))}")

else:
    print("No PDF file selected.")
