#!/usr/bin/env python3
"""
Image Caption Generator using BLIP
Extracts images from markdown files and generates captions.
"""

import re
import sys
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration


def extract_image_urls_from_markdown(markdown_file):
    """
    Extract image URLs from a markdown file.
    Handles both ![alt](url) and raw URLs in markdown.
    """
    image_urls = []
    
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{markdown_file}' not found.")
        sys.exit(1)
    
    # Find markdown image syntax: ![optional alt text](url)
    markdown_pattern = r'!\[.*?\]\((.*?)\)'
    urls = re.findall(markdown_pattern, content)
    image_urls.extend(urls)
    
    # Find direct URLs (http:// or https://)
    url_pattern = r'https?://[^\s\)\]\}]+'
    all_urls = re.findall(url_pattern, content)
    
    # Filter to only image URLs
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
    for url in all_urls:
        if url.lower().endswith(image_extensions) or '?' in url:
            # URL with query params might be an image
            if url not in image_urls:
                image_urls.append(url)
    
    return image_urls


def load_image_from_url(url):
    """
    Load an image from a URL.
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None


def generate_caption(processor, model, image):
    """
    Generate a caption for an image using BLIP.
    """
    try:
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens = 40)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python caption_generator.py <markdown_file>")
        print("Example: python caption_generator.py sample_images/README.md")
        sys.exit(1)
    
    markdown_file = sys.argv[1]
    
    print(f"Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    print(f"Extracting images from {markdown_file}...")
    image_urls = extract_image_urls_from_markdown(markdown_file)
    
    if not image_urls:
        print("No images found in the markdown file.")
        return
    
    print(f"Found {len(image_urls)} image(s).\n")
    
    for idx, url in enumerate(image_urls, 1):
        print(f"[{idx}] Image URL: {url}")
        image = load_image_from_url(url)
        
        if image is None:
            print(f"    Caption: [Failed to load image]\n")
            continue
        
        caption = generate_caption(processor, model, image)
        if caption:
            print(f"    Caption: {caption}\n")
        else:
            print(f"    Caption: [Failed to generate caption]\n")


if __name__ == "__main__":
    main()
