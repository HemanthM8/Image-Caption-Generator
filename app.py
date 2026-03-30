#!/usr/bin/env python3
"""
Streamlit Image Caption Generator Interface
"""

import re
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Tuple


@st.cache_resource
def load_model():
    """Load BLIP model and processor (cached)."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


def extract_image_urls_from_markdown(markdown_content: str) -> List[str]:
    """
    Extract image URLs from markdown content.
    Handles both ![alt](url) and raw URLs in markdown.
    """
    image_urls = []
    
    # Find markdown image syntax: ![optional alt text](url)
    markdown_pattern = r'!\[.*?\]\((.*?)\)'
    urls = re.findall(markdown_pattern, markdown_content)
    image_urls.extend(urls)
    
    # Find direct URLs (http:// or https://)
    url_pattern = r'https?://[^\s\)\]\}]+'
    all_urls = re.findall(url_pattern, markdown_content)
    
    # Filter to only image URLs
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
    for url in all_urls:
        if url.lower().endswith(image_extensions) or '?' in url:
            # URL with query params might be an image
            if url not in image_urls:
                image_urls.append(url)
    
    return image_urls


def load_image_from_url(url: str) -> Image.Image | None:
    """Load an image from a URL."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        st.error(f"Error loading image from {url}: {e}")
        return None


def generate_caption(processor, model, image: Image.Image) -> str | None:
    """Generate a caption for an image using BLIP."""
    try:
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return None


def main():
    st.set_page_config(page_title="Image Caption Generator", layout="wide")
    
    st.title("🖼️ Image Caption Generator")
    st.markdown("Extract images from markdown and generate captions using BLIP")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Markdown File", "Paste Markdown Content"],
        horizontal=True
    )
    
    markdown_content = None
    
    if input_method == "Upload Markdown File":
        uploaded_file = st.file_uploader(
            "Upload a markdown file (.md)",
            type=["md", "txt"]
        )
        if uploaded_file is not None:
            markdown_content = uploaded_file.read().decode("utf-8")
    else:
        markdown_content = st.text_area(
            "Paste your markdown content here:",
            height=200,
            placeholder="Paste markdown text with image URLs...\n\n![image](https://example.com/image.jpg)"
        )
    
    if markdown_content:
        # Extract images
        st.subheader("📋 Extracting Images...")
        image_urls = extract_image_urls_from_markdown(markdown_content)
        
        if not image_urls:
            st.warning("No images found in the markdown.")
            return
        
        st.success(f"Found {len(image_urls)} image(s)")
        
        # Load model
        with st.spinner("Loading BLIP model..."):
            processor, model = load_model()
        
        # Generate captions for each image
        st.subheader("🎯 Image Captions")
        
        for idx, url in enumerate(image_urls, 1):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write(f"**Image {idx}**")
                image = load_image_from_url(url)
                if image:
                    st.image(image, width=300)
                else:
                    st.error("Failed to load image")
            
            with col2:
                st.write(f"**URL:** {url}")
                if image:
                    with st.spinner(f"Generating caption for image {idx}..."):
                        caption = generate_caption(processor, model, image)
                    if caption:
                        st.success(f"**Caption:** {caption}")
                    else:
                        st.error("Failed to generate caption")
                st.markdown("---")


if __name__ == "__main__":
    main()
