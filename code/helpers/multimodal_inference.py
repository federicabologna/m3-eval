"""
Multimodal LLM inference with image support for WoundCare visual evaluation.
"""

import base64
import os
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Navigate up from helpers/ to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(code_dir)
env_path = os.path.join(project_root, '.env')

if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}", flush=True)
else:
    print(f"Warning: .env file not found at {env_path}", flush=True)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string
    """
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_woundcare_images(image_ids: List[str], images_dir: str = None, split: str = None) -> List[Dict]:
    """
    Load WoundCare images and prepare them for multimodal LLMs.

    Args:
        image_ids: List of image filenames (e.g., ['IMG_ENC0385_0001.jpg'])
        images_dir: Path to extracted images directory (optional)
        split: Dataset split ('test' or 'valid') - used to determine correct image directory

    Returns:
        List of image dicts with paths and base64 encodings
    """
    if images_dir is None:
        # Determine path based on split
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(script_dir)
        base_path = os.path.join(
            project_root,
            'data',
            'old',
            'woundcare',
            'dataset-challenge-mediqa-2025-wv'
        )

        # Test images are in nested structure, valid images are flat
        if split == 'valid':
            images_dir = os.path.join(base_path, 'images_valid')
        else:
            # Default to test (nested structure)
            images_dir = os.path.join(base_path, 'images_test', 'images_test')

    images = []
    for image_id in image_ids:
        image_path = os.path.join(images_dir, image_id)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Get file extension for media type
        ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/jpeg')

        images.append({
            'id': image_id,
            'path': image_path,
            'base64': encode_image_to_base64(image_path),
            'media_type': media_type
        })

    return images


def create_multimodal_message_openai(
    text: str,
    images: List[Dict],
    role: str = 'user'
) -> Dict:
    """
    Create a multimodal message for OpenAI API (GPT-4o, GPT-4 Turbo).

    Args:
        text: Text content
        images: List of image dicts with 'base64' and 'media_type'
        role: Message role ('user', 'assistant', 'system')

    Returns:
        Message dict in OpenAI format
    """
    if not images:
        # Text-only message
        return {'role': role, 'content': text}

    # Multimodal message with images
    content = [{'type': 'text', 'text': text}]

    for img in images:
        content.append({
            'type': 'image_url',
            'image_url': {
                'url': f"data:{img['media_type']};base64,{img['base64']}"
            }
        })

    return {'role': role, 'content': content}


def create_multimodal_message_anthropic(
    text: str,
    images: List[Dict],
    role: str = 'user'
) -> Dict:
    """
    Create a multimodal message for Anthropic API (Claude Opus 4.5, Sonnet 4.5).

    Args:
        text: Text content
        images: List of image dicts with 'base64' and 'media_type'
        role: Message role ('user', 'assistant')

    Returns:
        Message dict in Anthropic format
    """
    if not images:
        # Text-only message
        return {'role': role, 'content': text}

    # Multimodal message with images
    content = []

    # Add images first
    for img in images:
        content.append({
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': img['media_type'],
                'data': img['base64']
            }
        })

    # Add text after images
    content.append({'type': 'text', 'text': text})

    return {'role': role, 'content': content}


def create_multimodal_message_google(
    text: str,
    images: List[Dict],
    role: str = 'user'
) -> Dict:
    """
    Create a multimodal message for Google Gemini API.

    Args:
        text: Text content
        images: List of image dicts with 'base64' and 'media_type'
        role: Message role ('user', 'model')

    Returns:
        Message dict in Gemini format
    """
    # Gemini role mapping
    if role == 'assistant':
        role = 'model'

    if not images:
        # Text-only message
        return {'role': role, 'parts': [{'text': text}]}

    # Multimodal message with images
    parts = []

    # Add images
    for img in images:
        parts.append({
            'inline_data': {
                'mime_type': img['media_type'],
                'data': img['base64']
            }
        })

    # Add text
    parts.append({'text': text})

    return {'role': role, 'parts': parts}


def get_multimodal_response(
    text: str,
    images: List[Dict],
    model: str,
    system_message: Optional[str] = None,
    n: int = 1,
    return_all: bool = False
):
    """
    Get response from multimodal LLM with image support.

    Args:
        text: User prompt text
        images: List of image dicts with 'base64' and 'media_type'
        model: Model name (determines provider)
        system_message: Optional system message
        n: Number of completions to request (default: 1)
        return_all: If True, return list of all n completions

    Returns:
        String response (if return_all=False) or List of responses (if return_all=True)
    """
    # Import here to avoid circular dependency
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from multi_llm_inference import get_provider_from_model, get_response

    provider = get_provider_from_model(model)

    # Create multimodal message based on provider
    if provider == 'openai':
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append(create_multimodal_message_openai(text, images))

    elif provider == 'anthropic':
        messages = [create_multimodal_message_anthropic(text, images)]
        # System message is handled separately in get_response

    elif provider == 'google':
        messages = []
        if system_message:
            # Gemini handles system via system_instruction parameter
            messages.append({'role': 'system', 'content': system_message})
        messages.append(create_multimodal_message_google(text, images))

    else:
        # Fallback to text-only for unsupported models (like local Qwen)
        print(f"Warning: Model {model} may not support images. Using text-only mode.")
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': text})

    return get_response(messages, model, n=n, return_all=return_all)


# Example usage
if __name__ == '__main__':
    # Test image loading
    test_image_ids = ['IMG_ENC0385_0001.jpg']
    images = load_woundcare_images(test_image_ids)

    if images:
        print(f"Loaded {len(images)} images")
        print(f"First image: {images[0]['id']}")
        print(f"Path: {images[0]['path']}")
        print(f"Base64 length: {len(images[0]['base64'])} chars")
        print(f"Media type: {images[0]['media_type']}")

        # Test message creation
        print("\n" + "=" * 80)
        print("OpenAI format:")
        print(create_multimodal_message_openai("Describe this wound", images[:1]))

        print("\n" + "=" * 80)
        print("Anthropic format (truncated):")
        msg = create_multimodal_message_anthropic("Describe this wound", images[:1])
        msg_str = str(msg)[:500]
        print(msg_str + "...")

        print("\n" + "=" * 80)
        print("Gemini format (truncated):")
        msg = create_multimodal_message_google("Describe this wound", images[:1])
        msg_str = str(msg)[:500]
        print(msg_str + "...")
    else:
        print("No images loaded")
