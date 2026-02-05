import os
import sys
import time

# Print diagnostic info before importing torch
print("=" * 80, file=sys.stderr)
print("CUDA/PyTorch Diagnostics:", file=sys.stderr)
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}", file=sys.stderr)
print("=" * 80, file=sys.stderr)

import torch

# Print PyTorch info after successful import
print(f"PyTorch version: {torch.__version__}", file=sys.stderr)
print(f"PyTorch CUDA compiled version: {torch.version.cuda}", file=sys.stderr)
print(f"CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
print("=" * 80, file=sys.stderr)

# Try to import unsloth BEFORE transformers (required for optimizations)
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: unsloth not installed. Install with: pip install unsloth")

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global variables to cache the Qwen model and tokenizer
_qwen_model = None
_qwen_tokenizer = None
_using_unsloth = False


def get_provider_from_model(model):
    """Deduce provider from model name."""
    model_lower = model.lower()

    if model_lower.startswith(('gpt', 'o1', 'o3')):
        return 'openai'
    elif model_lower.startswith('claude'):
        return 'anthropic'
    elif model_lower.startswith('gemini'):
        return 'google'
    else:
        return 'qwen'


def load_qwen_model(model_name):
    """Load the Qwen model once and cache it globally."""
    global _qwen_model, _qwen_tokenizer, _using_unsloth

    if _qwen_model is not None and _qwen_tokenizer is not None:
        return _qwen_model, _qwen_tokenizer

    print(f"Loading Qwen model: {model_name}... This may take a few minutes on first run.")

    # Check if GPU supports unsloth (requires CUDA capability >= 7.0)
    use_unsloth = UNSLOTH_AVAILABLE
    if torch.cuda.is_available() and UNSLOTH_AVAILABLE:
        try:
            capability = torch.cuda.get_device_capability(0)
            if capability[0] < 7:
                print(f"WARNING: GPU has CUDA capability {capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]})")
                print("Unsloth requires CUDA capability >= 7.0. Disabling unsloth...")
                use_unsloth = False
        except:
            pass

    # Check for available device (prioritize CUDA > MPS > CPU)
    if torch.cuda.is_available():
        print(f"Using CUDA (GPU: {torch.cuda.get_device_name(0)})")
        torch_dtype = torch.float16  # Use half precision for faster inference on GPU

        # Use unsloth for faster inference on CUDA (only if GPU is compatible)
        if use_unsloth:
            print("Using unsloth for optimized inference with 4-bit quantization...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=f"Qwen/{model_name}",
                max_seq_length=8192,  # Increased to handle long prompts with examples
                dtype=torch_dtype,
                load_in_4bit=True,  # Use 4-bit quantization for faster inference
            )
            # Enable native 2x faster inference
            FastLanguageModel.for_inference(model)
            _using_unsloth = True
            _qwen_model = model
            _qwen_tokenizer = tokenizer
            print("Model loaded successfully with unsloth (4-bit)!\n")
            return model, tokenizer
        else:
            print("Using standard transformers with CUDA...")

    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        torch_dtype = torch.float16
    else:
        print("Using CPU (Warning: This will be very slow!)")
        torch_dtype = torch.float32

    # Load tokenizer (standard transformers)
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}")

    # Load model (standard transformers)
    model = AutoModelForCausalLM.from_pretrained(
        f"Qwen/{model_name}",
        torch_dtype=torch_dtype,
        device_map="auto"  # Automatically map to available devices
    )

    _qwen_model = model
    _qwen_tokenizer = tokenizer

    print("Model loaded successfully!\n")
    return model, _qwen_tokenizer


def get_qwen_response(messages, model_name):
    """Get response from local Qwen model."""
    # Load model (will use cached version if already loaded)
    model, tokenizer = load_qwen_model(model_name)

    # Determine device (prioritize CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=1.0,
        do_sample=True,
        top_p=0.9
    )

    # Decode only the generated part (skip the input prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # print(f"\nModel response:\n{response}\n")

    return response


def get_response_with_averaging(messages, model="Qwen3-1.7B", n=5, parse_fn=None):
    """
    Get n responses and return all for averaging.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name
        n: Number of completions to request (default: 5)
        parse_fn: Optional function to parse each response before returning

    Returns:
        List of n responses (parsed if parse_fn provided)
    """
    responses = get_response(messages, model, n=n, return_all=True)

    if parse_fn:
        return [parse_fn(r) for r in responses]
    else:
        return responses


def get_response(messages, model="Qwen3-1.7B", n=1, return_all=False):
    """
    Get response from various LLM providers.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Specific model name (defaults: Qwen3-1.7B)
              Provider is automatically deduced from model name
        n: Number of completions to request (default: 1)
           For OpenAI: Uses n parameter (single API call with n completions)
           For other providers: Makes n separate calls
        return_all: If True, return list of all n completions
                   If False, return first completion (default)

    Returns:
        String response (if return_all=False) or List of responses (if return_all=True)
    """

    # Deduce provider from model name
    provider = get_provider_from_model(model)

    if provider == "qwen":
        # Use local Qwen model (doesn't support n natively, make multiple calls)
        if n == 1:
            return get_qwen_response(messages, model)
        else:
            responses = [get_qwen_response(messages, model) for _ in range(n)]
            if return_all:
                return responses
            else:
                return responses[0]

    elif provider == "openai":
        # OpenAI API (GPT models, O3, etc.)
        if model is None:
            model = "gpt-4o"  # Default model

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_tokens=2048,
            n=n  # Request n completions in single API call
        )

        if return_all:
            # Return all n completions
            return [choice.message.content for choice in response.choices]
        else:
            # Return first completion
            return response.choices[0].message.content

    elif provider == "anthropic":
        # Anthropic API (Claude models)
        if model is None:
            model = "claude-opus-4-5-20251101"  # Claude Opus 4.5

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Convert messages format for Anthropic
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                anthropic_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

        # Anthropic doesn't support n parameter, so make multiple calls if needed
        responses = []
        for _ in range(n):
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=1.0,
                system=system_message,
                messages=anthropic_messages
            )
            responses.append(response.content[0].text)

        if return_all:
            return responses
        else:
            return responses[0]

    elif provider == "google":
        # Google Gemini API (new google.genai)
        if model is None:
            model = "gemini-2.0-flash-exp"  # Default Gemini model

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # Convert messages to Gemini format
        system_instruction = None
        contents = []

        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = msg['content']
            elif msg['role'] == 'user':
                contents.append({'role': 'user', 'parts': [{'text': msg['content']}]})
            elif msg['role'] == 'assistant':
                contents.append({'role': 'model', 'parts': [{'text': msg['content']}]})

        # Build config
        config = {
            'temperature': 1.0,
            'max_output_tokens': 2048
        }

        if system_instruction:
            config['system_instruction'] = system_instruction

        # Google doesn't support n parameter, so make multiple calls if needed
        responses = []
        for _ in range(n):
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            responses.append(response.text)

        if return_all:
            return responses
        else:
            return responses[0]

    else:
        raise ValueError(f"Unknown provider: {provider}")
