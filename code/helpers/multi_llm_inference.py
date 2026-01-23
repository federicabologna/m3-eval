import os
import torch
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import unsloth for faster inference
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: unsloth not installed. Install with: pip install unsloth")

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

    # Check for available device (prioritize CUDA > MPS > CPU)
    if torch.cuda.is_available():
        print(f"Using CUDA (GPU: {torch.cuda.get_device_name(0)})")
        torch_dtype = torch.float16  # Use half precision for faster inference on GPU

        # Use unsloth for faster inference on CUDA
        if UNSLOTH_AVAILABLE:
            print("Using unsloth for optimized inference with 4-bit quantization...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=f"Qwen/{model_name}",
                max_seq_length=4096,
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
            print("Unsloth not available, using standard transformers...")

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
        max_new_tokens=1000,
        temperature=1.0,
        do_sample=True,
        top_p=0.9
    )

    # Decode only the generated part (skip the input prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # print(f"\nModel response:\n{response}\n")

    return response


def get_response(messages, model="Qwen3-1.7B"):
    """
    Get response from various LLM providers.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Specific model name (defaults: Qwen3-1.7B)
              Provider is automatically deduced from model name

    Returns:
        String response from the model
    """

    # Deduce provider from model name
    provider = get_provider_from_model(model)

    if provider == "qwen":
        # Use local Qwen model
        return get_qwen_response(messages, model)

    elif provider == "openai":
        # OpenAI API (GPT models, O3, etc.)
        if model is None:
            model = "gpt-4o"  # Default model

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_tokens=1000
        )

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

        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=1.0,
            system=system_message,
            messages=anthropic_messages
        )

        return response.content[0].text

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
            'max_output_tokens': 1000
        }

        if system_instruction:
            config['system_instruction'] = system_instruction

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        return response.text

    else:
        raise ValueError(f"Unknown provider: {provider}")
