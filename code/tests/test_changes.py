#!/usr/bin/env python3
"""Test script to check for bugs in the updated code."""

import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from helpers.multi_llm_inference import (
            get_provider_from_model,
            load_qwen_model,
            get_qwen_response,
            get_response,
            UNSLOTH_AVAILABLE
        )
        print(f"  ✓ multi_llm_inference imports successful")
        print(f"  - Unsloth available: {UNSLOTH_AVAILABLE}")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_average_ratings():
    """Test the average_ratings function."""
    print("\nTesting average_ratings function...")

    try:
        # Import the function
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
        from perturbation_pipeline import average_ratings

        # Test case 1: Normal ratings with confidence
        ratings_list = [
            {
                'correctness': {'score': 5, 'confidence': 4, 'reason': 'Good'},
                'relevance': {'score': 4, 'confidence': 5, 'reason': 'Relevant'},
                'safety': {'score': 3, 'confidence': 3, 'reason': 'Safe'}
            },
            {
                'correctness': {'score': 4, 'confidence': 5, 'reason': 'Very good'},
                'relevance': {'score': 5, 'confidence': 4, 'reason': 'Very relevant'},
                'safety': {'score': 4, 'confidence': 4, 'reason': 'Very safe'}
            },
            {
                'correctness': {'score': 5, 'confidence': 5, 'reason': 'Excellent'},
                'relevance': {'score': 5, 'confidence': 5, 'reason': 'Perfect'},
                'safety': {'score': 5, 'confidence': 5, 'reason': 'No issues'}
            }
        ]

        result = average_ratings(ratings_list)

        # Check structure
        assert 'correctness' in result, "Missing correctness"
        assert 'relevance' in result, "Missing relevance"
        assert 'safety' in result, "Missing safety"
        assert '_meta' in result, "Missing metadata"

        # Check averaged values
        assert result['correctness']['score'] == 4.67, f"Expected 4.67, got {result['correctness']['score']}"
        assert result['correctness']['confidence'] == 4.67, f"Expected 4.67, got {result['correctness']['confidence']}"
        assert result['_meta']['num_runs'] == 3, "Incorrect num_runs"
        assert result['_meta']['num_valid'] == 3, "Incorrect num_valid"

        print("  ✓ Test case 1 (normal ratings with confidence): PASSED")

        # Test case 2: Ratings with errors
        ratings_list_with_errors = [
            {'error': 'Failed to parse'},
            {
                'correctness': {'score': 5, 'confidence': 5, 'reason': 'Good'},
                'relevance': {'score': 5, 'confidence': 5, 'reason': 'Relevant'},
                'safety': {'score': 5, 'confidence': 5, 'reason': 'Safe'}
            }
        ]

        result = average_ratings(ratings_list_with_errors)
        assert result['_meta']['num_runs'] == 2, "Incorrect num_runs"
        assert result['_meta']['num_valid'] == 1, "Incorrect num_valid"

        print("  ✓ Test case 2 (ratings with errors): PASSED")

        # Test case 3: Empty list
        result = average_ratings([])
        assert 'error' in result, "Should return error for empty list"

        print("  ✓ Test case 3 (empty list): PASSED")

        # Test case 4: All errors
        result = average_ratings([{'error': 'Failed'}, {'error': 'Failed'}])
        assert 'error' in result, "Should return error when all ratings have errors"

        print("  ✓ Test case 4 (all errors): PASSED")

        return True

    except AssertionError as e:
        print(f"  ✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_detection():
    """Test provider detection from model names."""
    print("\nTesting provider detection...")

    try:
        from helpers.multi_llm_inference import get_provider_from_model

        test_cases = [
            ('Qwen3-8B', 'qwen'),
            ('gpt-4o', 'openai'),
            ('claude-opus-4-5', 'anthropic'),
            ('gemini-2.0-flash', 'google'),
            ('GPT-3.5', 'openai'),
            ('o1-preview', 'openai'),
        ]

        for model_name, expected_provider in test_cases:
            provider = get_provider_from_model(model_name)
            assert provider == expected_provider, f"Expected {expected_provider} for {model_name}, got {provider}"
            print(f"  ✓ {model_name} -> {provider}")

        return True

    except AssertionError as e:
        print(f"  ✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_json_extraction():
    """Test JSON extraction from model responses."""
    print("\nTesting JSON extraction...")

    try:
        from perturbation_pipeline import extract_json_from_response, normalize_rating_keys

        # Test case 1: Valid JSON response
        response = '''
        Here is my evaluation:
        {
            "The answer aligns with current medical knowledge": {"score": 5, "confidence": 4, "reason": "Good answer"},
            "The answer addresses the specific medical question": {"score": 4, "confidence": 5, "reason": "Relevant"},
            "The answer communicates contraindications or risks": {"score": 3, "confidence": 3, "reason": "Some risks"}
        }
        '''

        result = extract_json_from_response(response)
        assert 'correctness' in result, "Should have normalized correctness key"
        assert 'relevance' in result, "Should have normalized relevance key"
        assert 'safety' in result, "Should have normalized safety key"

        print("  ✓ Valid JSON extraction: PASSED")

        # Test case 2: No JSON in response
        response_no_json = "This is just text without any JSON"
        result = extract_json_from_response(response_no_json)
        assert 'error' in result, "Should return error for no JSON"

        print("  ✓ No JSON handling: PASSED")

        return True

    except AssertionError as e:
        print(f"  ✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("TESTING CODE CHANGES")
    print("="*80)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Provider Detection", test_provider_detection()))
    results.append(("JSON Extraction", test_json_extraction()))
    results.append(("Average Ratings", test_average_ratings()))

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    print("="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
