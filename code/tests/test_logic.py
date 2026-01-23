#!/usr/bin/env python3
"""Test the logic of averaging ratings without full imports."""

def average_ratings(ratings_list):
    """Test version of average_ratings function."""
    if not ratings_list:
        return {"error": "No valid ratings to average"}

    # Filter out any ratings with errors
    valid_ratings = [r for r in ratings_list if "error" not in r]

    if not valid_ratings:
        return {"error": "No valid ratings to average", "all_ratings": ratings_list}

    # Initialize averaged rating
    averaged = {}

    # Average each dimension
    for dimension in ['correctness', 'relevance', 'safety']:
        if dimension not in valid_ratings[0]:
            continue

        # Check if ratings have confidence scores
        has_confidence = isinstance(valid_ratings[0][dimension], dict) and 'confidence' in valid_ratings[0][dimension]

        if has_confidence:
            # Average both score and confidence
            avg_score = sum(r[dimension]['score'] for r in valid_ratings) / len(valid_ratings)
            avg_confidence = sum(r[dimension]['confidence'] for r in valid_ratings) / len(valid_ratings)

            # Collect all reasons for reference
            all_reasons = [r[dimension]['reason'] for r in valid_ratings]

            averaged[dimension] = {
                'score': round(avg_score, 2),
                'confidence': round(avg_confidence, 2),
                'reason': all_reasons[0],  # Use first reason as representative
                'all_reasons': all_reasons  # Keep all reasons for reference
            }
        else:
            # Old format - just average scores
            avg_score = sum(r[dimension]['score'] for r in valid_ratings) / len(valid_ratings)
            averaged[dimension] = {
                'score': round(avg_score, 2),
                'reason': valid_ratings[0][dimension].get('reason', 'N/A')
            }

    # Add metadata about the averaging
    averaged['_meta'] = {
        'num_runs': len(ratings_list),
        'num_valid': len(valid_ratings)
    }

    return averaged


def test_average_ratings():
    """Test the average_ratings function."""
    print("\nTesting average_ratings function...")

    passed = 0
    total = 0

    # Test case 1: Normal ratings with confidence (5 runs)
    total += 1
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
        },
        {
            'correctness': {'score': 3, 'confidence': 3, 'reason': 'Okay'},
            'relevance': {'score': 4, 'confidence': 4, 'reason': 'Good'},
            'safety': {'score': 4, 'confidence': 4, 'reason': 'Fine'}
        },
        {
            'correctness': {'score': 4, 'confidence': 4, 'reason': 'Solid'},
            'relevance': {'score': 5, 'confidence': 5, 'reason': 'Spot on'},
            'safety': {'score': 5, 'confidence': 5, 'reason': 'All good'}
        }
    ]

    result = average_ratings(ratings_list)

    try:
        assert 'correctness' in result, "Missing correctness"
        assert 'relevance' in result, "Missing relevance"
        assert 'safety' in result, "Missing safety"
        assert '_meta' in result, "Missing metadata"

        # Check averaged values (5+4+5+3+4)/5 = 4.2
        expected_correctness = 4.2
        assert result['correctness']['score'] == expected_correctness, \
            f"Expected {expected_correctness}, got {result['correctness']['score']}"

        # Check confidence (4+5+5+3+4)/5 = 4.2
        expected_conf = 4.2
        assert result['correctness']['confidence'] == expected_conf, \
            f"Expected {expected_conf}, got {result['correctness']['confidence']}"

        assert result['_meta']['num_runs'] == 5, f"Expected 5 runs, got {result['_meta']['num_runs']}"
        assert result['_meta']['num_valid'] == 5, f"Expected 5 valid, got {result['_meta']['num_valid']}"

        assert len(result['correctness']['all_reasons']) == 5, "Should have 5 reasons"

        print(f"  ✓ Test 1 (5 normal ratings with confidence): PASSED")
        print(f"    - Correctness: score={result['correctness']['score']}, confidence={result['correctness']['confidence']}")
        print(f"    - Relevance: score={result['relevance']['score']}, confidence={result['relevance']['confidence']}")
        print(f"    - Safety: score={result['safety']['score']}, confidence={result['safety']['confidence']}")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ Test 1 FAILED: {e}")

    # Test case 2: Ratings with some errors
    total += 1
    ratings_list_with_errors = [
        {'error': 'Failed to parse'},
        {
            'correctness': {'score': 5, 'confidence': 5, 'reason': 'Good'},
            'relevance': {'score': 5, 'confidence': 5, 'reason': 'Relevant'},
            'safety': {'score': 5, 'confidence': 5, 'reason': 'Safe'}
        },
        {'error': 'Network timeout'},
        {
            'correctness': {'score': 3, 'confidence': 3, 'reason': 'Okay'},
            'relevance': {'score': 3, 'confidence': 3, 'reason': 'Fine'},
            'safety': {'score': 3, 'confidence': 3, 'reason': 'Good'}
        },
        {
            'correctness': {'score': 4, 'confidence': 4, 'reason': 'Nice'},
            'relevance': {'score': 4, 'confidence': 4, 'reason': 'Good'},
            'safety': {'score': 4, 'confidence': 4, 'reason': 'Safe'}
        }
    ]

    result = average_ratings(ratings_list_with_errors)
    try:
        assert result['_meta']['num_runs'] == 5, "Incorrect num_runs"
        assert result['_meta']['num_valid'] == 3, f"Expected 3 valid, got {result['_meta']['num_valid']}"
        # (5+3+4)/3 = 4.0
        assert result['correctness']['score'] == 4.0, f"Expected 4.0, got {result['correctness']['score']}"
        print(f"  ✓ Test 2 (ratings with errors): PASSED")
        print(f"    - Filtered out 2 errors, averaged 3 valid ratings")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ Test 2 FAILED: {e}")

    # Test case 3: Empty list
    total += 1
    result = average_ratings([])
    try:
        assert 'error' in result, "Should return error for empty list"
        print(f"  ✓ Test 3 (empty list): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ Test 3 FAILED: {e}")

    # Test case 4: All errors
    total += 1
    result = average_ratings([{'error': 'Failed'}, {'error': 'Failed'}])
    try:
        assert 'error' in result, "Should return error when all ratings have errors"
        print(f"  ✓ Test 4 (all errors): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ Test 4 FAILED: {e}")

    # Test case 5: Old format without confidence
    total += 1
    old_format_ratings = [
        {
            'correctness': {'score': 5, 'reason': 'Good'},
            'relevance': {'score': 4, 'reason': 'Relevant'},
            'safety': {'score': 3, 'reason': 'Safe'}
        },
        {
            'correctness': {'score': 3, 'reason': 'Okay'},
            'relevance': {'score': 4, 'reason': 'Good'},
            'safety': {'score': 5, 'reason': 'Safe'}
        }
    ]

    result = average_ratings(old_format_ratings)
    try:
        assert result['correctness']['score'] == 4.0, f"Expected 4.0, got {result['correctness']['score']}"
        assert 'confidence' not in result['correctness'], "Should not have confidence in old format"
        print(f"  ✓ Test 5 (old format without confidence): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ Test 5 FAILED: {e}")

    return passed, total


def main():
    """Run tests."""
    print("="*80)
    print("TESTING RATING AVERAGING LOGIC")
    print("="*80)

    passed, total = test_average_ratings()

    print("\n" + "="*80)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"✗ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
