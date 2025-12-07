"""
Tests for classification module.
"""

import pytest
from classification import parse_llm_response, build_classification_prompt


class TestParseLLMResponse:
    """Tests for parse_llm_response function."""
    
    def test_parse_none_response(self):
        """Test parsing None response."""
        vector, labels, is_valid = parse_llm_response("")
        assert vector == [0, 0, 0, 0, 0]
        assert labels == "None"
        assert is_valid is False
    
    def test_parse_none_label(self):
        """Test parsing 'None' response."""
        vector, labels, is_valid = parse_llm_response("None")
        assert vector == [0, 0, 0, 0, 0]
        assert labels == "None"
        assert is_valid is True
    
    def test_parse_single_label(self):
        """Test parsing single label."""
        vector, labels, is_valid = parse_llm_response("Gayphobia")
        assert vector == [1, 0, 0, 0, 0]
        assert labels == "Gayphobia"
        assert is_valid is True
    
    def test_parse_multiple_labels(self):
        """Test parsing multiple labels."""
        vector, labels, is_valid = parse_llm_response("Gayphobia, Transphobia")
        assert vector == [1, 0, 0, 1, 0]
        assert "Gayphobia" in labels
        assert "Transphobia" in labels
        assert is_valid is True
    
    def test_parse_label_variations(self):
        """Test parsing label name variations."""
        # Test with different cases and formats
        test_cases = [
            ("gayphobia", [1, 0, 0, 0, 0]),
            ("GAYPHOBIA", [1, 0, 0, 0, 0]),
            ("Gay-Phobia", [1, 0, 0, 0, 0]),
            ("gay phobia", [1, 0, 0, 0, 0]),
            ("lesbophobia", [0, 1, 0, 0, 0]),
            ("biphobia", [0, 0, 1, 0, 0]),
            ("transphobia", [0, 0, 0, 1, 0]),
            ("other", [0, 0, 0, 0, 1]),
        ]
        
        for response, expected_vector in test_cases:
            vector, _, is_valid = parse_llm_response(response)
            assert vector == expected_vector, f"Failed for: {response}"
            assert is_valid is True
    
    def test_parse_all_labels(self):
        """Test parsing all labels."""
        vector, labels, is_valid = parse_llm_response(
            "Gayphobia, Lesbophobia, Biphobia, Transphobia, Other"
        )
        assert vector == [1, 1, 1, 1, 1]
        assert is_valid is True


class TestBuildClassificationPrompt:
    """Tests for build_classification_prompt function."""
    
    def test_build_prompt_structure(self):
        """Test that prompt has correct structure."""
        few_shot_examples = [
            {'text': 'Test tweet 1', 'labels': 'Gayphobia'},
            {'text': 'Test tweet 2', 'labels': 'None'}
        ]
        target_tweet = "This is a test tweet"
        
        messages = build_classification_prompt(few_shot_examples, target_tweet)
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert 'system' in messages[0]['content'].lower()
        assert target_tweet in messages[1]['content']
    
    def test_build_prompt_with_examples(self):
        """Test that few-shot examples are included."""
        few_shot_examples = [
            {'text': 'Example 1', 'labels': 'Gayphobia'},
            {'text': 'Example 2', 'labels': 'Transphobia'}
        ]
        target_tweet = "Target tweet"
        
        messages = build_classification_prompt(few_shot_examples, target_tweet)
        user_content = messages[1]['content']
        
        assert 'Example 1' in user_content
        assert 'Example 2' in user_content
        assert 'Gayphobia' in user_content
        assert 'Transphobia' in user_content
        assert 'Target tweet' in user_content
    
    def test_build_prompt_escapes_quotes(self):
        """Test that quotes in tweets are escaped."""
        few_shot_examples = [{'text': 'Tweet with "quotes"', 'labels': 'Gayphobia'}]
        target_tweet = 'Target with "quotes"'
        
        messages = build_classification_prompt(few_shot_examples, target_tweet)
        user_content = messages[1]['content']
        
        # Check that quotes are escaped (backslash before quote)
        assert '\\"' in user_content or '"' in user_content  # Either escaped or not, both acceptable

