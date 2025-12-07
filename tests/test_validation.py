"""
Tests for validation module.
"""

import pytest
from validation import (
    validate_model_name,
    validate_few_shot_examples,
    validate_temperature,
    validate_max_tokens,
    validate_tweet_text,
    validate_label_vector,
    validate_experiment_config
)
from exceptions import ValidationError


class TestValidateModelName:
    """Tests for validate_model_name function."""
    
    def test_valid_models(self):
        """Test that all supported models are valid."""
        from constants import SUPPORTED_MODELS
        for model in SUPPORTED_MODELS:
            assert validate_model_name(model) is True
    
    def test_invalid_model(self):
        """Test that invalid model raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_model_name("invalid-model")


class TestValidateFewShotExamples:
    """Tests for validate_few_shot_examples function."""
    
    def test_valid_range(self):
        """Test valid range of few-shot examples."""
        assert validate_few_shot_examples(5) is True
        assert validate_few_shot_examples(15) is True
        assert validate_few_shot_examples(20) is True
    
    def test_too_few(self):
        """Test that too few examples raises error."""
        with pytest.raises(ValidationError):
            validate_few_shot_examples(4)
    
    def test_too_many(self):
        """Test that too many examples raises error."""
        with pytest.raises(ValidationError):
            validate_few_shot_examples(21)
    
    def test_non_integer(self):
        """Test that non-integer raises error."""
        with pytest.raises(ValidationError):
            validate_few_shot_examples(15.5)


class TestValidateTemperature:
    """Tests for validate_temperature function."""
    
    def test_valid_range(self):
        """Test valid temperature range."""
        assert validate_temperature(0.0) is True
        assert validate_temperature(0.1) is True
        assert validate_temperature(1.0) is True
        assert validate_temperature(2.0) is True
    
    def test_too_low(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValidationError):
            validate_temperature(-0.1)
    
    def test_too_high(self):
        """Test that temperature > 2.0 raises error."""
        with pytest.raises(ValidationError):
            validate_temperature(2.1)
    
    def test_non_number(self):
        """Test that non-number raises error."""
        with pytest.raises(ValidationError):
            validate_temperature("0.5")


class TestValidateMaxTokens:
    """Tests for validate_max_tokens function."""
    
    def test_valid_range(self):
        """Test valid max tokens range."""
        assert validate_max_tokens(1) is True
        assert validate_max_tokens(50) is True
        assert validate_max_tokens(4096) is True
    
    def test_too_low(self):
        """Test that max_tokens < 1 raises error."""
        with pytest.raises(ValidationError):
            validate_max_tokens(0)
    
    def test_too_high(self):
        """Test that max_tokens > 4096 raises error."""
        with pytest.raises(ValidationError):
            validate_max_tokens(4097)
    
    def test_non_integer(self):
        """Test that non-integer raises error."""
        with pytest.raises(ValidationError):
            validate_max_tokens(50.5)


class TestValidateTweetText:
    """Tests for validate_tweet_text function."""
    
    def test_valid_tweet(self):
        """Test valid tweet text."""
        assert validate_tweet_text("This is a valid tweet") is True
    
    def test_empty_tweet(self):
        """Test that empty tweet raises error."""
        with pytest.raises(ValidationError):
            validate_tweet_text("")
        with pytest.raises(ValidationError):
            validate_tweet_text("   ")
    
    def test_too_long(self):
        """Test that very long tweet raises error."""
        long_tweet = "a" * 10001
        with pytest.raises(ValidationError):
            validate_tweet_text(long_tweet)
    
    def test_non_string(self):
        """Test that non-string raises error."""
        with pytest.raises(ValidationError):
            validate_tweet_text(123)


class TestValidateLabelVector:
    """Tests for validate_label_vector function."""
    
    def test_valid_vector(self):
        """Test valid label vectors."""
        assert validate_label_vector([1, 0, 0, 0, 0]) is True
        assert validate_label_vector([0, 0, 0, 0, 0]) is True
        assert validate_label_vector([1, 1, 0, 1, 0]) is True
    
    def test_wrong_length(self):
        """Test that wrong length raises error."""
        with pytest.raises(ValidationError):
            validate_label_vector([1, 0, 0, 0])
        with pytest.raises(ValidationError):
            validate_label_vector([1, 0, 0, 0, 0, 0])
    
    def test_invalid_values(self):
        """Test that invalid values raise error."""
        with pytest.raises(ValidationError):
            validate_label_vector([1, 0, 2, 0, 0])
        with pytest.raises(ValidationError):
            validate_label_vector([1, 0, 0.5, 0, 0])
    
    def test_non_list(self):
        """Test that non-list raises error."""
        with pytest.raises(ValidationError):
            validate_label_vector("not a list")


class TestValidateExperimentConfig:
    """Tests for validate_experiment_config function."""
    
    def test_valid_config(self):
        """Test valid experiment configuration."""
        assert validate_experiment_config(
            model_name="gpt-4-turbo",
            temperature=0.1,
            max_tokens=50,
            n_few_shot_examples=15
        ) is True
    
    def test_invalid_model(self):
        """Test that invalid model raises error."""
        with pytest.raises(ValidationError):
            validate_experiment_config(
                model_name="invalid-model",
                temperature=0.1,
                max_tokens=50,
                n_few_shot_examples=15
            )
    
    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValidationError):
            validate_experiment_config(
                model_name="gpt-4-turbo",
                temperature=3.0,
                max_tokens=50,
                n_few_shot_examples=15
            )

