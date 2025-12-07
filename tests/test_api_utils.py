"""
Tests for api_utils module.
"""

import pytest
from api_utils import get_model_pricing, estimate_api_cost


class TestGetModelPricing:
    """Tests for get_model_pricing function."""
    
    def test_gpt4_turbo_pricing(self):
        """Test GPT-4 Turbo pricing."""
        pricing = get_model_pricing("gpt-4-turbo")
        assert pricing['input'] == 10.00
        assert pricing['output'] == 30.00
    
    def test_gpt4_pricing(self):
        """Test GPT-4 pricing."""
        pricing = get_model_pricing("gpt-4")
        assert pricing['input'] == 30.00
        assert pricing['output'] == 60.00
    
    def test_gpt35_turbo_pricing(self):
        """Test GPT-3.5 Turbo pricing."""
        pricing = get_model_pricing("gpt-3.5-turbo")
        assert pricing['input'] == 0.50
        assert pricing['output'] == 1.50
    
    def test_gpt41_nano_pricing(self):
        """Test GPT-4.1 Nano pricing."""
        pricing = get_model_pricing("gpt-4.1-nano")
        assert pricing['input'] == 0.15
        assert pricing['output'] == 0.60
    
    def test_case_insensitive(self):
        """Test that model name matching is case-insensitive."""
        pricing1 = get_model_pricing("GPT-4-TURBO")
        pricing2 = get_model_pricing("gpt-4-turbo")
        assert pricing1 == pricing2
    
    def test_unknown_model_defaults(self):
        """Test that unknown models default to gpt-4-turbo pricing."""
        pricing = get_model_pricing("unknown-model")
        assert pricing['input'] == 10.00
        assert pricing['output'] == 30.00


class TestEstimateAPICost:
    """Tests for estimate_api_cost function."""
    
    def test_cost_calculation(self):
        """Test basic cost calculation."""
        cost = estimate_api_cost(
            model_name="gpt-4-turbo",
            num_test_tweets=100,
            num_ablations=1,
            few_shot_examples=15
        )
        
        assert 'total_cost' in cost
        assert 'input_cost' in cost
        assert 'output_cost' in cost
        assert cost['total_cost'] > 0
        assert cost['total_cost'] == cost['input_cost'] + cost['output_cost']
    
    def test_cost_increases_with_tweets(self):
        """Test that cost increases with number of tweets."""
        cost1 = estimate_api_cost(
            model_name="gpt-4-turbo",
            num_test_tweets=50,
            num_ablations=1,
            few_shot_examples=15
        )
        cost2 = estimate_api_cost(
            model_name="gpt-4-turbo",
            num_test_tweets=100,
            num_ablations=1,
            few_shot_examples=15
        )
        
        assert cost2['total_cost'] > cost1['total_cost']
    
    def test_cost_increases_with_examples(self):
        """Test that cost increases with few-shot examples."""
        cost1 = estimate_api_cost(
            model_name="gpt-4-turbo",
            num_test_tweets=100,
            num_ablations=1,
            few_shot_examples=5
        )
        cost2 = estimate_api_cost(
            model_name="gpt-4-turbo",
            num_test_tweets=100,
            num_ablations=1,
            few_shot_examples=15
        )
        
        assert cost2['total_cost'] > cost1['total_cost']
    
    def test_cost_structure(self):
        """Test that cost estimate has all required keys."""
        cost = estimate_api_cost(
            model_name="gpt-4-turbo",
            num_test_tweets=100,
            num_ablations=1,
            few_shot_examples=15
        )
        
        required_keys = [
            'total_cost',
            'input_cost',
            'output_cost',
            'cost_per_ablation',
            'total_input_tokens',
            'total_output_tokens',
            'tokens_per_request_input',
            'tokens_per_request_output'
        ]
        
        for key in required_keys:
            assert key in cost

