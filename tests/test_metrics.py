"""Tests for fairness and bias metrics."""

import pytest
import numpy as np
from fingerprint_squared.metrics.fairness import FairnessMetrics
from fingerprint_squared.metrics.bias_scores import BiasScorer
from fingerprint_squared.metrics.intersectional import IntersectionalAnalyzer


class TestFairnessMetrics:
    """Tests for FairnessMetrics class."""

    def setup_method(self):
        self.fm = FairnessMetrics(epsilon=0.1)

    def test_demographic_parity_equal(self):
        """Test demographic parity with equal rates."""
        predictions = {
            "group_a": [1, 1, 0, 0, 1],
            "group_b": [1, 1, 0, 0, 1],
        }
        result = self.fm.demographic_parity(predictions)
        assert result.value == 0.0
        assert result.is_fair == True

    def test_demographic_parity_unequal(self):
        """Test demographic parity with unequal rates."""
        predictions = {
            "group_a": [1, 1, 1, 1, 1],
            "group_b": [0, 0, 0, 0, 0],
        }
        result = self.fm.demographic_parity(predictions)
        assert result.value == 1.0
        assert result.is_fair == False

    def test_equalized_odds(self):
        """Test equalized odds computation."""
        predictions = {
            "group_a": [1, 1, 0, 0],
            "group_b": [1, 0, 1, 0],
        }
        labels = {
            "group_a": [1, 0, 0, 1],
            "group_b": [1, 1, 0, 0],
        }
        result = self.fm.equalized_odds(predictions, labels)
        assert "tpr_gap" in result.details
        assert "fpr_gap" in result.details

    def test_equal_opportunity(self):
        """Test equal opportunity computation."""
        predictions = {
            "group_a": [1, 1, 0, 0],
            "group_b": [0, 1, 0, 1],
        }
        labels = {
            "group_a": [1, 1, 0, 0],
            "group_b": [1, 1, 0, 0],
        }
        result = self.fm.equal_opportunity(predictions, labels)
        assert result.metric_name == "equal_opportunity"

    def test_compute_all(self):
        """Test computing all metrics."""
        predictions = {
            "group_a": [1, 1, 0, 0],
            "group_b": [0, 1, 0, 1],
        }
        labels = {
            "group_a": [1, 1, 0, 0],
            "group_b": [1, 1, 0, 0],
        }
        results = self.fm.compute_all(predictions, labels)
        assert "demographic_parity" in results
        assert "equalized_odds" in results
        assert "equal_opportunity" in results


class TestBiasScorer:
    """Tests for BiasScorer class."""

    def setup_method(self):
        self.scorer = BiasScorer()

    def test_compute_bias_score_clean(self):
        """Test bias score with clean text."""
        texts = [
            "The doctor examined the patient.",
            "The engineer completed the project.",
        ]
        result = self.scorer.compute_bias_score(texts)
        assert result.overall_score >= 0
        assert result.overall_score <= 1

    def test_compute_bias_score_biased(self):
        """Test bias score with biased text."""
        texts = [
            "The female doctor was surprisingly competent.",
            "She was emotional during the meeting.",
        ]
        result = self.scorer.compute_bias_score(texts)
        assert len(result.detections) > 0

    def test_stereotype_association(self):
        """Test stereotype association detection."""
        outputs_by_group = {
            "female": ["She was nurturing and emotional."],
            "male": ["He was strong and logical."],
        }
        scores = self.scorer.compute_stereotype_association_score(outputs_by_group)
        assert "female" in scores
        assert "male" in scores


class TestIntersectionalAnalyzer:
    """Tests for IntersectionalAnalyzer class."""

    def setup_method(self):
        self.analyzer = IntersectionalAnalyzer(
            protected_attributes=["gender", "race"],
            min_group_size=3,
        )

    def test_analyze_basic(self):
        """Test basic intersectional analysis."""
        data = []
        for gender in ["male", "female"]:
            for race in ["white", "black"]:
                for _ in range(5):
                    score = 0.8 if gender == "male" and race == "white" else 0.6
                    data.append({"gender": gender, "race": race, "score": score})

        result = self.analyzer.analyze(data, metric_key="score")
        assert result.intersectional_gap > 0
        assert result.worst_performing_group is not None
        assert result.best_performing_group is not None

    def test_disparity_matrix(self):
        """Test disparity matrix generation."""
        data = [
            {"gender": "male", "race": "white", "score": 0.8},
            {"gender": "male", "race": "black", "score": 0.7},
            {"gender": "female", "race": "white", "score": 0.75},
            {"gender": "female", "race": "black", "score": 0.65},
        ]
        matrix = self.analyzer.generate_disparity_matrix(
            data, "gender", "race", "score"
        )
        assert "matrix" in matrix
        assert "disparity_range" in matrix


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
