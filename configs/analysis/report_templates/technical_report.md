# {{title}} - Technical Report

**Experiment:** {{experiment_name}}
**Date:** {{timestamp}}
**Author:** {{author}}
**Framework Version:** {{framework_version}}

## Executive Summary

{{executive_summary}}

## Methodology

### Experimental Setup

**Dataset:** {{dataset_name}}
**Sample Size:** {{sample_size}}
**Evaluation Metrics:** {{evaluation_metrics}}
**Statistical Tests:** {{statistical_tests}}

### Models Evaluated

{% for model in models_evaluated %}
- **{{model.name}}**
  - Architecture: {{model.architecture}}
  - Parameters: {{model.parameters}}
  - Training Data: {{model.training_data}}
  - Inference Time: {{model.inference_time}}ms
{% endfor %}

### Analysis Framework

- **Statistical Significance Level:** α = {{alpha}}
- **Bootstrap Samples:** {{n_bootstrap}}
- **Monte Carlo Samples:** {{n_monte_carlo}}
- **Multiple Comparison Correction:** {{correction_method}}
- **Random Seed:** {{random_state}}

## Results

### Performance Metrics

#### Overall Performance
| Model | MSE | MAE | R² Score | Max Error | Inference Time (ms) |
|-------|-----|-----|----------|-----------|-------------------|
{% for model in performance_results %}
| {{model.name}} | {{model.mse}} | {{model.mae}} | {{model.r2}} | {{model.max_error}} | {{model.inference_time}} |
{% endfor %}

#### Physics-Aware Metrics
{% if physics_metrics %}
| Model | Divergence Error | Mass Conservation | Energy Spectrum |
|-------|-----------------|------------------|-----------------|
{% for model in physics_metrics %}
| {{model.name}} | {{model.divergence_error}} | {{model.mass_conservation}} | {{model.energy_spectrum}} |
{% endfor %}
{% endif %}

### Statistical Analysis

#### Pairwise Comparisons
{% for comparison in pairwise_comparisons %}
**{{comparison.model1}} vs {{comparison.model2}}**
- Test Statistic: {{comparison.test_statistic}}
- P-value: {{comparison.p_value}}
- Effect Size (Cohen's d): {{comparison.effect_size}}
- 95% CI: [{{comparison.ci_lower}}, {{comparison.ci_upper}}]
- Interpretation: {{comparison.interpretation}}

{% endfor %}

#### Multiple Comparison Results
After {{correction_method}} correction:

| Comparison | Original P-value | Corrected P-value | Significant |
|------------|------------------|-------------------|-------------|
{% for comparison in corrected_comparisons %}
| {{comparison.comparison}} | {{comparison.original_p}} | {{comparison.corrected_p}} | {{comparison.significant}} |
{% endfor %}

### Uncertainty Quantification

#### Epistemic Uncertainty
{% for model in uncertainty_results %}
**{{model.name}}**
- Mean Uncertainty: {{model.mean_uncertainty}}
- 95th Percentile: {{model.uncertainty_95th}}
- Calibration Score: {{model.calibration_score}}
- Coverage Probability: {{model.coverage_probability}}

{% endfor %}

#### Reliability Analysis
{% if calibration_results %}
- **Overall Calibration Error:** {{overall_calibration_error}}
- **Expected Calibration Error (ECE):** {{ece}}
- **Maximum Calibration Error (MCE):** {{mce}}
- **Brier Score:** {{brier_score}}
{% endif %}

### Model Rankings

Based on comprehensive analysis incorporating performance, statistical significance, and uncertainty:

| Rank | Model | Composite Score | Performance | Uncertainty | Physics Compliance |
|------|-------|----------------|-------------|-------------|-------------------|
{% for ranking in model_rankings %}
| {{ranking.rank}} | {{ranking.model}} | {{ranking.composite_score}} | {{ranking.performance_score}} | {{ranking.uncertainty_score}} | {{ranking.physics_score}} |
{% endfor %}

### Rollout Stability Analysis
{% if rollout_analysis %}
| Model | Stable Steps | Divergence Time | Error Growth Rate |
|-------|--------------|-----------------|-------------------|
{% for model in rollout_results %}
| {{model.name}} | {{model.stable_steps}} | {{model.divergence_time}} | {{model.error_growth_rate}} |
{% endfor %}
{% endif %}

## Detailed Analysis

### Best Model: {{best_model}}

#### Performance Characteristics
- **Superior Metrics:** {{best_model_superior_metrics}}
- **Statistical Significance:** {{best_model_significance}}
- **Effect Size:** {{best_model_effect_size}} ({{effect_size_interpretation}})
- **Confidence Interval:** [{{best_model_ci_lower}}, {{best_model_ci_upper}}]

#### Uncertainty Profile
- **Epistemic Uncertainty:** {{best_model_epistemic}}
- **Aleatoric Uncertainty:** {{best_model_aleatoric}}
- **Total Uncertainty:** {{best_model_total}}
- **Calibration Quality:** {{best_model_calibration}}

#### Physics Compliance
{% for physics_check in best_model_physics %}
- **{{physics_check.name}}:** {{physics_check.status}} ({{physics_check.score}})
{% endfor %}

### Comparative Analysis

#### Key Differentiators
{% for differentiator in key_differentiators %}
- **{{differentiator.category}}:** {{differentiator.description}}
  - Best: {{differentiator.best_model}} ({{differentiator.best_score}})
  - Worst: {{differentiator.worst_model}} ({{differentiator.worst_score}})
{% endfor %}

#### Trade-off Analysis
{% for tradeoff in tradeoffs %}
- **{{tradeoff.metric1}} vs {{tradeoff.metric2}}**
  - Correlation: {{tradeoff.correlation}}
  - Interpretation: {{tradeoff.interpretation}}
{% endfor %}

## Limitations and Assumptions

### Experimental Limitations
{% for limitation in experimental_limitations %}
- {{limitation}}
{% endfor %}

### Statistical Assumptions
{% for assumption in statistical_assumptions %}
- {{assumption}}
{% endfor %}

### Model Limitations
{% for limitation in model_limitations %}
- {{limitation}}
{% endfor %}

## Recommendations

### Model Selection
**Primary Recommendation:** {{primary_recommendation}}

**Rationale:**
{% for rationale in primary_rationale %}
- {{rationale}}
{% endfor %}

### Alternative Scenarios
{% for scenario in alternative_scenarios %}
**{{scenario.condition}}:** {{scenario.recommendation}}
- Rationale: {{scenario.rationale}}
{% endfor %}

### Implementation Considerations
{% for consideration in implementation_considerations %}
- **{{consideration.category}}:** {{consideration.description}}
  - Priority: {{consideration.priority}}
  - Timeline: {{consideration.timeline}}
{% endfor %}

### Future Work
{% for future_work in future_recommendations %}
- {{future_work}}
{% endfor %}

## Validation and Reproducibility

### Reproducibility Information
- **Random Seed:** {{random_state}}
- **Framework Version:** {{framework_version}}
- **Dependencies:** {{dependencies_version}}
- **Hardware:** {{hardware_info}}
- **Execution Time:** {{total_execution_time}}

### Cross-Validation Results
{% if cross_validation %}
| Model | CV Mean | CV Std | 95% CI |
|-------|---------|--------|--------|
{% for cv_result in cross_validation %}
| {{cv_result.model}} | {{cv_result.mean}} | {{cv_result.std}} | [{{cv_result.ci_lower}}, {{cv_result.ci_upper}}] |
{% endfor %}
{% endif %}

### Sensitivity Analysis
{% if sensitivity_analysis %}
{% for sensitivity in sensitivity_analysis %}
**{{sensitivity.parameter}}**
- Baseline: {{sensitivity.baseline}}
- Range: [{{sensitivity.min}}, {{sensitivity.max}}]
- Impact on Results: {{sensitivity.impact}}
{% endfor %}
{% endif %}

## Appendices

### Appendix A: Detailed Statistical Results
{{detailed_statistical_results}}

### Appendix B: Visualization Gallery
{% if visualization_paths %}
{% for viz in visualization_paths %}
- {{viz.description}}: `{{viz.path}}`
{% endfor %}
{% endif %}

### Appendix C: Configuration
```yaml
{{analysis_configuration}}
```

### Appendix D: Raw Data Summary
- **Total Samples:** {{total_samples}}
- **Features:** {{n_features}}
- **Missing Values:** {{missing_values}}
- **Outliers Detected:** {{outliers_detected}}
- **Data Quality Score:** {{data_quality_score}}

---

**Report Generated:** {{report_generation_timestamp}}
**Analysis Framework:** [Generative Operator Analysis Framework](https://github.com/your-org/generatively-stabilised-nos)
**Contact:** {{contact_email}}