# {{title}} - Executive Summary

**Experiment:** {{experiment_name}}
**Date:** {{timestamp}}
**Author:** {{author}}

## Key Findings

### Best Performing Model
**{{best_model}}** achieved the highest overall performance with statistical significance.

- **Performance Improvement:** {{performance_improvement}}% over baseline
- **Statistical Confidence:** {{confidence_level}}
- **Effect Size:** {{effect_size}} ({{effect_size_interpretation}})

### Model Rankings

| Rank | Model | Performance Score | Confidence Interval | P-value |
|------|-------|------------------|-------------------|---------|
{% for model in model_rankings %}
| {{loop.index}} | {{model.name}} | {{model.score}} | [{{model.ci_lower}}, {{model.ci_upper}}] | {{model.p_value}} |
{% endfor %}

## Critical Insights

### Statistical Significance
{{statistical_summary}}

### Uncertainty Analysis
{{uncertainty_summary}}

### Physics Compliance
{{physics_summary}}

## Recommendations

### Immediate Actions
{% for recommendation in immediate_actions %}
- {{recommendation}}
{% endfor %}

### Strategic Decisions
{% for recommendation in strategic_decisions %}
- {{recommendation}}
{% endfor %}

### Risk Assessment
{% for risk in identified_risks %}
- **{{risk.category}}:** {{risk.description}}
  - **Impact:** {{risk.impact}}
  - **Mitigation:** {{risk.mitigation}}
{% endfor %}

## Business Impact

### Performance Gains
- **Accuracy Improvement:** {{accuracy_improvement}}%
- **Computational Efficiency:** {{efficiency_gain}}%
- **Resource Optimization:** {{resource_savings}}%

### Cost-Benefit Analysis
- **Implementation Cost:** {{implementation_cost}}
- **Expected ROI:** {{expected_roi}}
- **Break-even Timeline:** {{breakeven_timeline}}

## Next Steps

1. **{{next_step_1}}**
2. **{{next_step_2}}**
3. **{{next_step_3}}**

---

*This executive summary provides high-level insights from comprehensive statistical analysis. For detailed technical information, refer to the full technical report.*

**Confidence Level:** {{overall_confidence}}
**Analysis Framework:** Generative Operator Analysis Framework v{{framework_version}}