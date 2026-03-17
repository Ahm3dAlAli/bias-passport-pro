"""
Report generation for Fingerprint² evaluation results.

Generates comprehensive HTML and markdown reports with
visualizations, metrics, and recommendations.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fingerprint_squared.core.fingerprint import ModelFingerprint


class ReportGenerator:
    """
    Generate comprehensive evaluation reports.

    Creates HTML and markdown reports with visualizations,
    detailed metrics, and actionable recommendations.

    Example:
        >>> generator = ReportGenerator()
        >>> generator.generate_html_report(result, output_dir="./reports")
    """

    def __init__(
        self,
        include_visualizations: bool = True,
        include_raw_data: bool = False,
    ):
        self.include_visualizations = include_visualizations
        self.include_raw_data = include_raw_data

    def generate_html_report(
        self,
        result: Any,  # EvaluationResult
        fingerprint: Optional[ModelFingerprint] = None,
        output_dir: Path = Path("./reports"),
    ) -> Path:
        """
        Generate a comprehensive HTML report.

        Args:
            result: EvaluationResult instance
            fingerprint: Optional ModelFingerprint
            output_dir: Output directory

        Returns:
            Path to generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"{result.model_name}_{result.timestamp}_report.html"

        html_content = self._build_html_report(result, fingerprint)

        with open(report_path, "w") as f:
            f.write(html_content)

        # Save JSON data
        json_path = output_dir / f"{result.model_name}_{result.timestamp}_data.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return report_path

    def _build_html_report(
        self,
        result: Any,
        fingerprint: Optional[ModelFingerprint],
    ) -> str:
        """Build the HTML report content."""
        bias_level_color = {
            "low": "#28a745",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545",
        }

        fairness_level_color = {
            "high": "#28a745",
            "medium": "#ffc107",
            "low": "#fd7e14",
            "poor": "#dc3545",
        }

        bias_color = bias_level_color.get(
            fingerprint.bias_level if fingerprint else "medium", "#6c757d"
        )
        fairness_color = fairness_level_color.get(
            fingerprint.fairness_level if fingerprint else "medium", "#6c757d"
        )

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fingerprint² Report: {result.model_name}</title>
    <style>
        :root {{
            --primary-color: #4a90d9;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --background-color: #f8f9fa;
            --card-background: #ffffff;
            --text-color: #333333;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #4a90d9, #357abd);
            color: white;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .card {{
            background: var(--card-background);
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 25px;
            margin-bottom: 25px;
        }}

        .card h2 {{
            color: var(--primary-color);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .metric-card {{
            background: var(--background-color);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .metric-label {{
            color: var(--secondary-color);
            font-size: 0.9em;
        }}

        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            color: white;
        }}

        .progress-bar {{
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}

        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}

        th {{
            background: var(--background-color);
            font-weight: 600;
        }}

        .risk-list {{
            list-style: none;
        }}

        .risk-list li {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            display: flex;
            align-items: center;
        }}

        .risk-list li.high {{
            background: #fff3cd;
            border-left: 4px solid var(--warning-color);
        }}

        .risk-list li.critical {{
            background: #f8d7da;
            border-left: 4px solid var(--danger-color);
        }}

        .strength-list li {{
            background: #d4edda;
            border-left: 4px solid var(--success-color);
        }}

        .recommendations {{
            background: #e7f3ff;
            border-left: 4px solid var(--primary-color);
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}

        footer {{
            text-align: center;
            padding: 30px;
            color: var(--secondary-color);
            font-size: 0.9em;
        }}

        .fingerprint-hash {{
            font-family: monospace;
            background: #e9ecef;
            padding: 5px 10px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Fingerprint² Evaluation Report</h1>
        <p class="subtitle">{result.model_name} ({result.model_provider})</p>
        <p style="margin-top: 10px;">Generated: {result.timestamp}</p>
    </header>

    <div class="container">
        <!-- Executive Summary -->
        <div class="card">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" style="color: {bias_color};">
                        {result.overall_bias_score:.2f}
                    </div>
                    <div class="metric-label">Overall Bias Score</div>
                    <span class="badge" style="background: {bias_color};">
                        {fingerprint.bias_level.upper() if fingerprint else 'N/A'}
                    </span>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: {fairness_color};">
                        {result.overall_fairness_score:.2f}
                    </div>
                    <div class="metric-label">Overall Fairness Score</div>
                    <span class="badge" style="background: {fairness_color};">
                        {fingerprint.fairness_level.upper() if fingerprint else 'N/A'}
                    </span>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.total_probes}</div>
                    <div class="metric-label">Total Probes Run</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.total_responses}</div>
                    <div class="metric-label">Valid Responses</div>
                </div>
            </div>
            {f'<p><strong>Fingerprint Hash:</strong> <span class="fingerprint-hash">{fingerprint.fingerprint_hash}</span></p>' if fingerprint else ''}
        </div>

        <!-- Bias Scores by Dimension -->
        <div class="card">
            <h2>📈 Bias Scores by Dimension</h2>
            {self._generate_dimension_table(result.bias_scores)}
        </div>

        <!-- Fairness Metrics -->
        <div class="card">
            <h2>⚖️ Fairness Metrics</h2>
            {self._generate_fairness_table(result.fairness_results)}
        </div>

        <!-- Risk Areas and Strengths -->
        <div class="card">
            <h2>⚠️ Risk Assessment</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h3 style="color: var(--danger-color);">Risk Areas</h3>
                    <ul class="risk-list">
                        {self._generate_risk_list(fingerprint.risk_areas if fingerprint else [])}
                    </ul>
                </div>
                <div>
                    <h3 style="color: var(--success-color);">Strengths</h3>
                    <ul class="risk-list strength-list">
                        {self._generate_strength_list(fingerprint.strengths if fingerprint else [])}
                    </ul>
                </div>
            </div>
        </div>

        <!-- Recommendations -->
        <div class="card">
            <h2>💡 Recommendations</h2>
            {self._generate_recommendations(result, fingerprint)}
        </div>

        <!-- Detailed Results -->
        <div class="card">
            <h2>📋 Detailed Results</h2>
            <h3>Stereotype Analysis</h3>
            <p>Total stereotype probes: {len(result.stereotype_results)}</p>
            <p>Stereotypical responses: {sum(1 for r in result.stereotype_results if r.is_stereotypical)}</p>

            <h3 style="margin-top: 20px;">Counterfactual Analysis</h3>
            <p>Total counterfactual pairs: {len(result.counterfactual_results)}</p>
            <p>Fair responses: {sum(1 for r in result.counterfactual_results if r.is_fair)}</p>
        </div>
    </div>

    <footer>
        <p>Generated by Fingerprint² - Ethical AI Assessment Framework</p>
        <p>Version 0.1.0 | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </footer>
</body>
</html>
"""
        return html

    def _generate_dimension_table(self, bias_scores: Dict[str, Any]) -> str:
        """Generate HTML table for dimension scores."""
        rows = []
        for dim, score in bias_scores.items():
            if hasattr(score, 'overall_score'):
                value = score.overall_score
            else:
                value = score

            color = "#28a745" if value < 0.3 else "#ffc107" if value < 0.6 else "#dc3545"
            rows.append(f"""
                <tr>
                    <td>{dim.replace('_', ' ').title()}</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {value * 100}%; background: {color};"></div>
                        </div>
                    </td>
                    <td style="color: {color}; font-weight: bold;">{value:.3f}</td>
                </tr>
            """)

        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Dimension</th>
                        <th>Score</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """

    def _generate_fairness_table(self, fairness_results: Dict[str, Any]) -> str:
        """Generate HTML table for fairness metrics."""
        rows = []
        for metric, result in fairness_results.items():
            value = result.value
            is_fair = result.is_fair
            color = "#28a745" if is_fair else "#dc3545"
            status = "✓ Fair" if is_fair else "✗ Unfair"

            rows.append(f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{value:.3f}</td>
                    <td>{result.threshold:.3f}</td>
                    <td style="color: {color};">{status}</td>
                </tr>
            """)

        if not rows:
            return "<p>No fairness metrics available.</p>"

        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Threshold</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """

    def _generate_risk_list(self, risks: List[str]) -> str:
        """Generate HTML list for risk areas."""
        if not risks:
            return "<li>No significant risk areas identified.</li>"

        items = []
        for risk in risks:
            items.append(f'<li class="high">⚠️ {risk.replace("_", " ").title()}</li>')
        return ''.join(items)

    def _generate_strength_list(self, strengths: List[str]) -> str:
        """Generate HTML list for strengths."""
        if not strengths:
            return "<li>No particular strengths identified.</li>"

        items = []
        for strength in strengths:
            items.append(f'<li>✓ {strength.replace("_", " ").title()}</li>')
        return ''.join(items)

    def _generate_recommendations(
        self,
        result: Any,
        fingerprint: Optional[ModelFingerprint],
    ) -> str:
        """Generate recommendations based on results."""
        recommendations = []

        # Based on bias level
        if fingerprint:
            if fingerprint.bias_level in ["high", "critical"]:
                recommendations.append(
                    "Consider implementing bias mitigation techniques such as "
                    "debiasing during fine-tuning or output filtering."
                )

            if fingerprint.fairness_level in ["low", "poor"]:
                recommendations.append(
                    "Review training data for demographic imbalances and "
                    "consider balanced sampling or data augmentation."
                )

            # Based on specific risk areas
            for risk in fingerprint.risk_areas[:3]:
                if "gender" in risk.lower():
                    recommendations.append(
                        "Address gender bias by reviewing occupation and trait associations."
                    )
                elif "racial" in risk.lower() or "race" in risk.lower():
                    recommendations.append(
                        "Implement additional safeguards for racial sensitivity in outputs."
                    )
                elif "stereotype" in risk.lower():
                    recommendations.append(
                        "Add stereotype detection and mitigation to the output pipeline."
                    )

        # Based on counterfactual results
        if result.counterfactual_results:
            unfair_ratio = sum(1 for r in result.counterfactual_results if not r.is_fair) / len(result.counterfactual_results)
            if unfair_ratio > 0.3:
                recommendations.append(
                    "High counterfactual sensitivity detected. Consider counterfactual "
                    "data augmentation during training."
                )

        if not recommendations:
            recommendations.append(
                "Continue monitoring bias metrics as the model is updated."
            )

        html = ""
        for rec in recommendations:
            html += f'<div class="recommendations">💡 {rec}</div>'

        return html

    def generate_markdown_report(
        self,
        result: Any,
        fingerprint: Optional[ModelFingerprint] = None,
        output_dir: Path = Path("./reports"),
    ) -> Path:
        """Generate a markdown report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"{result.model_name}_{result.timestamp}_report.md"

        md_content = self._build_markdown_report(result, fingerprint)

        with open(report_path, "w") as f:
            f.write(md_content)

        return report_path

    def _build_markdown_report(
        self,
        result: Any,
        fingerprint: Optional[ModelFingerprint],
    ) -> str:
        """Build markdown report content."""
        md = f"""# Fingerprint² Evaluation Report

## Model: {result.model_name}
**Provider:** {result.model_provider}
**Timestamp:** {result.timestamp}

---

## Executive Summary

| Metric | Value | Level |
|--------|-------|-------|
| Overall Bias Score | {result.overall_bias_score:.3f} | {fingerprint.bias_level if fingerprint else 'N/A'} |
| Overall Fairness Score | {result.overall_fairness_score:.3f} | {fingerprint.fairness_level if fingerprint else 'N/A'} |
| Total Probes | {result.total_probes} | - |
| Valid Responses | {result.total_responses} | - |

{f'**Fingerprint Hash:** `{fingerprint.fingerprint_hash}`' if fingerprint else ''}

---

## Bias Scores by Dimension

| Dimension | Score |
|-----------|-------|
"""
        for dim, score in result.bias_scores.items():
            value = score.overall_score if hasattr(score, 'overall_score') else score
            md += f"| {dim.replace('_', ' ').title()} | {value:.3f} |\n"

        md += """
---

## Fairness Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
"""
        for metric, fr in result.fairness_results.items():
            status = "✓ Fair" if fr.is_fair else "✗ Unfair"
            md += f"| {metric.replace('_', ' ').title()} | {fr.value:.3f} | {fr.threshold:.3f} | {status} |\n"

        if fingerprint:
            md += f"""
---

## Risk Assessment

### Risk Areas
"""
            for risk in fingerprint.risk_areas:
                md += f"- ⚠️ {risk.replace('_', ' ').title()}\n"

            md += """
### Strengths
"""
            for strength in fingerprint.strengths:
                md += f"- ✓ {strength.replace('_', ' ').title()}\n"

        md += f"""
---

## Detailed Results

- **Stereotype Probes:** {len(result.stereotype_results)}
- **Stereotypical Responses:** {sum(1 for r in result.stereotype_results if r.is_stereotypical)}
- **Counterfactual Pairs:** {len(result.counterfactual_results)}
- **Fair Responses:** {sum(1 for r in result.counterfactual_results if r.is_fair)}

---

*Generated by Fingerprint² - Ethical AI Assessment Framework*
*Version 0.1.0 | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        return md
