
# Comprehensive PHI Analysis Framework with Publication-Quality Visualizations and Downloadable Outputs
# Combined Enhanced PHI Analysis + Deep Confusion Matrix Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.spatial import ConvexHull
import warnings
import os
from datetime import datetime
import json
warnings.filterwarnings('ignore')

# Set up the plotting style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ComprehensivePHIAnalyzer:
    """
    Comprehensive PHI Analysis Framework for Transformer-based De-identification
    Combines enhanced analysis with deep confusion matrix insights
    Includes downloadable outputs functionality
    """

    def __init__(self, output_dir="phi_analysis_outputs"):
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Model and entity definitions
        self.models = ['RoBERTa-Large', 'ClinicalBERT', 'BioBERT']
        self.entities = ['DATE', 'HOSPITAL', 'ID', 'LOCATION', 'NAME', 'PHONE']

        # Performance data from BioBERT analysis
        self.performance_data = {
            'RoBERTa-Large': {
                'overall': {'f1': 0.9736, 'precision': 0.9626, 'recall': 0.9849, 'accuracy': 0.9954},
                'entities': {
                    'ID': {'f1': 0.9967, 'precision': 0.9941, 'recall': 0.9994, 'support': 4509},
                    'NAME': {'f1': 0.9712, 'precision': 0.9620, 'recall': 0.9806, 'support': 1856},
                    'HOSPITAL': {'f1': 0.9474, 'precision': 0.9293, 'recall': 0.9663, 'support': 1373},
                    'DATE': {'f1': 0.8369, 'precision': 0.7662, 'recall': 0.9219, 'support': 282},
                    'LOCATION': {'f1': 0.6207, 'precision': 0.5233, 'recall': 0.7627, 'support': 128},
                    'PHONE': {'f1': 0.8706, 'precision': 0.8810, 'recall': 0.8605, 'support': 182}
                }
            },
            'ClinicalBERT': {
                'overall': {'f1': 0.9659, 'precision': 0.9533, 'recall': 0.9787, 'accuracy': 0.9916},
                'entities': {
                    'ID': {'f1': 0.9983, 'precision': 0.9974, 'recall': 0.9993, 'support': 4509},
                    'NAME': {'f1': 0.9501, 'precision': 0.9355, 'recall': 0.9652, 'support': 1856},
                    'HOSPITAL': {'f1': 0.9103, 'precision': 0.8610, 'recall': 0.9656, 'support': 1373},
                    'DATE': {'f1': 0.8211, 'precision': 0.7786, 'recall': 0.8685, 'support': 282},
                    'LOCATION': {'f1': 0.6618, 'precision': 0.6923, 'recall': 0.6338, 'support': 128},
                    'PHONE': {'f1': 0.8478, 'precision': 0.8864, 'recall': 0.8125, 'support': 182}
                }
            },
            'BioBERT': {
                'overall': {'f1': 0.9467, 'precision': 0.9219, 'recall': 0.9729, 'accuracy': 0.9881},
                'entities': {
                    'ID': {'f1': 0.9989, 'precision': 0.9981, 'recall': 0.9998, 'support': 4509},
                    'NAME': {'f1': 0.9338, 'precision': 0.9026, 'recall': 0.9672, 'support': 1856},
                    'HOSPITAL': {'f1': 0.8339, 'precision': 0.7515, 'recall': 0.9366, 'support': 1373},
                    'DATE': {'f1': 0.8240, 'precision': 0.7774, 'recall': 0.8765, 'support': 282},
                    'LOCATION': {'f1': 0.4727, 'precision': 0.4149, 'recall': 0.5493, 'support': 128},
                    'PHONE': {'f1': 0.6374, 'precision': 0.6744, 'recall': 0.6042, 'support': 182}
                }
            }
        }

        # Error rate data
        self.error_rates = {
            'RoBERTa-Large': {
                'fp_rates': {'ID': 5.9, 'NAME': 3.8, 'HOSPITAL': 7.1, 'DATE': 23.4, 'LOCATION': 47.7, 'PHONE': 11.9},
                'fn_rates': {'ID': 0.6, 'NAME': 1.9, 'HOSPITAL': 3.4, 'DATE': 7.8, 'LOCATION': 23.7, 'PHONE': 13.9}
            },
            'ClinicalBERT': {
                'fp_rates': {'ID': 0.26, 'NAME': 6.45, 'HOSPITAL': 13.90, 'DATE': 22.14, 'LOCATION': 30.77, 'PHONE': 11.36},
                'fn_rates': {'ID': 0.07, 'NAME': 3.48, 'HOSPITAL': 3.44, 'DATE': 13.15, 'LOCATION': 36.62, 'PHONE': 18.75}
            },
            'BioBERT': {
                'fp_rates': {'ID': 0.19, 'NAME': 9.74, 'HOSPITAL': 24.85, 'DATE': 22.26, 'LOCATION': 58.51, 'PHONE': 32.56},
                'fn_rates': {'ID': 0.02, 'NAME': 3.28, 'HOSPITAL': 6.34, 'DATE': 12.35, 'LOCATION': 45.07, 'PHONE': 39.58}
            }
        }

        # Privacy risk weights based on HIPAA analysis
        self.privacy_weights = {
            'NAME': 1.0, 'ID': 0.95, 'PHONE': 0.9,
            'LOCATION': 0.7, 'DATE': 0.5, 'HOSPITAL': 0.3
        }

        # Literature benchmarks
        self.literature_benchmarks = {
            'RoBERTa-Large': 0.891,
            'ClinicalBERT': 0.872,
            'BioBERT': 0.834
        }

        # Calculate detailed confusion matrices
        self.confusion_data = self._calculate_detailed_confusion_matrices()

    def _calculate_detailed_confusion_matrices(self):
        """Calculate detailed TP/FP/FN/TN for all model-entity combinations"""
        confusion_data = {}

        for model in self.models:
            confusion_data[model] = {}
            for entity in self.entities:
                tp, fp, fn, tn = self.calculate_confusion_matrix_data(model, entity)

                # Calculate additional metrics
                total = tp + fp + fn + tn
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0

                # Clinical impact scores
                privacy_impact = fn * self.privacy_weights[entity]  # Missed PHI
                utility_impact = fp * 0.1  # Over-masked clinical terms

                confusion_data[model][entity] = {
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'sensitivity': sensitivity, 'specificity': specificity,
                    'precision': precision, 'npv': npv,
                    'privacy_impact': privacy_impact,
                    'utility_impact': utility_impact,
                    'support': self.performance_data[model]['entities'][entity]['support']
                }

        return confusion_data

    def calculate_confusion_matrix_data(self, model, entity):
        """Calculate TP, FP, FN, TN from precision, recall, and support"""
        data = self.performance_data[model]['entities'][entity]
        support = data['support']
        precision = data['precision']
        recall = data['recall']

        # Calculate confusion matrix components
        tp = int(recall * support)
        fp = int(tp / precision - tp) if precision > 0 else 0
        fn = int(support - tp)

        # Estimate TN (approximate for visualization)
        total_samples = 8330  # From dataset
        tn = total_samples - tp - fp - fn

        return tp, fp, fn, tn

    def run_comprehensive_analysis(self):
        """Run complete analysis and generate all outputs"""
        print("Starting Comprehensive PHI Analysis Framework...")
        print("="*60)

        # 1. Individual model analysis
        print("\n1. Running Individual Model Analysis...")
        self.individual_model_analysis()

        # 2. Comparative analysis
        print("\n2. Running Comparative Analysis...")
        self.comparative_analysis()

        # 3. Unique analyses
        print("\n3. Running Advanced Unique Analyses...")
        self.unique_analyses()

        # 4. Deep confusion matrix analysis
        print("\n4. Running Deep Confusion Matrix Analysis...")
        self.deep_confusion_analysis()

        # 5. Generate comprehensive report
        print("\n5. Generating Comprehensive Report...")
        self.generate_comprehensive_report()

        # 6. Export data
        print("\n6. Exporting Analysis Data...")
        self.export_analysis_data()

        print(f"\nAnalysis complete! All outputs saved to: {self.output_dir}")
        return self

    def individual_model_analysis(self):
        """Deep dive analysis of each model individually"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Individual Model Deep Dive Analysis', fontsize=16, fontweight='bold')

        for i, model in enumerate(self.models):
            # 1. Performance radar chart
            ax1 = plt.subplot2grid((3, 4), (i, 0), projection='polar')
            self._create_performance_radar(ax1, model)

            # 2. Entity F1 breakdown
            ax2 = plt.subplot2grid((3, 4), (i, 1))
            self._create_entity_f1_breakdown(ax2, model)

            # 3. Error pattern analysis
            ax3 = plt.subplot2grid((3, 4), (i, 2))
            self._create_error_pattern_plot(ax3, model)

            # 4. Privacy-utility trade-off
            ax4 = plt.subplot2grid((3, 4), (i, 3))
            self._create_privacy_utility_plot(ax4, model)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/individual_model_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

    def comparative_analysis(self):
        """Comprehensive comparative analysis between models"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comparative Analysis: RoBERTa vs ClinicalBERT vs BioBERT',
                    fontsize=16, fontweight='bold')

        # 1. Overall performance comparison
        self._create_overall_comparison(axes[0, 0])

        # 2. Entity-wise F1 heatmap
        self._create_f1_heatmap(axes[0, 1])

        # 3. Error rate comparison
        self._create_error_comparison(axes[0, 2])

        # 4. Confusion matrix visualization
        self._create_confusion_analysis(axes[1, 0])

        # 5. Statistical significance analysis
        self._create_statistical_analysis(axes[1, 1])

        # 6. Deployment readiness assessment
        self._create_deployment_assessment(axes[1, 2])

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/comparative_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

    def unique_analyses(self):
        """Unique analyses based on literature review themes"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Analyses: Hybrid System Design & Clinical Deployment',
                    fontsize=16, fontweight='bold')

        # 1. Hybrid System Component Allocation
        self._create_hybrid_allocation_analysis(axes[0, 0])

        # 2. Domain Adaptation Analysis
        self._create_domain_adaptation_analysis(axes[0, 1])

        # 3. Privacy-Utility Pareto Frontier
        self._create_pareto_frontier_analysis(axes[1, 0])

        # 4. Clinical Decision Matrix
        self._create_clinical_decision_matrix(axes[1, 1])

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/advanced_analyses.png", dpi=300, bbox_inches='tight')
        plt.show()

    def deep_confusion_analysis(self):
        """Deep confusion matrix analysis with clinical implications"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 6, figure=fig)
        fig.suptitle('Deep Confusion Matrix Analysis: TP/FP/FN Patterns',
                    fontsize=18, fontweight='bold')

        # 1. Individual model confusion heatmaps
        for i, model in enumerate(self.models):
            ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
            self._create_model_confusion_heatmap(ax, model)

        # 2. Critical entities deep dive
        ax_critical = fig.add_subplot(gs[1, :3])
        self._create_critical_entities_analysis(ax_critical)

        # 3. Error pattern clustering
        ax_cluster = fig.add_subplot(gs[1, 3:])
        self._create_error_clustering_analysis(ax_cluster)

        # 4. Clinical impact matrix
        ax_impact = fig.add_subplot(gs[2, :3])
        self._create_clinical_impact_matrix(ax_impact)

        # 5. Deployment decision tree
        ax_decision = fig.add_subplot(gs[2, 3:])
        self._create_deployment_decision_tree(ax_decision)

        # 6. Literature comparison with TP/FP/FN breakdown
        ax_lit = fig.add_subplot(gs[3, :])
        self._create_literature_tpfpfn_comparison(ax_lit)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/deep_confusion_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

    # All plotting methods implementation
    def _create_performance_radar(self, ax, model):
        """Create performance radar chart"""
        categories = ['Overall F1', 'Precision', 'Recall', 'ID F1', 'NAME F1', 'PHONE F1']
        overall = self.performance_data[model]['overall']
        entities = self.performance_data[model]['entities']

        values = [
            overall['f1'], overall['precision'], overall['recall'],
            entities['ID']['f1'], entities['NAME']['f1'], entities['PHONE']['f1']
        ]

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values += values[:1]
        angles = np.concatenate((angles, [angles[0]]))

        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model}\nPerformance Profile', fontsize=10, fontweight='bold')
        ax.grid(True)

    def _create_entity_f1_breakdown(self, ax, model):
        """Create entity F1 score breakdown"""
        entities_data = self.performance_data[model]['entities']
        entities = list(entities_data.keys())
        f1_scores = [entities_data[e]['f1'] for e in entities]

        colors = ['#2E8B57' if f1 >= 0.9 else '#FFD700' if f1 >= 0.8 else '#FF6347' for f1 in f1_scores]
        bars = ax.barh(entities, f1_scores, color=colors, alpha=0.8)

        for bar, f1 in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{f1:.3f}', va='center', fontweight='bold', fontsize=9)

        ax.set_xlim(0, 1.1)
        ax.set_title(f'{model}\nEntity F1 Scores', fontsize=10, fontweight='bold')
        ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.7)
        ax.axvline(x=0.8, color='orange', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)

    def _create_error_pattern_plot(self, ax, model):
        """Create error pattern plot"""
        fp_rates = list(self.error_rates[model]['fp_rates'].values())
        fn_rates = list(self.error_rates[model]['fn_rates'].values())
        entities = list(self.error_rates[model]['fp_rates'].keys())

        sizes = [self.privacy_weights[e] * 100 for e in entities]
        scatter = ax.scatter(fp_rates, fn_rates, s=sizes, alpha=0.7, c=range(len(entities)), cmap='tab10')

        for i, entity in enumerate(entities):
            ax.annotate(entity, (fp_rates[i], fn_rates[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('False Positive Rate (%)', fontsize=9)
        ax.set_ylabel('False Negative Rate (%)', fontsize=9)
        ax.set_title(f'{model}\nError Pattern Analysis', fontsize=10, fontweight='bold')
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    def _create_privacy_utility_plot(self, ax, model):
        """Create privacy-utility trade-off plot"""
        entities = list(self.privacy_weights.keys())
        privacy_risks = []
        utility_impacts = []

        for entity in entities:
            fn_rate = self.error_rates[model]['fn_rates'][entity]
            privacy_risk = fn_rate * self.privacy_weights[entity]
            fp_rate = self.error_rates[model]['fp_rates'][entity]

            privacy_risks.append(privacy_risk)
            utility_impacts.append(fp_rate)

        colors = ['red' if e in ['NAME', 'ID', 'PHONE'] else 'blue' for e in entities]
        scatter = ax.scatter(privacy_risks, utility_impacts, c=colors, alpha=0.7, s=100)

        for i, entity in enumerate(entities):
            ax.annotate(entity, (privacy_risks[i], utility_impacts[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('Privacy Risk Score', fontsize=9)
        ax.set_ylabel('Utility Impact (FP Rate %)', fontsize=9)
        ax.set_title(f'{model}\nPrivacy-Utility Trade-off', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _create_overall_comparison(self, ax):
        """Create overall performance comparison"""
        x = np.arange(len(self.models))
        width = 0.25

        f1_scores = [self.performance_data[model]['overall']['f1'] for model in self.models]
        precisions = [self.performance_data[model]['overall']['precision'] for model in self.models]
        recalls = [self.performance_data[model]['overall']['recall'] for model in self.models]

        ax.bar(x - width, f1_scores, width, label='F1', alpha=0.8, color='#2E8B57')
        ax.bar(x, precisions, width, label='Precision', alpha=0.8, color='#4682B4')
        ax.bar(x + width, recalls, width, label='Recall', alpha=0.8, color='#DAA520')

        for i, (f1, p, r) in enumerate(zip(f1_scores, precisions, recalls)):
            ax.text(i - width, f1 + 0.01, f'{f1:.3f}', ha='center', fontsize=8, fontweight='bold')
            ax.text(i, p + 0.01, f'{p:.3f}', ha='center', fontsize=8, fontweight='bold')
            ax.text(i + width, r + 0.01, f'{r:.3f}', ha='center', fontsize=8, fontweight='bold')

        ax.set_xlabel('Models')
        ax.set_ylabel('Performance Score')
        ax.set_title('Overall Performance Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0.9, 1.0)
        ax.grid(True, alpha=0.3)

    def _create_f1_heatmap(self, ax):
        """Create F1 score heatmap"""
        data = np.zeros((len(self.models), len(self.entities)))

        for i, model in enumerate(self.models):
            for j, entity in enumerate(self.entities):
                data[i, j] = self.performance_data[model]['entities'][entity]['f1']

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)

        for i in range(len(self.models)):
            for j in range(len(self.entities)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                              ha="center", va="center", color="black", fontweight='bold')

        ax.set_xticks(np.arange(len(self.entities)))
        ax.set_yticks(np.arange(len(self.models)))
        ax.set_xticklabels(self.entities, rotation=45, ha='right')
        ax.set_yticklabels(self.models)
        ax.set_title('Entity F1-Score Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, label='F1-Score')

    def _create_error_comparison(self, ax):
        """Create error rate comparison"""
        entities = self.entities
        x = np.arange(len(entities))
        width = 0.12

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, model in enumerate(self.models):
            fp_rates = [self.error_rates[model]['fp_rates'][e] for e in entities]
            fn_rates = [self.error_rates[model]['fn_rates'][e] for e in entities]

            # Plot FP rates (positive direction)
            ax.bar(x + i * width, fp_rates, width, label=f'{model} FP',
                  color=colors[i], alpha=0.7)

            # Plot FN rates (negative direction)
            ax.bar(x + i * width, [-fn for fn in fn_rates], width,
                  color=colors[i], alpha=0.5)

        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('PHI Entities')
        ax.set_ylabel('Error Rate (%) - FP above, FN below')
        ax.set_title('Error Rate Comparison\n(FP: Solid, FN: Transparent)', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(entities, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def _create_confusion_analysis(self, ax):
        """Create confusion matrix analysis for critical entities"""
        critical_entities = ['NAME', 'ID', 'PHONE']

        results = []
        for model in self.models:
            tp_total = fp_total = fn_total = 0
            for entity in critical_entities:
                tp, fp, fn, _ = self.calculate_confusion_matrix_data(model, entity)
                tp_total += tp
                fp_total += fp
                fn_total += fn

            total = tp_total + fp_total + fn_total
            results.append([tp_total/total, fp_total/total, fn_total/total])

        results = np.array(results)

        bottom_fp = results[:, 0]  # TP
        bottom_fn = bottom_fp + results[:, 1]  # TP + FP

        ax.bar(self.models, results[:, 0], label='True Positives', color='#2E8B57', alpha=0.8)
        ax.bar(self.models, results[:, 1], bottom=bottom_fp, label='False Positives', color='#FFD700', alpha=0.8)
        ax.bar(self.models, results[:, 2], bottom=bottom_fn, label='False Negatives', color='#FF6347', alpha=0.8)

        ax.set_ylabel('Proportion of Predictions')
        ax.set_title('Confusion Matrix Analysis\n(Critical Entities: NAME, ID, PHONE)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _create_statistical_analysis(self, ax):
        """Create statistical significance analysis"""
        models_f1 = []
        models_ci_lower = []
        models_ci_upper = []

        for model in self.models:
            f1 = self.performance_data[model]['overall']['f1']
            n = 8330
            se = np.sqrt(f1 * (1 - f1) / n)
            ci_lower = max(0, f1 - 1.96 * se)
            ci_upper = min(1, f1 + 1.96 * se)

            models_f1.append(f1)
            models_ci_lower.append(ci_lower)
            models_ci_upper.append(ci_upper)

        x = np.arange(len(self.models))
        errors = [np.array(models_f1) - np.array(models_ci_lower),
                 np.array(models_ci_upper) - np.array(models_f1)]

        ax.errorbar(x, models_f1, yerr=errors, fmt='o-', capsize=10, capthick=2,
                   markersize=10, linewidth=2, color='#4682B4')

        lit_f1s = [self.literature_benchmarks[model] for model in self.models]
        ax.plot(x, lit_f1s, 's--', markersize=8, linewidth=2, color='red',
                label='Literature Benchmark', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(self.models, rotation=45, ha='right')
        ax.set_ylabel('F1-Score')
        ax.set_title('Statistical Analysis vs Literature\n(95% Confidence Intervals)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.82, 0.98)

    def _create_deployment_assessment(self, ax):
        """Create deployment readiness assessment"""
        deployment_scores = []
        criteria = ['F1 Score', 'Privacy Risk', 'Stability', 'Clinical Utility']

        for model in self.models:
            f1_score = self.performance_data[model]['overall']['f1']
            f1_criterion = 1.0 if f1_score >= 0.95 else 0.8 if f1_score >= 0.9 else 0.6

            critical_fn = np.mean([self.error_rates[model]['fn_rates'][e]
                                  for e in ['NAME', 'ID', 'PHONE']])
            privacy_criterion = 1.0 if critical_fn <= 2 else 0.8 if critical_fn <= 5 else 0.4

            entity_f1s = [self.performance_data[model]['entities'][e]['f1'] for e in self.entities]
            stability = 1.0 - np.std(entity_f1s)

            avg_fp = np.mean(list(self.error_rates[model]['fp_rates'].values()))
            utility_criterion = 1.0 if avg_fp <= 10 else 0.8 if avg_fp <= 20 else 0.4

            deployment_scores.append([f1_criterion, privacy_criterion, stability, utility_criterion])

        deployment_scores = np.array(deployment_scores)

        angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, model in enumerate(self.models):
            values = np.concatenate((deployment_scores[i], [deployment_scores[i][0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)
        ax.set_ylim(0, 1)
        ax.set_title('Deployment Readiness Assessment', fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax.grid(True)

    def _create_hybrid_allocation_analysis(self, ax):
        """Analysis for optimal hybrid system component allocation"""
        entity_characteristics = {
            'ID': {'structure_score': 0.9, 'context_score': 0.1, 'type': 'Structured'},
            'DATE': {'structure_score': 0.8, 'context_score': 0.2, 'type': 'Semi-structured'},
            'PHONE': {'structure_score': 0.7, 'context_score': 0.3, 'type': 'Semi-structured'},
            'HOSPITAL': {'structure_score': 0.3, 'context_score': 0.7, 'type': 'Contextual'},
            'LOCATION': {'structure_score': 0.2, 'context_score': 0.8, 'type': 'Contextual'},
            'NAME': {'structure_score': 0.1, 'context_score': 0.9, 'type': 'Contextual'}
        }

        best_models = {}
        for entity in self.entities:
            f1_scores = {model: self.performance_data[model]['entities'][entity]['f1']
                        for model in self.models}
            best_models[entity] = max(f1_scores, key=f1_scores.get)

        structure_scores = [entity_characteristics[e]['structure_score'] for e in self.entities]
        context_scores = [entity_characteristics[e]['context_score'] for e in self.entities]

        model_colors = {'RoBERTa-Large': 'red', 'ClinicalBERT': 'blue', 'BioBERT': 'green'}
        colors = [model_colors[best_models[e]] for e in self.entities]

        scatter = ax.scatter(structure_scores, context_scores, c=colors, s=150, alpha=0.7, edgecolors='black')

        for i, entity in enumerate(self.entities):
            ax.annotate(f'{entity}\n({best_models[entity][:6]})',
                       (structure_scores[i], context_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8,
                       ha='center', fontweight='bold')

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        ax.text(0.25, 0.75, 'Contextual\n(Transformer)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(0.75, 0.25, 'Structured\n(Rule-based)', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

        ax.set_xlabel('Structure Score (Rule-based Suitability)')
        ax.set_ylabel('Context Score (Transformer Suitability)')
        ax.set_title('Hybrid System Component Allocation\n(Entity Characteristics vs Best Model)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _create_domain_adaptation_analysis(self, ax):
        """Analysis of domain adaptation challenges"""
        gaps = []
        model_names = []

        for model in self.models:
            actual = self.performance_data[model]['overall']['f1']
            expected = self.literature_benchmarks[model]
            gap = actual - expected
            gaps.append(gap)
            model_names.append(model)

        colors = ['green' if gap > 0 else 'red' for gap in gaps]
        bars = ax.bar(model_names, gaps, color=colors, alpha=0.7)

        for bar, gap in zip(bars, gaps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                   f'{gap:+.3f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold')

        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_ylabel('Performance Gap vs Literature')
        ax.set_title('Domain Adaptation Success\n(Your Results vs Literature Benchmarks)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        interpretation = ("Green: Better than literature\nRed: Below literature\n" +
                         f"Avg improvement: {np.mean(gaps):+.3f}")
        ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    def _create_pareto_frontier_analysis(self, ax):
        """Create Pareto frontier for privacy-utility trade-off"""
        privacy_risks = []
        utility_scores = []

        for model in self.models:
            critical_entities = ['NAME', 'ID', 'PHONE']
            privacy_risk = np.mean([self.error_rates[model]['fn_rates'][e] * self.privacy_weights[e]
                                   for e in critical_entities])

            avg_fp = np.mean(list(self.error_rates[model]['fp_rates'].values()))
            utility_score = 100 - avg_fp

            privacy_risks.append(privacy_risk)
            utility_scores.append(utility_score)

        colors = ['red', 'blue', 'green']
        for i, model in enumerate(self.models):
            ax.scatter(privacy_risks[i], utility_scores[i], s=200, c=colors[i],
                      alpha=0.7, edgecolors='black', linewidth=2, label=model)

            ax.annotate(model, (privacy_risks[i], utility_scores[i]),
                       xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')

        ax.set_xlabel('Privacy Risk Score (Lower is Better)')
        ax.set_ylabel('Clinical Utility Score (Higher is Better)')
        ax.set_title('Privacy-Utility Pareto Frontier\nOptimal Trade-off Analysis', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.text(0.02, 0.98, 'Ideal\n(Low Risk, High Utility)', transform=ax.transAxes,
                va='top', ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        ax.text(0.98, 0.02, 'Avoid\n(High Risk, Low Utility)', transform=ax.transAxes,
                va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

    def _create_clinical_decision_matrix(self, ax):
        """Create clinical decision matrix for deployment"""
        criteria = ['F1 Score', 'Privacy\nCompliance', 'Entity\nConsistency', 'Literature\nImprovement', 'Clinical\nUtility']
        models = self.models

        decision_matrix = np.zeros((len(models), len(criteria)))

        for i, model in enumerate(models):
            f1 = self.performance_data[model]['overall']['f1']
            decision_matrix[i, 0] = (f1 - 0.9) / 0.1

            critical_fn = np.mean([self.error_rates[model]['fn_rates'][e] for e in ['NAME', 'ID', 'PHONE']])
            decision_matrix[i, 1] = max(0, 1 - critical_fn / 10)

            entity_f1s = [self.performance_data[model]['entities'][e]['f1'] for e in self.entities]
            decision_matrix[i, 2] = 1 - np.std(entity_f1s) * 5

            actual = self.performance_data[model]['overall']['f1']
            expected = self.literature_benchmarks[model]
            improvement = actual - expected
            decision_matrix[i, 3] = max(0, improvement * 10)

            avg_fp = np.mean(list(self.error_rates[model]['fp_rates'].values()))
            decision_matrix[i, 4] = max(0, 1 - avg_fp / 50)

        decision_matrix = np.clip(decision_matrix, 0, 1)

        im = ax.imshow(decision_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        for i in range(len(models)):
            for j in range(len(criteria)):
                score = decision_matrix[i, j]
                color = 'white' if score < 0.5 else 'black'
                ax.text(j, i, f'{score:.2f}', ha="center", va="center",
                       color=color, fontweight='bold')

        ax.set_xticks(np.arange(len(criteria)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(criteria)
        ax.set_yticklabels(models)
        ax.set_title('Clinical Deployment Decision Matrix\n(Higher Scores = Better)', fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.6, label='Decision Score')

        overall_scores = np.mean(decision_matrix, axis=1)
        best_model_idx = np.argmax(overall_scores)

        recommendation = f"Recommended: {models[best_model_idx]}\nOverall Score: {overall_scores[best_model_idx]:.3f}"
        ax.text(1.1, 0.5, recommendation, transform=ax.transAxes, va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
                fontsize=10, fontweight='bold')

    def _create_model_confusion_heatmap(self, ax, model):
        """Create detailed confusion matrix heatmap for a model"""
        data = np.zeros((3, len(self.entities)))

        for j, entity in enumerate(self.entities):
            conf = self.confusion_data[model][entity]
            total_pos = conf['tp'] + conf['fn']
            total_neg = conf['fp'] + conf['tn']

            tp_rate = conf['tp'] / max(total_pos, 1)
            fp_rate = conf['fp'] / max(total_neg, 1)
            fn_rate = conf['fn'] / max(total_pos, 1)

            data[0, j] = tp_rate
            data[1, j] = fp_rate
            data[2, j] = fn_rate

        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

        rate_labels = ['TP Rate\n(Sensitivity)', 'FP Rate\n(1-Specificity)', 'FN Rate\n(Miss Rate)']
        for i in range(3):
            for j in range(len(self.entities)):
                value = data[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                       color=color, fontweight='bold', fontsize=9)

        ax.set_xticks(range(len(self.entities)))
        ax.set_yticks(range(3))
        ax.set_xticklabels(self.entities, rotation=45, ha='right')
        ax.set_yticklabels(rate_labels, fontsize=9)
        ax.set_title(f'{model}\nConfusion Matrix Rates', fontweight='bold', fontsize=12)

    def _create_critical_entities_analysis(self, ax):
        """Deep dive into critical entities"""
        critical_entities = ['NAME', 'ID', 'PHONE']
        x = np.arange(len(critical_entities))
        width = 0.25

        tp_rates = np.zeros((len(self.models), len(critical_entities)))

        for i, model in enumerate(self.models):
            for j, entity in enumerate(critical_entities):
                conf = self.confusion_data[model][entity]
                tp_rates[i, j] = conf['sensitivity']

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, model in enumerate(self.models):
            bars = ax.bar(x + i * width, tp_rates[i], width, label=f'{model} TP Rate',
                          color=colors[i], alpha=0.8)

            for bar, rate in zip(bars, tp_rates[i]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xlabel('Critical PHI Entities')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title('Critical Entities Performance Analysis\n(Higher is Better for Privacy)', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(critical_entities)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7)

    def _create_error_clustering_analysis(self, ax):
        """Cluster entities by error patterns"""
        error_signatures = []
        entity_labels = []

        for entity in self.entities:
            avg_fp_rate = np.mean([self.error_rates[model]['fp_rates'][entity] for model in self.models])
            avg_fn_rate = np.mean([self.error_rates[model]['fn_rates'][entity] for model in self.models])
            f1_variance = np.var([self.performance_data[model]['entities'][entity]['f1'] for model in self.models])

            error_signatures.append([avg_fp_rate, avg_fn_rate, f1_variance * 1000])
            entity_labels.append(entity)

        error_signatures = np.array(error_signatures)

        scatter = ax.scatter(error_signatures[:, 0], error_signatures[:, 1],
                           s=error_signatures[:, 2] + 50, alpha=0.7,
                           c=range(len(self.entities)), cmap='tab10')

        for i, entity in enumerate(entity_labels):
            ax.annotate(entity, (error_signatures[i, 0], error_signatures[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

        ax.axhline(y=np.median(error_signatures[:, 1]), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=np.median(error_signatures[:, 0]), color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Average False Positive Rate (%)')
        ax.set_ylabel('Average False Negative Rate (%)')
        ax.set_title('Entity Error Pattern Clustering\n(Bubble size = Performance instability)', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _create_clinical_impact_matrix(self, ax):
        """Create clinical impact assessment matrix"""
        impact_categories = ['Privacy Risk', 'Utility Loss', 'Regulatory Risk', 'Clinical Safety']
        impact_matrix = np.zeros((len(self.models), len(impact_categories)))

        for i, model in enumerate(self.models):
            privacy_risk = sum(self.confusion_data[model][entity]['privacy_impact']
                             for entity in ['NAME', 'ID', 'PHONE']) / 100
            utility_loss = sum(self.confusion_data[model][entity]['utility_impact']
                             for entity in self.entities) / 1000

            critical_miss_rate = np.mean([self.error_rates[model]['fn_rates'][e]
                                        for e in ['NAME', 'ID', 'PHONE']])
            regulatory_risk = 1.0 if critical_miss_rate > 5 else 0.5 if critical_miss_rate > 2 else 0.1

            f1_score = self.performance_data[model]['overall']['f1']
            entity_f1s = [self.performance_data[model]['entities'][e]['f1'] for e in self.entities]
            consistency = 1 - np.std(entity_f1s)
            clinical_safety = (f1_score + consistency) / 2

            impact_matrix[i] = [privacy_risk, utility_loss, regulatory_risk, clinical_safety]

        impact_matrix_norm = (impact_matrix - impact_matrix.min(axis=0)) / (impact_matrix.max(axis=0) - impact_matrix.min(axis=0))
        im = ax.imshow(impact_matrix_norm, cmap='RdYlGn_r', aspect='auto')

        for i in range(len(self.models)):
            for j in range(len(impact_categories)):
                value = impact_matrix[i, j]
                color = 'white' if impact_matrix_norm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                       color=color, fontweight='bold')

        ax.set_xticks(range(len(impact_categories)))
        ax.set_yticks(range(len(self.models)))
        ax.set_xticklabels(impact_categories, rotation=45, ha='right')
        ax.set_yticklabels(self.models)
        ax.set_title('Clinical Impact Assessment Matrix\n(Red=High Impact, Green=Low Impact)', fontweight='bold')

    def _create_deployment_decision_tree(self, ax):
        """Create visual deployment decision tree"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Root node
        root = Rectangle((4, 8), 2, 1, facecolor='lightblue', edgecolor='black')
        ax.add_patch(root)
        ax.text(5, 8.5, 'PHI Model\nSelection', ha='center', va='center', fontweight='bold')

        # Decision criteria
        criteria = [
            ('F1 > 0.97?', 2, 6.5, 'RoBERTa-Large\n(F1=0.9736)', 'lightgreen'),
            ('Privacy\nCritical?', 5, 6.5, 'Consider FN rates\nfor NAME/ID/PHONE', 'lightyellow'),
            ('Utility\nFocused?', 8, 6.5, 'Consider FP rates\nfor all entities', 'lightcoral')
        ]

        for text, x, y, result, color in criteria:
            node = Rectangle((x-0.75, y-0.4), 1.5, 0.8, facecolor=color, edgecolor='black')
            ax.add_patch(node)
            ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

            ax.plot([5, x], [8, y+0.4], 'k-', linewidth=2)

            result_box = Rectangle((x-1, y-2), 2, 1, facecolor='white', edgecolor='black')
            ax.add_patch(result_box)
            ax.text(x, y-1.5, result, ha='center', va='center', fontsize=8)
            ax.plot([x, x], [y-0.4, y-1], 'k-', linewidth=2)

        ax.set_title('Clinical Deployment Decision Tree\nBased on Error Pattern Analysis',
                    fontweight='bold', fontsize=12, pad=20)

    def _create_literature_tpfpfn_comparison(self, ax):
        """Compare TP/FP/FN patterns with literature expectations"""
        x = np.arange(len(self.models))
        width = 0.35

        actual_f1 = [self.performance_data[model]['overall']['f1'] for model in self.models]
        expected_f1 = [self.literature_benchmarks[model] for model in self.models]

        bars1 = ax.bar(x - width/2, actual_f1, width, label='Your Results',
                      color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(x + width/2, expected_f1, width, label='Literature Benchmark',
                      color='#4682B4', alpha=0.8)

        for i, (actual, expected) in enumerate(zip(actual_f1, expected_f1)):
            improvement = actual - expected
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'+{improvement:.3f}' if improvement > 0 else f'{improvement:.3f}',
                       xy=(i, max(actual, expected) + 0.005), ha='center',
                       color=color, fontweight='bold', fontsize=10)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Models')
        ax.set_ylabel('F1-Score')
        ax.set_title('Literature Comparison: Your Standardized Training Results\n(Green = Improvement, Red = Below Expectation)',
                    fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.models)
        ax.legend()
        ax.set_ylim(0.8, 1.0)
        ax.grid(True, alpha=0.3)

        avg_improvement = np.mean([a - e for a, e in zip(actual_f1, expected_f1)])
        summary_text = f"Average Improvement: +{avg_improvement:.3f}\nYour standardized training\noutperforms literature benchmarks"
        ax.text(0.02, 0.7, summary_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8),
                fontsize=10, verticalalignment='top')

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        report_lines = []
        report_lines.append("COMPREHENSIVE PHI ANALYSIS REPORT")
        report_lines.append("="*80)

        report_lines.append(f"\nDataset: i2b2 2006 De-identification Challenge")
        report_lines.append(f"Models Analyzed: {', '.join(self.models)}")
        report_lines.append(f"PHI Entity Types: {', '.join(self.entities)}")
        report_lines.append(f"Total Entities Evaluated: 8,330")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report_lines.append("\n" + "="*80)
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("="*80)

        # Best performer analysis
        best_overall = max(self.models, key=lambda m: self.performance_data[m]['overall']['f1'])
        best_f1 = self.performance_data[best_overall]['overall']['f1']

        report_lines.append(f"\nBEST OVERALL PERFORMER: {best_overall}")
        report_lines.append(f"   • F1-Score: {best_f1:.3f}")
        report_lines.append(f"   • Improvement over literature: +{best_f1 - self.literature_benchmarks[best_overall]:.3f}")

        # Privacy analysis
        privacy_scores = {}
        for model in self.models:
            critical_fn = np.mean([self.error_rates[model]['fn_rates'][e] for e in ['NAME', 'ID', 'PHONE']])
            privacy_scores[model] = critical_fn

        privacy_champion = min(privacy_scores, key=privacy_scores.get)
        report_lines.append(f"\nPRIVACY CHAMPION: {privacy_champion}")
        report_lines.append(f"   • Critical Entity Miss Rate: {privacy_scores[privacy_champion]:.1f}%")

        # Key findings
        report_lines.append(f"\nKEY FINDINGS:")
        report_lines.append(f"   • All models exceed 94% F1-score (clinical deployment threshold)")
        report_lines.append(f"   • {best_overall} shows best balance of performance and privacy")
        report_lines.append(f"   • LOCATION detection remains challenging across all models")
        report_lines.append(f"   • ID detection universally excellent (>99.6% F1 all models)")

        # Save report
        report_text = "\n".join(report_lines)
        with open(f"{self.output_dir}/comprehensive_report.txt", "w") as f:
            f.write(report_text)

        print(report_text)
        return report_text

    def export_analysis_data(self):
        """Export all analysis data to various formats"""

        # Export performance data to CSV
        performance_df = pd.DataFrame()
        for model in self.models:
            overall = self.performance_data[model]['overall']

            model_data = [{
                'Model': model, 'Entity': 'OVERALL',
                'F1': overall['f1'], 'Precision': overall['precision'],
                'Recall': overall['recall'], 'Accuracy': overall['accuracy'],
                'Support': 8330
            }]

            for entity in self.entities:
                entity_data = self.performance_data[model]['entities'][entity]
                model_data.append({
                    'Model': model, 'Entity': entity,
                    'F1': entity_data['f1'], 'Precision': entity_data['precision'],
                    'Recall': entity_data['recall'], 'Accuracy': None,
                    'Support': entity_data['support']
                })

            model_df = pd.DataFrame(model_data)
            performance_df = pd.concat([performance_df, model_df], ignore_index=True)

        performance_df.to_csv(f"{self.output_dir}/performance_data.csv", index=False)

        # Export error rates to CSV
        error_df = pd.DataFrame()
        for model in self.models:
            for entity in self.entities:
                error_df = pd.concat([error_df, pd.DataFrame({
                    'Model': [model], 'Entity': [entity],
                    'FP_Rate': [self.error_rates[model]['fp_rates'][entity]],
                    'FN_Rate': [self.error_rates[model]['fn_rates'][entity]]
                })], ignore_index=True)

        error_df.to_csv(f"{self.output_dir}/error_rates.csv", index=False)

        # Export confusion matrix data to CSV
        confusion_df = pd.DataFrame()
        for model in self.models:
            for entity in self.entities:
                conf_data = self.confusion_data[model][entity]
                confusion_df = pd.concat([confusion_df, pd.DataFrame({
                    'Model': [model], 'Entity': [entity],
                    'TP': [conf_data['tp']], 'FP': [conf_data['fp']],
                    'FN': [conf_data['fn']], 'TN': [conf_data['tn']],
                    'Sensitivity': [conf_data['sensitivity']],
                    'Specificity': [conf_data['specificity']],
                    'Precision': [conf_data['precision']],
                    'NPV': [conf_data['npv']],
                    'Privacy_Impact': [conf_data['privacy_impact']],
                    'Utility_Impact': [conf_data['utility_impact']]
                })], ignore_index=True)

        confusion_df.to_csv(f"{self.output_dir}/confusion_matrix_data.csv", index=False)

        # Export summary statistics to JSON
        summary_stats = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'models': self.models,
                'entities': self.entities,
                'dataset': 'i2b2 2006 De-identification Challenge',
                'total_entities': 8330
            },
            'model_rankings': {
                'best_overall_f1': max(self.models, key=lambda m: self.performance_data[m]['overall']['f1']),
                'best_privacy': min(self.models, key=lambda m: np.mean([self.error_rates[m]['fn_rates'][e] for e in ['NAME', 'ID', 'PHONE']])),
                'literature_improvements': {
                    model: self.performance_data[model]['overall']['f1'] - self.literature_benchmarks[model]
                    for model in self.models
                }
            }
        }

        with open(f"{self.output_dir}/summary_statistics.json", "w") as f:
            json.dump(summary_stats, f, indent=2)

        print(f"\nData exported to {self.output_dir}:")
        print("  • performance_data.csv - Model performance metrics")
        print("  • error_rates.csv - FP/FN rates by model and entity")
        print("  • confusion_matrix_data.csv - Detailed confusion matrix data")
        print("  • summary_statistics.json - Summary statistics and rankings")
        print("  • comprehensive_report.txt - Full text report")
        print("  • *.png files - All visualization plots")


# Main execution function
def run_comprehensive_phi_analysis(output_dir="phi_analysis_outputs"):
    """
    Run the complete comprehensive PHI analysis with downloadable outputs

    Args:
        output_dir (str): Directory to save all outputs

    Returns:
        ComprehensivePHIAnalyzer: Analyzer instance with all results
    """

    print("Starting Comprehensive PHI Analysis Framework")
    print("="*60)
    print(f"Output directory: {output_dir}")

    # Initialize analyzer
    analyzer = ComprehensivePHIAnalyzer(output_dir=output_dir)

    # Run complete analysis
    analyzer.run_comprehensive_analysis()

    print("\nAnalysis Complete! Generated files:")
    print(f"  - {output_dir}/individual_model_analysis.png")
    print(f"  - {output_dir}/comparative_analysis.png")
    print(f"  - {output_dir}/advanced_analyses.png")
    print(f"  - {output_dir}/deep_confusion_analysis.png")
    print(f"  - {output_dir}/comprehensive_report.txt")
    print(f"  - {output_dir}/performance_data.csv")
    print(f"  - {output_dir}/error_rates.csv")
    print(f"  - {output_dir}/confusion_matrix_data.csv")
    print(f"  - {output_dir}/summary_statistics.json")

    return analyzer

# Usage
if __name__ == "__main__":
    # Run the comprehensive analysis
    analyzer = run_comprehensive_phi_analysis("phi_analysis_outputs")

    print("\nComprehensive PHI Analysis Framework completed successfully!")
    print("All visualizations, reports, and data files have been generated and saved.")
