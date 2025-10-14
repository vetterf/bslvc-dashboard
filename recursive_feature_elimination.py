#!/usr/bin/env python3
"""
Recursive Feature Elimination with Random Forest and Permutation Importance
for BSLVC Grammar Data

This script loads the imputed grammatical data from the SQLite database,
applies recursive feature elimination using Random Forest with permutation
importance, and uses MainVariety as the grouping variable.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Add the pages directory to the path to import retrieve_data
sys.path.append(os.path.join(os.path.dirname(__file__), 'pages'))
import pages.data.retrieve_data as retrieve_data

class GrammarFeatureSelection:
    """
    Class to perform recursive feature elimination on grammatical data
    using Random Forest with permutation importance.
    """
    
    def __init__(self, random_state=42, mode_filter=None, exclude_varieties=None, exclude_items=None):
        """
        Initialize the feature selection analysis.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        mode_filter : str or None
            Filter for mode: 'spoken', 'written', or None for both
        exclude_varieties : list or None
            List of varieties to exclude. Defaults to ['Other'] if None.
        exclude_items : list or None
            List of specific grammatical items to exclude from features.
        """
        self.random_state = random_state
        self.mode_filter = mode_filter
        self.exclude_varieties = exclude_varieties if exclude_varieties is not None else ['Other']
        self.exclude_items = exclude_items if exclude_items is not None else []
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.rf_model = None
        self.rfe_model = None
        self.selected_features = None
        self.feature_importance_scores = None
        
    def get_available_items(self):
        """Get list of available grammatical items for the current mode."""
        if self.mode_filter == 'spoken':
            grammar_cols = retrieve_data.getGrammarItemsCols(type="spoken")
        elif self.mode_filter == 'written':
            grammar_cols = retrieve_data.getGrammarItemsCols(type="written")
        else:
            grammar_cols = retrieve_data.getGrammarItemsCols(type="all")
        
        # Filter to only include grammar columns that exist in the data
        if self.data is not None:
            available_grammar_cols = [col for col in grammar_cols if col in self.data.columns]
        else:
            available_grammar_cols = grammar_cols
            
        return available_grammar_cols
    
    def load_data(self):
        """Load imputed grammatical data from the database."""
        print("Loading imputed grammatical data...")
        
        # Load the imputed grammar data with all participants
        self.data = retrieve_data.getGrammarData(imputed=True)
        
        print(f"Initial data shape: {self.data.shape}")
        print(f"Initial MainVariety distribution:")
        print(self.data['MainVariety'].value_counts())
        
        # Filter out excluded varieties
        if self.exclude_varieties:
            print(f"\nExcluding varieties: {self.exclude_varieties}")
            initial_count = len(self.data)
            self.data = self.data[~self.data['MainVariety'].isin(self.exclude_varieties)]
            excluded_count = initial_count - len(self.data)
            print(f"Excluded {excluded_count} participants from varieties: {self.exclude_varieties}")
        
        print(f"Final data shape after variety filtering: {self.data.shape}")
        print(f"MainVariety distribution after filtering:")
        print(self.data['MainVariety'].value_counts())
        
        return self.data
    
    def prepare_features_target(self):
        """Prepare features (grammar items) and target (MainVariety)."""
        print("\nPreparing features and target variable...")
        
        # Get grammar item columns based on mode filter
        if self.mode_filter == 'spoken':
            print("Using only SPOKEN features...")
            grammar_cols = retrieve_data.getGrammarItemsCols(type="spoken")
        elif self.mode_filter == 'written':
            print("Using only WRITTEN features...")
            grammar_cols = retrieve_data.getGrammarItemsCols(type="written")
        else:
            print("Using ALL features (spoken + written)...")
            grammar_cols = retrieve_data.getGrammarItemsCols(type="all")
        
        # Filter to only include grammar columns that exist in the data
        available_grammar_cols = [col for col in grammar_cols if col in self.data.columns]
        
        # Exclude specific grammatical items if specified
        if self.exclude_items:
            print(f"Excluding specific grammatical items: {self.exclude_items}")
            initial_count = len(available_grammar_cols)
            available_grammar_cols = [col for col in available_grammar_cols if col not in self.exclude_items]
            excluded_count = initial_count - len(available_grammar_cols)
            print(f"Excluded {excluded_count} grammatical items")
        
        print(f"Total grammar items available for mode '{self.mode_filter}': {len(available_grammar_cols)}")
        
        # Check if we have enough features left
        if len(available_grammar_cols) < 5:
            raise ValueError(f"Too few features remaining ({len(available_grammar_cols)}). "
                           "Consider reducing item exclusions or changing mode filter.")
        
        # Prepare features (X) - grammar items only
        self.X = self.data[available_grammar_cols].copy()
        
        # Remove any rows with missing values in grammar items
        initial_rows = len(self.X)
        self.X = self.X.dropna()
        final_rows = len(self.X)
        
        if initial_rows != final_rows:
            print(f"Removed {initial_rows - final_rows} rows with missing grammar data")
        
        # Prepare target (y) - MainVariety
        # Get the corresponding MainVariety values for the remaining rows
        self.y = self.data.loc[self.X.index, 'MainVariety'].copy()
        
        # Remove any rows where MainVariety is missing
        valid_mask = self.y.notna()
        self.X = self.X[valid_mask]
        self.y = self.y[valid_mask]
        
        # Encode MainVariety labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Store feature names
        self.feature_names = list(self.X.columns)
        
        print(f"Final dataset shape: {self.X.shape}")
        print(f"Features: {len(self.feature_names)} grammar items")
        print(f"Target classes: {list(self.label_encoder.classes_)}")
        print(f"Class distribution: {dict(zip(*np.unique(self.y_encoded, return_counts=True)))}")
        
        # Check if we have enough samples per class
        min_samples = min(np.bincount(self.y_encoded))
        if min_samples < 5:
            print(f"WARNING: Some classes have very few samples (minimum: {min_samples})")
            print("Consider adjusting variety exclusions or mode filter.")
        
        return self.X, self.y_encoded
    
    def train_initial_model(self, n_estimators=100):
        """Train initial Random Forest model to get baseline performance."""
        print(f"\nTraining initial Random Forest model with {n_estimators} estimators...")
        
        # Split data for initial evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.3, random_state=self.random_state, 
            stratify=self.y_encoded
        )
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate baseline performance
        y_pred = self.rf_model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.rf_model, self.X, self.y_encoded, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='accuracy'
        )
        
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return baseline_accuracy, cv_scores
    
    def calculate_permutation_importance(self, n_repeats=50):
        """Calculate permutation importance for all features."""
        print(f"\nCalculating permutation importance with {n_repeats} repeats...")
        
        # Split data for permutation importance calculation
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.3, random_state=self.random_state,
            stratify=self.y_encoded
        )
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.rf_model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        self.feature_importance_scores = importance_df
        
        print(f"Top 10 most important features:")
        print(importance_df.head(10))
        
        return importance_df
    
    def perform_rfe(self, n_features_to_select=None, step=1):
        """
        Perform Recursive Feature Elimination using Random Forest.
        
        Parameters:
        -----------
        n_features_to_select : int or None
            Number of features to select. If None, select half of the features.
        step : int
            Number of features to remove at each iteration.
        """
        if n_features_to_select is None:
            n_features_to_select = len(self.feature_names) // 2
            
        print(f"\nPerforming RFE to select {n_features_to_select} features out of {len(self.feature_names)}")
        print(f"Step size: {step}")
        
        # Create RFE with Random Forest
        self.rfe_model = RFE(
            estimator=RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1
            ),
            n_features_to_select=n_features_to_select,
            step=step,
            verbose=1
        )
        
        # Fit RFE
        self.rfe_model.fit(self.X, self.y_encoded)
        
        # Get selected features
        selected_mask = self.rfe_model.support_
        self.selected_features = [feature for feature, selected in zip(self.feature_names, selected_mask) if selected]
        
        print(f"\nSelected {len(self.selected_features)} features:")
        print(self.selected_features)
        
        # Evaluate performance with selected features
        X_selected = self.X.iloc[:, selected_mask]
        
        cv_scores_rfe = cross_val_score(
            self.rfe_model.estimator_, X_selected, self.y_encoded,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='accuracy'
        )
        
        print(f"RFE model cross-validation accuracy: {cv_scores_rfe.mean():.4f} (+/- {cv_scores_rfe.std() * 2:.4f})")
        
        return self.selected_features, cv_scores_rfe
    
    def evaluate_final_model(self):
        """Evaluate the final model with selected features."""
        print("\nEvaluating final model with selected features...")
        
        # Get selected features
        selected_mask = self.rfe_model.support_
        X_selected = self.X.iloc[:, selected_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, self.y_encoded, test_size=0.3, random_state=self.random_state,
            stratify=self.y_encoded
        )
        
        # Train final model
        final_model = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        final_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = final_model.predict(X_test)
        
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Final model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return final_model, accuracy, cm
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance from permutation importance."""
        if self.feature_importance_scores is None:
            print("No feature importance scores available. Run calculate_permutation_importance() first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Get top N features
        top_features = self.feature_importance_scores.head(top_n)
        
        # Create bar plot
        plt.barh(range(len(top_features)), top_features['importance_mean'], 
                xerr=top_features['importance_std'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_n} Feature Importance (Permutation)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plt.savefig('feature_importance_permutation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Final Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename=None):
        """Save results to CSV file."""
        if self.selected_features is None:
            print("No selected features available. Run perform_rfe() first.")
            return
        
        # Generate filename based on analysis parameters
        if filename is None:
            mode_str = f"_{self.mode_filter}" if self.mode_filter else "_all"
            exclude_str = f"_excl{'_'.join(self.exclude_varieties)}" if self.exclude_varieties else ""
            items_str = f"_excludeItems{len(self.exclude_items)}" if self.exclude_items else ""
            filename = f'grammar_rfe_results{mode_str}{exclude_str}{items_str}.csv'
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'selected_feature': self.selected_features
        })
        
        # Add importance scores if available
        if self.feature_importance_scores is not None:
            importance_dict = dict(zip(
                self.feature_importance_scores['feature'],
                self.feature_importance_scores['importance_mean']
            ))
            
            results_df['permutation_importance'] = results_df['selected_feature'].map(importance_dict)
            results_df = results_df.sort_values('permutation_importance', ascending=False)
        
        # Add analysis metadata
        results_df['mode_filter'] = self.mode_filter if self.mode_filter else 'all'
        results_df['excluded_varieties'] = str(self.exclude_varieties)
        results_df['excluded_items'] = str(self.exclude_items)
        results_df['n_excluded_items'] = len(self.exclude_items)
        
        # Save to CSV
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        return results_df


def main():
    """Main function to run the recursive feature elimination analysis."""
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Recursive Feature Elimination Analysis for BSLVC Grammar Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        choices=['spoken', 'written', 'all'], 
        default='all',
        help='Choose feature type: spoken, written, or all features'
    )
    
    parser.add_argument(
        '--exclude-varieties',
        nargs='*',
        default=['Other'],
        help='Varieties to exclude from analysis (default: Other)'
    )
    
    parser.add_argument(
        '--exclude-items',
        nargs='*',
        default=[],
        help='Specific grammatical items to exclude from features'
    )
    
    parser.add_argument(
        '--n-features',
        type=int,
        default=None,
        help='Number of features to select (default: half of available features)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=10,
        help='Number of features to remove at each RFE iteration'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--list-items',
        action='store_true',
        help='List available grammatical items and exit'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert mode
    mode_filter = None if args.mode == 'all' else args.mode
    
    # Handle list-items option
    if args.list_items:
        print(f"Available grammatical items for mode '{args.mode}':")
        print("=" * 50)
        
        # Create temporary analysis object to get items
        temp_analysis = GrammarFeatureSelection(mode_filter=mode_filter)
        temp_analysis.load_data()
        available_items = temp_analysis.get_available_items()
        
        for i, item in enumerate(sorted(available_items), 1):
            print(f"{i:3d}. {item}")
        
        print(f"\nTotal: {len(available_items)} items")
        return
    
    print("Starting Recursive Feature Elimination Analysis for BSLVC Grammar Data")
    print("=" * 70)
    print(f"Mode filter: {args.mode}")
    print(f"Excluded varieties: {args.exclude_varieties}")
    print(f"Excluded items: {args.exclude_items}")
    print(f"Random state: {args.random_state}")
    print("=" * 70)
    
    # Initialize the analysis
    rfe_analysis = GrammarFeatureSelection(
        random_state=args.random_state,
        mode_filter=mode_filter,
        exclude_varieties=args.exclude_varieties,
        exclude_items=args.exclude_items
    )
    
    # Load data
    data = rfe_analysis.load_data()
    
    # Check if we have enough data after filtering
    if len(data) < 50:
        print(f"WARNING: Only {len(data)} samples remaining after filtering.")
        print("Consider adjusting your filters.")
        return
    
    # Prepare features and target
    X, y = rfe_analysis.prepare_features_target()
    
    # Check if we have enough features
    if len(rfe_analysis.feature_names) < 10:
        print(f"WARNING: Only {len(rfe_analysis.feature_names)} features available.")
        print("Consider using 'all' mode or different filters.")
        return
    
    # Train initial model for baseline
    baseline_acc, cv_scores = rfe_analysis.train_initial_model(n_estimators=200)
    
    # Calculate permutation importance
    importance_df = rfe_analysis.calculate_permutation_importance(n_repeats=30)
    
    # Determine number of features to select
    n_features = args.n_features
    if n_features is None:
        n_features = max(10, len(rfe_analysis.feature_names) // 2)
    
    # Perform RFE
    selected_features, rfe_cv_scores = rfe_analysis.perform_rfe(
        n_features_to_select=n_features, 
        step=args.step
    )
    
    # Evaluate final model
    final_model, final_accuracy, cm = rfe_analysis.evaluate_final_model()
    
    # Create visualizations (unless disabled)
    if not args.no_plots:
        try:
            rfe_analysis.plot_feature_importance(top_n=min(30, len(selected_features)))
            rfe_analysis.plot_confusion_matrix(cm)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    # Save results
    results_df = rfe_analysis.save_results()
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Analysis mode: {args.mode}")
    print(f"Excluded varieties: {args.exclude_varieties}")
    print(f"Excluded items: {len(args.exclude_items)} items")
    if args.exclude_items:
        print(f"  Excluded item examples: {args.exclude_items[:5]}{'...' if len(args.exclude_items) > 5 else ''}")
    print(f"Total features (grammar items): {len(rfe_analysis.feature_names)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Baseline model accuracy: {baseline_acc:.4f}")
    print(f"Final model accuracy: {final_accuracy:.4f}")
    print(f"Baseline CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"RFE CV accuracy: {rfe_cv_scores.mean():.4f} (+/- {rfe_cv_scores.std() * 2:.4f})")
    
    print(f"\nTop 10 selected features by permutation importance:")
    if 'permutation_importance' in results_df.columns:
        top_features = results_df.head(10)[['selected_feature', 'permutation_importance']]
        for idx, row in top_features.iterrows():
            print(f"  {row['selected_feature']}: {row['permutation_importance']:.4f}")
    else:
        for feature in results_df.head(10)['selected_feature']:
            print(f"  {feature}")
    
    print(f"\nResults saved to: {results_df.iloc[0]['mode_filter']}_excl{('_'.join(args.exclude_varieties) if args.exclude_varieties else 'none')}.csv")
    if not args.no_plots:
        print(f"Plots saved: feature_importance_permutation.png, confusion_matrix.png")


def run_interactive():
    """Run interactive version with user prompts."""
    print("Interactive Recursive Feature Elimination Analysis")
    print("=" * 50)
    
    # Get user choices
    print("\nChoose analysis mode:")
    print("1. All features (spoken + written)")
    print("2. Spoken features only")
    print("3. Written features only")
    
    while True:
        try:
            mode_choice = int(input("Enter choice (1-3): "))
            if mode_choice in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
    
    mode_map = {1: None, 2: 'spoken', 3: 'written'}
    mode_filter = mode_map[mode_choice]
    mode_name = {1: 'all', 2: 'spoken', 3: 'written'}[mode_choice]
    
    # Get varieties to exclude
    print(f"\nDefault excluded varieties: ['Other']")
    exclude_input = input("Enter additional varieties to exclude (comma-separated, or press Enter for default): ").strip()
    
    if exclude_input:
        additional_varieties = [v.strip() for v in exclude_input.split(',')]
        exclude_varieties = ['Other'] + additional_varieties
    else:
        exclude_varieties = ['Other']
    
    print(f"Excluded varieties: {exclude_varieties}")
    
    # Get items to exclude
    print(f"\nWould you like to exclude specific grammatical items?")
    print("Enter grammatical item names to exclude (comma-separated, or press Enter to skip): ")
    exclude_items_input = input().strip()
    
    if exclude_items_input:
        exclude_items = [item.strip() for item in exclude_items_input.split(',')]
        print(f"Will exclude {len(exclude_items)} grammatical items")
    else:
        exclude_items = []
    
    # Run analysis
    rfe_analysis = GrammarFeatureSelection(
        random_state=42,
        mode_filter=mode_filter,
        exclude_varieties=exclude_varieties,
        exclude_items=exclude_items
    )
    
    print(f"\nRunning analysis with mode: {mode_name}, excluded varieties: {exclude_varieties}, excluded items: {len(exclude_items)}")
    
    # Load and process data
    data = rfe_analysis.load_data()
    X, y = rfe_analysis.prepare_features_target()
    
    # Run full analysis
    baseline_acc, cv_scores = rfe_analysis.train_initial_model(n_estimators=200)
    importance_df = rfe_analysis.calculate_permutation_importance()
    
    n_features = len(rfe_analysis.feature_names) // 2
    selected_features, rfe_cv_scores = rfe_analysis.perform_rfe(n_features_to_select=n_features)
    
    final_model, final_accuracy, cm = rfe_analysis.evaluate_final_model()
    
    # Generate plots
    try:
        rfe_analysis.plot_feature_importance()
        rfe_analysis.plot_confusion_matrix(cm)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Save results
    results_df = rfe_analysis.save_results()
    
    print(f"\nAnalysis complete! Results saved.")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"Selected {len(selected_features)} features out of {len(rfe_analysis.feature_names)}")


if __name__ == "__main__":
    import sys
    
    # Check if running interactively
    if len(sys.argv) == 1:
        # No command line arguments, run interactive mode
        run_interactive()
    else:
        # Command line arguments provided
        main()
