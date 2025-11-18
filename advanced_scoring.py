"""
Advanced Credit Scoring Module with ML Ensemble and Alternative Data
==================================================================

This module demonstrates next-generation credit scoring capabilities including:
- Ensemble machine learning (Random Forest, Gradient Boosting, Neural Networks)
- Alternative data integration (digital footprint, social sentiment, geolocation)
- Advanced risk metrics (PD, EL, VaR, SHAP explanations)
- Monte Carlo simulations for stress testing
- Explainable AI for regulatory compliance

Designed for deployment using Pyodide in browser environments.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class AdvancedCreditScorer:
    """Advanced ML-powered credit scoring system with alternative data integration."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rf_model = None
        self.gb_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature definitions
        self.traditional_features = [
            'monthly_income', 'monthly_expenses', 'existing_emi', 'desired_emi',
            'years_in_business', 'household_size', 'past_delay_days'
        ]
        
        self.alternative_features = [
            'digital_footprint', 'social_sentiment', 'location_stability',
            'transaction_velocity', 'network_score'
        ]
        
        self.all_features = self.traditional_features + self.alternative_features
        
    def generate_synthetic_data(self, n_samples=2000):
        """Generate synthetic training data with realistic microfinance patterns."""
        np.random.seed(self.random_state)
        
        # Traditional financial features with realistic distributions
        data = {}
        
        # Income follows log-normal distribution (common in microfinance)
        data['monthly_income'] = np.random.lognormal(mean=10, sigma=0.5, size=n_samples).clip(5000, 150000)
        
        # Expenses are correlated with income but with variance
        income_ratio = np.random.beta(2, 3, n_samples) * 0.8  # 0-80% of income
        data['monthly_expenses'] = data['monthly_income'] * income_ratio
        
        # Existing EMI burden
        data['existing_emi'] = np.random.exponential(3000, n_samples).clip(0, 25000)
        
        # Desired EMI influenced by income capacity
        capacity = data['monthly_income'] - data['monthly_expenses'] - data['existing_emi']
        data['desired_emi'] = np.abs(capacity * np.random.uniform(0.3, 1.5, n_samples)).clip(1000, 30000)
        
        # Business experience (power law distribution)
        data['years_in_business'] = np.random.pareto(2, n_samples).clip(0, 25)
        
        # Household size (Poisson distribution)
        data['household_size'] = np.random.poisson(4.2, n_samples).clip(1, 12)
        
        # Past delays (exponential with many zeros)
        delay_mask = np.random.random(n_samples) < 0.7  # 70% have no delays
        data['past_delay_days'] = np.where(delay_mask, 0, 
                                          np.random.exponential(8, n_samples)).clip(0, 120)
        
        # Alternative data features
        data['digital_footprint'] = np.random.beta(2, 2, n_samples) * 9 + 1  # 1-10 scale
        data['social_sentiment'] = np.random.normal(0.1, 0.4, n_samples).clip(-1, 1)
        data['location_stability'] = np.random.exponential(30, n_samples).clip(3, 180)
        data['transaction_velocity'] = np.random.poisson(45, n_samples).clip(0, 300)
        data['network_score'] = np.random.normal(6.5, 1.8, n_samples).clip(1, 10)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate realistic default labels using complex risk logic
        risk_features = self._calculate_risk_features(df)
        default_probability = self._calculate_default_probability(df, risk_features)
        
        # Add noise and create binary target
        noisy_prob = default_probability + np.random.normal(0, 0.05, n_samples)
        y = (np.random.random(n_samples) < np.clip(noisy_prob, 0.02, 0.8)).astype(int)
        
        return df[self.all_features], y
    
    def _calculate_risk_features(self, df):
        """Calculate derived risk features for realistic default modeling."""
        # Debt service coverage ratio
        capacity = df['monthly_income'] - df['monthly_expenses'] - df['existing_emi']
        dscr = capacity / (df['desired_emi'] + 1e-6)
        
        # Debt-to-income ratio
        total_debt = df['existing_emi'] + df['desired_emi']
        dti = total_debt / df['monthly_income']
        
        # Alternative data composite score
        alt_score = (
            df['digital_footprint'] / 10 * 0.3 +
            (df['social_sentiment'] + 1) / 2 * 0.2 +
            np.minimum(df['location_stability'] / 60, 1) * 0.2 +
            np.minimum(df['transaction_velocity'] / 100, 1) * 0.15 +
            df['network_score'] / 10 * 0.15
        )
        
        return {
            'dscr': dscr,
            'dti': dti,
            'alt_score': alt_score,
            'capacity': capacity
        }
    
    def _calculate_default_probability(self, df, risk_features):
        """Calculate realistic default probability based on risk factors."""
        base_risk = 0.1  # 10% base default rate
        
        # DSCR impact (most important factor)
        dscr_risk = np.where(
            risk_features['dscr'] < 0.8, 0.4,
            np.where(risk_features['dscr'] < 1.0, 0.25,
                    np.where(risk_features['dscr'] < 1.2, 0.1, -0.1))
        )
        
        # DTI impact
        dti_risk = np.where(risk_features['dti'] > 0.6, 0.2, 
                           np.where(risk_features['dti'] > 0.4, 0.1, 0))
        
        # Experience factor
        experience_risk = np.where(df['years_in_business'] < 1, 0.15,
                                  np.where(df['years_in_business'] < 2, 0.05, -0.02))
        
        # Past behavior (strong predictor)
        behavior_risk = np.where(df['past_delay_days'] == 0, -0.05,
                                np.where(df['past_delay_days'] < 15, 0.1,
                                        np.where(df['past_delay_days'] < 30, 0.25, 0.4)))
        
        # Household stress
        household_risk = np.where(df['household_size'] > 6, 0.08, 
                                 np.where(df['household_size'] < 3, -0.02, 0))
        
        # Alternative data impact (moderating factor)
        alt_data_risk = (1 - risk_features['alt_score']) * 0.15
        
        total_risk = (base_risk + dscr_risk + dti_risk + experience_risk + 
                     behavior_risk + household_risk + alt_data_risk)
        
        return np.clip(total_risk, 0.01, 0.85)
    
    def train_ensemble(self, n_samples=2000):
        """Train ensemble of ML models on synthetic data."""
        print(f"Generating {n_samples} synthetic training samples...")
        X, y = self.generate_synthetic_data(n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        print(f"Training ensemble models on {len(X_train)} samples...")
        
        # Train Random Forest (represents CatBoost)
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=10,
            random_state=self.random_state, class_weight='balanced'
        )
        self.rf_model.fit(X_train, y_train)
        
        # Train Gradient Boosting (represents XGBoost)
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.1,
            random_state=self.random_state
        )
        self.gb_model.fit(X_train, y_train)
        
        # Train Neural Network
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(30, 20, 10), max_iter=500, learning_rate_init=0.01,
            random_state=self.random_state, early_stopping=True, validation_fraction=0.1
        )
        self.nn_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_auc = roc_auc_score(y_test, self.rf_model.predict_proba(X_test)[:, 1])
        gb_auc = roc_auc_score(y_test, self.gb_model.predict_proba(X_test)[:, 1])
        nn_auc = roc_auc_score(y_test, self.nn_model.predict_proba(X_test_scaled)[:, 1])
        
        print(f"Model Performance (AUC):")
        print(f"  Random Forest: {rf_auc:.3f}")
        print(f"  Gradient Boosting: {gb_auc:.3f}")
        print(f"  Neural Network: {nn_auc:.3f}")
        
        self.is_trained = True
        return {
            'rf_auc': rf_auc,
            'gb_auc': gb_auc,
            'nn_auc': nn_auc
        }
    
    def predict_ensemble(self, features):
        """Make ensemble prediction with individual model contributions."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_ensemble() first.")
        
        # Ensure features is a 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Individual model predictions
        rf_prob = self.rf_model.predict_proba(features)[0, 1]
        gb_prob = self.gb_model.predict_proba(features)[0, 1]
        
        features_scaled = self.scaler.transform(features)
        nn_prob = self.nn_model.predict_proba(features_scaled)[0, 1]
        
        # Weighted ensemble (can be optimized based on model performance)
        ensemble_prob = rf_prob * 0.4 + gb_prob * 0.4 + nn_prob * 0.2
        
        return {
            'ensemble_prob': ensemble_prob,
            'rf_prob': rf_prob,
            'gb_prob': gb_prob,
            'nn_prob': nn_prob
        }
    
    def calculate_shap_approximation(self, features):
        """Approximate SHAP values using feature importance from Random Forest."""
        if not self.is_trained:
            return []
        
        importances = self.rf_model.feature_importances_
        
        # Normalize features relative to training distribution
        feature_means = np.array([20000, 12000, 3000, 7000, 3, 4, 2, 5, 0, 30, 50, 7])
        feature_stds = np.array([8000, 5000, 2000, 3000, 2, 2, 5, 2, 0.3, 20, 30, 1.5])
        
        normalized_features = (features - feature_means) / (feature_stds + 1e-6)
        contributions = normalized_features * importances * 0.1
        
        shap_values = list(zip(self.all_features, contributions))
        return sorted(shap_values, key=lambda x: abs(x[1]), reverse=True)
    
    def monte_carlo_simulation(self, base_features, n_simulations=1000, noise_level=0.05):
        """Perform Monte Carlo simulation for risk estimation."""
        if not self.is_trained:
            return np.array([0.5])
        
        results = []
        
        for _ in range(n_simulations):
            # Add realistic noise to features
            noise = np.random.normal(0, noise_level, len(base_features))
            noisy_features = base_features + noise
            
            # Ensure features remain in realistic ranges
            noisy_features = np.maximum(noisy_features, 0)  # No negative values
            
            prediction = self.predict_ensemble(noisy_features)
            results.append(prediction['ensemble_prob'])
        
        return np.array(results)
    
    def calculate_risk_metrics(self, features, desired_emi):
        """Calculate advanced risk metrics (PD, EL, VaR, etc.)."""
        prediction = self.predict_ensemble(features)
        
        # Probability of Default (1 year)
        pd_1y = prediction['ensemble_prob']
        
        # Loss Given Default (typical for unsecured microfinance)
        lgd = 0.55  # 55% loss given default
        
        # Exposure at Default
        ead = desired_emi * 12  # Annual exposure
        
        # Expected Loss
        expected_loss = pd_1y * lgd * ead
        
        # Value at Risk (Monte Carlo based)
        mc_results = self.monte_carlo_simulation(features, 1000)
        var_95 = np.percentile(mc_results, 95) * ead
        var_99 = np.percentile(mc_results, 99) * ead
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean(mc_results[mc_results >= np.percentile(mc_results, 95)]) * ead
        
        return {
            'pd_1y': pd_1y,
            'lgd': lgd,
            'ead': ead,
            'expected_loss': expected_loss,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'mc_mean': np.mean(mc_results),
            'mc_std': np.std(mc_results)
        }
    
    def score_borrower_comprehensive(self, borrower_data):
        """Comprehensive borrower scoring with all advanced features."""
        
        # Parse and validate input data
        try:
            features_dict = {
                'monthly_income': max(float(borrower_data.get('monthly_income', 20000) or 20000), 1000),
                'monthly_expenses': max(float(borrower_data.get('monthly_expenses', 12000) or 12000), 0),
                'existing_emi': max(float(borrower_data.get('existing_emi', 0) or 0), 0),
                'desired_emi': max(float(borrower_data.get('desired_emi', 5000) or 5000), 500),
                'years_in_business': max(float(borrower_data.get('years_in_business', 3) or 3), 0),
                'household_size': max(float(borrower_data.get('household_size', 4) or 4), 1),
                'past_delay_days': max(float(borrower_data.get('past_delay_days', 0) or 0), 0),
                'digital_footprint': float(borrower_data.get('digital_footprint', 5) or 5),
                'social_sentiment': float(borrower_data.get('social_sentiment', 0) or 0),
                'location_stability': float(borrower_data.get('location_stability', 24) or 24),
                'transaction_velocity': float(borrower_data.get('transaction_velocity', 50) or 50),
                'network_score': float(borrower_data.get('network_score', 7) or 7)
            }
        except (ValueError, TypeError):
            # Fallback defaults
            features_dict = {
                'monthly_income': 20000, 'monthly_expenses': 12000, 'existing_emi': 0,
                'desired_emi': 5000, 'years_in_business': 3, 'household_size': 4,
                'past_delay_days': 0, 'digital_footprint': 5, 'social_sentiment': 0,
                'location_stability': 24, 'transaction_velocity': 50, 'network_score': 7
            }
        
        # Convert to feature array
        features = np.array([features_dict[f] for f in self.all_features])
        
        # ML Predictions
        predictions = self.predict_ensemble(features)
        
        # Traditional risk metrics
        capacity = features_dict['monthly_income'] - features_dict['monthly_expenses'] - features_dict['existing_emi']
        dscr = capacity / (features_dict['desired_emi'] + 1e-6)
        dti = (features_dict['existing_emi'] + features_dict['desired_emi']) / features_dict['monthly_income']
        
        # Advanced risk metrics
        risk_metrics = self.calculate_risk_metrics(features, features_dict['desired_emi'])
        
        # Alternative data score
        alt_data_score = np.mean([
            features_dict['digital_footprint'] / 10,
            (features_dict['social_sentiment'] + 1) / 2,
            min(features_dict['location_stability'] / 60, 1),
            min(features_dict['transaction_velocity'] / 100, 1),
            features_dict['network_score'] / 10
        ])
        
        # SHAP explanations
        shap_values = self.calculate_shap_approximation(features)
        
        # Credit score (0-100, higher is better)
        credit_score = max(0, min(100, 100 - (predictions['ensemble_prob'] * 100)))
        
        # Generate interpretable signals
        signals = self._generate_ml_signals(predictions, dscr, alt_data_score, features_dict)
        
        # Risk band classification
        band, summary = self._classify_risk_band(credit_score, predictions['ensemble_prob'], dscr)
        
        # Actionable recommendations
        recommendations = self._generate_recommendations(predictions['ensemble_prob'], dscr, alt_data_score, features_dict)
        
        return {
            'credit_score': round(credit_score, 1),
            'band': band,
            'summary': summary,
            'signals': signals,
            'recommendations': recommendations,
            'traditional_metrics': {
                'dscr': round(dscr, 2),
                'dti': round(dti, 3),
                'capacity': round(capacity, 0)
            },
            'ml_predictions': {
                'ensemble_prob': round(predictions['ensemble_prob'], 4),
                'rf_prob': round(predictions['rf_prob'], 4),
                'gb_prob': round(predictions['gb_prob'], 4),
                'nn_prob': round(predictions['nn_prob'], 4)
            },
            'risk_metrics': {
                'pd_1y_pct': round(risk_metrics['pd_1y'] * 100, 2),
                'expected_loss': round(risk_metrics['expected_loss'], 0),
                'var_95': round(risk_metrics['var_95'], 0),
                'var_99': round(risk_metrics['var_99'], 0)
            },
            'alternative_data': {
                'composite_score': round(alt_data_score, 3),
                'digital_footprint': features_dict['digital_footprint'],
                'social_sentiment': features_dict['social_sentiment'],
                'location_stability': features_dict['location_stability'],
                'network_score': features_dict['network_score']
            },
            'explainability': {
                'top_shap_features': shap_values[:5],
                'shap_summary': f"{shap_values[0][0]}: {shap_values[0][1]:.3f}" if shap_values else "N/A"
            }
        }
    
    def _generate_ml_signals(self, predictions, dscr, alt_data_score, features_dict):
        """Generate human-readable ML signals."""
        signals = []
        
        # Ensemble risk assessment
        if predictions['ensemble_prob'] < 0.15:
            signals.append("🟢 ML Ensemble: Very low default risk (PD < 15%)")
        elif predictions['ensemble_prob'] < 0.30:
            signals.append("🟡 ML Ensemble: Moderate default risk (PD 15-30%)")
        else:
            signals.append("🔴 ML Ensemble: High default risk (PD > 30%)")
        
        # Model agreement analysis
        model_variance = np.std([predictions['rf_prob'], predictions['gb_prob'], predictions['nn_prob']])
        if model_variance < 0.05:
            signals.append("🎯 High model consensus - confident prediction")
        elif model_variance > 0.15:
            signals.append("⚠️ Model disagreement detected - review case manually")
        
        # Traditional metrics
        if dscr >= 1.5:
            signals.append("💰 Strong debt service coverage ratio")
        elif dscr < 1.0:
            signals.append("⚠️ Insufficient debt service coverage")
        
        # Alternative data insights
        if alt_data_score > 0.7:
            signals.append("📱 Strong digital behavior & network signals")
        elif alt_data_score < 0.4:
            signals.append("⚠️ Weak alternative data signals detected")
        
        # Specific risk factors
        if features_dict['past_delay_days'] > 15:
            signals.append("🚨 Significant payment delays in credit history")
        elif features_dict['past_delay_days'] == 0:
            signals.append("✅ Clean payment history - no reported delays")
        
        return signals
    
    def _classify_risk_band(self, credit_score, ensemble_prob, dscr):
        """Classify borrower into risk bands with explanations."""
        if credit_score >= 75:
            band = "🟢 PRIME · ML-Approved"
            summary = f"Strong borrower profile with {ensemble_prob:.1%} default probability. ML models show consistent low-risk signals across traditional and alternative data."
        elif credit_score >= 60:
            band = "🟡 NEAR-PRIME · Conditional"
            summary = f"Moderate risk profile with {ensemble_prob:.1%} default probability. Consider risk-based pricing or additional safeguards."
        elif credit_score >= 40:
            band = "🟠 SUB-PRIME · Enhanced Review"
            summary = f"Elevated risk profile with {ensemble_prob:.1%} default probability. Requires enhanced due diligence and risk mitigation."
        else:
            band = "🔴 HIGH-RISK · Decline/Restructure"
            summary = f"High risk profile with {ensemble_prob:.1%} default probability. Consider decline or significant loan restructuring."
        
        return band, summary
    
    def _generate_recommendations(self, ensemble_prob, dscr, alt_data_score, features_dict):
        """Generate actionable recommendations based on risk assessment."""
        recommendations = []
        
        if ensemble_prob > 0.3:
            recommendations.append("Consider requiring co-guarantor or additional collateral")
        
        if dscr < 1.2:
            recommendations.append("Reduce loan amount or extend tenure to improve DSCR")
        
        if alt_data_score < 0.5:
            recommendations.append("Gather additional verification through references or utility bills")
        
        if features_dict['digital_footprint'] < 4:
            recommendations.append("Low digital presence - consider manual verification processes")
        
        if features_dict['years_in_business'] < 2:
            recommendations.append("Request business stability proof (customer references, supplier bills)")
        
        if features_dict['household_size'] > 6:
            recommendations.append("Large household - verify income sources and stability")
        
        if ensemble_prob < 0.15 and dscr > 1.5:
            recommendations.append("Excellent candidate - consider preferential pricing or higher limits")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Initialize and train the scorer
    scorer = AdvancedCreditScorer()
    performance = scorer.train_ensemble(n_samples=1500)
    
    # Example borrower data
    sample_borrower = {
        'monthly_income': 25000,
        'monthly_expenses': 15000,
        'existing_emi': 3000,
        'desired_emi': 6000,
        'years_in_business': 4,
        'household_size': 4,
        'past_delay_days': 0,
        'digital_footprint': 7,
        'social_sentiment': 0.2,
        'location_stability': 36,
        'transaction_velocity': 65,
        'network_score': 8
    }
    
    # Comprehensive scoring
    result = scorer.score_borrower_comprehensive(sample_borrower)
    
    print("\n" + "="*50)
    print("ADVANCED CREDIT SCORING RESULT")
    print("="*50)
    print(f"Credit Score: {result['credit_score']}/100")
    print(f"Risk Band: {result['band']}")
    print(f"Default Probability: {result['risk_metrics']['pd_1y_pct']}%")
    print(f"Expected Loss: ₹{result['risk_metrics']['expected_loss']:,.0f}")
    print(f"DSCR: {result['traditional_metrics']['dscr']}")
    print(f"\nAlternative Data Score: {result['alternative_data']['composite_score']:.1%}")
    print("\nML Model Predictions:")
    for model, prob in result['ml_predictions'].items():
        if 'prob' in model:
            print(f"  {model}: {prob:.1%}")
    
    print(f"\nTop Risk Signals:")
    for signal in result['signals'][:3]:
        print(f"  • {signal}")
    
    print(f"\nRecommendations:")
    for rec in result['recommendations'][:3]:
        print(f"  → {rec}")