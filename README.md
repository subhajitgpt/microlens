# MicroLens Advanced Credit Scoring Platform 🚀

A next-generation credit risk analytics platform featuring ensemble machine learning, alternative data integration, and explainable AI - all running locally in your browser via Pyodide.

## 🌟 Key Features

### Advanced Machine Learning
- **Ensemble Models**: Random Forest, Gradient Boosting, Neural Networks
- **Alternative Data**: Digital footprint, social sentiment, location stability, transaction velocity, network analysis
- **Risk Metrics**: Probability of Default (PD), Expected Loss (EL), Value at Risk (VaR), Expected Shortfall
- **Explainable AI**: SHAP value approximations for regulatory compliance
- **Monte Carlo Simulations**: Stress testing and scenario analysis

### Enterprise-Grade Privacy
- **Zero Data Transmission**: All ML processing happens locally in your browser
- **Pyodide-Powered**: Full Python scientific stack running client-side
- **No Backend Required**: Pure frontend deployment with enterprise-grade capabilities

### Advanced Analytics Dashboard
- **Real-time Risk Assessment**: Instant ML ensemble predictions
- **Model Performance Tracking**: Individual model contributions and consensus analysis
- **Alternative Data Integration**: Social, behavioral, and digital footprint scoring
- **Risk Band Classification**: Prime, Near-Prime, Sub-Prime, High-Risk categories
- **Actionable Recommendations**: ML-driven suggestions for risk mitigation

## 🛠 Technology Stack

- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
- **ML Runtime**: Pyodide (Python in the browser)
- **ML Libraries**: scikit-learn, numpy, pandas
- **Models**: Random Forest, Gradient Boosting, Multi-layer Perceptron
- **Deployment**: Static hosting (no server required)

## 📊 Advanced Risk Metrics

### Traditional Metrics
- **DSCR** (Debt Service Coverage Ratio): Cash flow adequacy
- **DTI** (Debt-to-Income): Total debt burden assessment
- **Capacity Analysis**: Net available cash flow

### Next-Gen Metrics  
- **PD (1Y)**: Probability of Default within 1 year
- **EL**: Expected Loss calculation
- **VaR 95%/99%**: Value at Risk at different confidence levels
- **ES**: Expected Shortfall (Conditional VaR)
- **Monte Carlo VaR**: Simulation-based risk assessment

### Alternative Data Features
- **Digital Footprint Score**: Online presence and engagement (1-10 scale)
- **Social Media Sentiment**: Sentiment analysis of social activity (-1 to 1)
- **Location Stability**: Address tenure in months
- **Transaction Velocity**: Monthly digital transaction frequency
- **Network Score**: Peer risk assessment (1-10 scale)

## 🎯 Model Ensemble

The platform uses a sophisticated ensemble approach:

1. **Random Forest (40% weight)**: Captures non-linear patterns and feature interactions
2. **Gradient Boosting (40% weight)**: Sequential learning for complex risk patterns  
3. **Neural Network (20% weight)**: Deep learning for alternative data patterns

### Model Performance
- **Cross-validation AUC**: >0.85 on synthetic microfinance data
- **Feature Importance**: SHAP-based explainability
- **Ensemble Consensus**: Model agreement analysis for prediction confidence

## 🚀 Getting Started

### Option 1: Direct Browser Access
1. Open `index.html` in any modern browser
2. The ML models will automatically load via Pyodide
3. Enter borrower information and run analysis
4. View comprehensive risk assessment with explanations

### Option 2: Development Setup
```bash
# Clone or download the repository
cd microlens

# Optional: Run local server for development
python -m http.server 8000
# or
npx serve .

# Open http://localhost:8000 in browser
```

### Option 3: Static Hosting Deployment
Deploy the entire folder to any static hosting service:
- GitHub Pages, Netlify, Vercel
- AWS S3, Azure Static Web Apps
- Any CDN or web server

## 🔒 Privacy & Security

### Client-Side Processing
- **Zero Server Calls**: All ML processing in browser
- **No Data Storage**: No borrower data persisted
- **Local Execution**: Pyodide ensures data privacy
- **Offline Capable**: Works without internet after initial load

### Enterprise Compliance
- **GDPR Ready**: No data collection or transmission
- **Audit Trail**: Full model explainability
- **Regulatory Friendly**: Transparent decision making
- **Data Sovereignty**: Customer data never leaves their environment

## ⚠️ Disclaimer

This is a demonstration platform using synthetic data and simplified models. 
For production lending decisions:

- Use real historical data for model training
- Implement proper model validation and testing
- Ensure regulatory compliance and fair lending practices
- Add comprehensive risk management frameworks
- Include human oversight and appeal processes

---

**Built with ❤️ for the future of credit scoring**

*Combining traditional risk assessment with cutting-edge AI and alternative data analytics*
