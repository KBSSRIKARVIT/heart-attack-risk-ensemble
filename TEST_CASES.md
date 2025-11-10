# 🧪 Test Cases for Heart Attack Risk Prediction App

## Test Case 1: Low Risk Patient (Healthy Individual)
**Input:**
- Gender: Female (2)
- Age: 35 years
- Height: 165 cm
- Weight: 60 kg
- Systolic BP: 120 mmHg
- Diastolic BP: 80 mmHg
- Cholesterol: Normal (1)
- Glucose: Normal (1)
- Smoking: No (0)
- Alcohol: No (0)
- Physical Activity: Yes (1)
- Protein Level: 14.0
- Ejection Fraction: 60.0

**Expected Output:**
- Risk Level: ✅ Low Risk
- Risk Probability: < 10% (typically 2-8%)
- Prediction: No Heart Disease
- Key Risk Factors: ✅ Health Status: Healthy indicators
- Model Breakdown:
  - XGBoost: ~5-8% risk
  - CatBoost: ~1-2% risk (most accurate for low risk)
  - LightGBM: ~20-25% risk (Note: LightGBM tends to be more conservative/risk-averse)
  - Ensemble: ~2-5% risk (weighted: 5% XGB + 85% CAT + 10% LGB)
- Recommendation: ✅ Low Risk - Continue maintaining a healthy lifestyle!

**Note:** LightGBM may show higher individual risk percentages due to its training characteristics, but the ensemble weights (85% CatBoost) ensure the final prediction remains accurate.

---

## Test Case 2: Moderate Risk Patient (Some Risk Factors)
**Input:**
- Gender: Male (1)
- Age: 55 years
- Height: 175 cm
- Weight: 85 kg (BMI ~27.8 - Overweight)
- Systolic BP: 135 mmHg
- Diastolic BP: 88 mmHg
- Cholesterol: Above Normal (2)
- Glucose: Normal (1)
- Smoking: No (0)
- Alcohol: Yes (1)
- Physical Activity: No (0)
- Protein Level: 6.5
- Ejection Fraction: 55.0

**Expected Output:**
- Risk Level: ⚠️ Moderate Risk
- Risk Probability: 30-50% (typically 35-45%)
- Prediction: May indicate risk
- Key Risk Factors: ⚠️ High BP, High cholesterol, Alcohol consumption, Physical inactivity
- Model Breakdown:
  - XGBoost: ~35-45% risk
  - CatBoost: ~35-45% risk
  - LightGBM: ~35-45% risk
  - Ensemble: ~35-45% risk
- Recommendation: ⚠️ Moderate Risk - Consider consulting a healthcare professional.

---

## Test Case 3: High Risk Patient (Multiple Risk Factors)
**Input:**
- Gender: Male (1)
- Age: 65 years
- Height: 170 cm
- Weight: 95 kg (BMI ~32.9 - Obese)
- Systolic BP: 150 mmHg
- Diastolic BP: 100 mmHg
- Cholesterol: Well Above Normal (3)
- Glucose: Well Above Normal (3)
- Smoking: Yes (1)
- Alcohol: Yes (1)
- Physical Activity: No (0)
- Protein Level: 6.0
- Ejection Fraction: 45.0

**Expected Output:**
- Risk Level: 🚨 Very High Risk
- Risk Probability: > 70% (typically 75-90%)
- Prediction: Heart Disease Detected
- Key Risk Factors: ⚠️ High BMI (>30), High BP, High cholesterol, High glucose, Smoking, Alcohol consumption, Physical inactivity
- Model Breakdown:
  - XGBoost: ~75-90% risk
  - CatBoost: ~75-90% risk
  - LightGBM: ~75-90% risk
  - Ensemble: ~75-90% risk
- Recommendation: ⚠️ High Risk Detected! Please consult with a healthcare professional immediately.

---

## Test Case 4: Borderline Case (Age Factor)
**Input:**
- Gender: Female (2)
- Age: 50 years
- Height: 160 cm
- Weight: 70 kg (BMI ~27.3 - Overweight)
- Systolic BP: 130 mmHg
- Diastolic BP: 85 mmHg
- Cholesterol: Above Normal (2)
- Glucose: Normal (1)
- Smoking: No (0)
- Alcohol: No (0)
- Physical Activity: Yes (1)
- Protein Level: 7.0
- Ejection Fraction: 58.0

**Expected Output:**
- Risk Level: ⚠️ Moderate Risk
- Risk Probability: 20-40% (typically 25-35%)
- Prediction: May indicate risk
- Key Risk Factors: ⚠️ High BMI (>30), High BP, High cholesterol
- Model Breakdown:
  - XGBoost: ~25-35% risk
  - CatBoost: ~25-35% risk
  - LightGBM: ~25-35% risk
  - Ensemble: ~25-35% risk
- Recommendation: ⚠️ Moderate Risk - Consider consulting a healthcare professional.

---

## Test Case 5: Young Patient with Lifestyle Risks
**Input:**
- Gender: Male (1)
- Age: 28 years
- Height: 180 cm
- Weight: 75 kg (BMI ~23.1 - Normal)
- Systolic BP: 125 mmHg
- Diastolic BP: 82 mmHg
- Cholesterol: Normal (1)
- Glucose: Normal (1)
- Smoking: Yes (1)
- Alcohol: Yes (1)
- Physical Activity: No (0)
- Protein Level: 14.5
- Ejection Fraction: 62.0

**Expected Output:**
- Risk Level: ⚠️ Moderate Risk
- Risk Probability: 15-30% (typically 20-28%)
- Prediction: May indicate risk
- Key Risk Factors: ⚠️ Smoking, Alcohol consumption, Physical inactivity
- Model Breakdown:
  - XGBoost: ~20-28% risk
  - CatBoost: ~20-28% risk
  - LightGBM: ~20-28% risk
  - Ensemble: ~20-28% risk
- Recommendation: ⚠️ Moderate Risk - Consider consulting a healthcare professional.

---

## Test Case 6: Elderly Patient with Good Health
**Input:**
- Gender: Female (2)
- Age: 70 years
- Height: 155 cm
- Weight: 58 kg (BMI ~24.1 - Normal)
- Systolic BP: 125 mmHg
- Diastolic BP: 78 mmHg
- Cholesterol: Normal (1)
- Glucose: Normal (1)
- Smoking: No (0)
- Alcohol: No (0)
- Physical Activity: Yes (1)
- Protein Level: 13.5
- Ejection Fraction: 58.0

**Expected Output:**
- Risk Level: ✅ Low to Moderate Risk
- Risk Probability: 10-25% (typically 15-22%)
- Prediction: No Heart Disease (or low risk)
- Key Risk Factors: ✅ Health Status: Healthy indicators (or minimal risk factors)
- Model Breakdown:
  - XGBoost: ~15-22% risk
  - CatBoost: ~15-22% risk
  - LightGBM: ~15-22% risk
  - Ensemble: ~15-22% risk
- Recommendation: ✅ Low Risk - Continue maintaining a healthy lifestyle! (or Moderate Risk warning)

---

## Test Case 7: Extreme High Risk (All Risk Factors)
**Input:**
- Gender: Male (1)
- Age: 60 years
- Height: 168 cm
- Weight: 100 kg (BMI ~35.4 - Obese)
- Systolic BP: 160 mmHg
- Diastolic BP: 105 mmHg
- Cholesterol: Well Above Normal (3)
- Glucose: Well Above Normal (3)
- Smoking: Yes (1)
- Alcohol: Yes (1)
- Physical Activity: No (0)
- Protein Level: 5.5
- Ejection Fraction: 40.0

**Expected Output:**
- Risk Level: 🚨 Very High Risk
- Risk Probability: > 85% (typically 88-95%)
- Prediction: Heart Disease Detected
- Key Risk Factors: ⚠️ High BMI (>30), High BP, High cholesterol, High glucose, Smoking, Alcohol consumption, Physical inactivity
- Model Breakdown:
  - XGBoost: ~88-95% risk
  - CatBoost: ~88-95% risk
  - LightGBM: ~88-95% risk
  - Ensemble: ~88-95% risk
- Recommendation: ⚠️ High Risk Detected! Please consult with a healthcare professional immediately.

---

## Test Case 8: Only Physical Inactivity
**Input:**
- Gender: Female (2)
- Age: 40 years
- Height: 165 cm
- Weight: 65 kg (BMI ~23.9 - Normal)
- Systolic BP: 118 mmHg
- Diastolic BP: 75 mmHg
- Cholesterol: Normal (1)
- Glucose: Normal (1)
- Smoking: No (0)
- Alcohol: No (0)
- Physical Activity: No (0)
- Protein Level: 14.0
- Ejection Fraction: 60.0

**Expected Output:**
- Risk Level: ✅ Low Risk
- Risk Probability: < 15% (typically 5-12%)
- Prediction: No Heart Disease
- Key Risk Factors: ℹ️ Lifestyle Note: Physical inactivity - Consider adding regular physical activity to reduce risk.
- Model Breakdown:
  - XGBoost: ~5-12% risk
  - CatBoost: ~5-12% risk
  - LightGBM: ~5-12% risk
  - Ensemble: ~5-12% risk
- Recommendation: ✅ Low Risk - Continue maintaining a healthy lifestyle!

---

## ✅ Verification Checklist

### UI Elements to Verify:
- [ ] Page title displays correctly: "Predicting Heart Attack Risk: An Ensemble Modeling Approach"
- [ ] Subtitle includes: "XGBoost, CatBoost, and LightGBM"
- [ ] Sidebar shows optimized ensemble weights (XGB: 5%, CAT: 85%, LGB: 10%)
- [ ] Sidebar displays Accuracy: 80.77% and Recall: 93.27%
- [ ] All input fields are present and functional
- [ ] Prediction button works correctly
- [ ] Results display with proper formatting

### Model Display to Verify:
- [ ] All 4 models displayed horizontally: XGBoost, CatBoost, LightGBM, Ensemble
- [ ] Each model shows progress bar with percentage inside
- [ ] Risk percentage displayed below each bar
- [ ] Color coding: Green (low), Orange (moderate), Red (high)
- [ ] Ensemble metrics section shows Accuracy and Recall

### Prediction Results to Verify:
- [ ] Risk probability displayed correctly
- [ ] Risk level matches probability range
- [ ] Key risk factors identified correctly
- [ ] Recommendations match risk level
- [ ] Model breakdown shows all 4 models
- [ ] Ensemble method info displayed

### Error Handling:
- [ ] App handles missing models gracefully
- [ ] Invalid inputs show appropriate warnings
- [ ] Error messages are user-friendly

---

## 📊 Expected Ensemble Metrics (Sidebar)
- **Accuracy**: 80.77%
- **Recall**: 93.27%
- **Ensemble Weights**: XGBoost: 5.0%, CatBoost: 85.0%, LightGBM: 10.0%

---

## 🎯 Quick Test Scenarios

1. **Minimum Input Test**: Use default values, click predict → Should show low risk
2. **Maximum Risk Test**: Set all risk factors to maximum → Should show very high risk
3. **Edge Case Test**: Age 20, all normal → Should show very low risk
4. **Edge Case Test**: Age 100, all normal → Should show moderate risk due to age
5. **Single Risk Factor**: Only smoking → Should show moderate risk
6. **Physical Inactivity Only**: Only inactive, all else normal → Should show info message (not warning)

---

## 📝 Notes
- Actual risk percentages may vary slightly (±2-3%) due to model variations
- The ensemble uses weighted average: 5% XGBoost + 85% CatBoost + 10% LightGBM
- **Important:** LightGBM may show higher individual risk percentages (15-25% for low-risk cases) due to its training characteristics. This is expected behavior and does not affect the final ensemble prediction, which is heavily weighted toward CatBoost (85%).
- The final ensemble prediction is the weighted average of all three models, so even if LightGBM shows higher values, the ensemble result remains accurate.
- For low-risk patients: CatBoost typically shows the most accurate low values (~1-2%), while LightGBM may show 20-25%. The ensemble (weighted) will be closer to CatBoost's prediction.

