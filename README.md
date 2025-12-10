# Train-Test Split Experimentation Project

## Project Overview

This project provides a comprehensive exploration of train-test splitting strategies in machine learning, with a focus on understanding different approaches for standard data, imbalanced datasets, and time-series data. Through hands-on experimentation, we demonstrate the critical importance of choosing the right splitting strategy for reliable model evaluation.

---

## 1. Windsurf Learning Log (5 pts)

### How did you use Windsurf for each project?

**Project Setup and Structure:**

- Used Windsurf to generate the initial notebook structure with clear sections for different splitting strategies
- Leveraged Windsurf's code generation to create synthetic datasets that mimic real-world scenarios (imbalanced classes, time-series data)
- Utilized Windsurf to implement multiple visualization approaches for comparing different splitting methods

**Code Development:**

- Prompted Windsurf to generate implementations for:
  - Basic train-test splits with different ratios (60/40, 70/30, 80/20, 90/10)
  - Stratified splitting for imbalanced classification
  - Temporal splitting for time-series data
  - Walk-forward validation with expanding windows
- Used Windsurf to create comprehensive visualizations showing data leakage problems and proper splitting approaches

**Documentation and Explanations:**

- Relied heavily on Windsurf to generate detailed markdown explanations for each concept
- Asked Windsurf to create the comprehensive final summary with practical takeaways
- Used Windsurf to add visual separators and formatting that enhances readability

### What prompts or approaches were most effective?

**Most Effective Prompts:**

1. **Specific, Context-Rich Requests:**

   - ✅ "Create a visualization comparing random vs temporal splitting for time-series data, showing date ranges and highlighting the data leakage problem"
   - vs ❌ "Make a plot about time series"

2. **Iterative Refinement:**

   - Started with: "Implement basic train-test split experimentation"
   - Refined to: "Add stratified splitting comparison with visualization of class distributions"
   - Further refined: "Create a decision tree flowchart in the summary to help users choose the right splitting strategy"

3. **Educational Focus:**

   - "Explain why random splitting fails for time-series data with concrete examples and visual demonstrations"
   - "Create a comprehensive summary that includes practical takeaways, common mistakes, and best practices"

4. **Structure-First Approach:**
   - Asked Windsurf to outline the entire notebook structure before generating code
   - This ensured logical flow: Basic concepts → Advanced techniques → Practical applications

**Approaches That Worked Well:**

- Breaking down complex topics into smaller, manageable sections
- Requesting both code AND explanatory markdown together
- Asking for multiple visualization styles to compare effectiveness
- Using Windsurf to generate realistic synthetic data that demonstrates specific problems

### What did Windsurf struggle with, and how did you address it?

**Struggles and Solutions:**

1. **Challenge: Initial Visualizations Were Too Simple**

   - Problem: First plots didn't clearly show the data leakage issue
   - Solution: Provided specific requirements: "Show overlapping date ranges in different colors with annotations explaining the problem"
   - Result: Much more informative visualizations that clearly communicate the concept

2. **Challenge: Inconsistent Output Formatting**

   - Problem: Print statements had varying levels of detail and formatting
   - Solution: Established a consistent format template and asked Windsurf to follow it throughout
   - Example: Used emojis (✓, ⚠️, ❌) and box characters (═, ║) for visual structure

3. **Challenge: Summary Was Initially Too Technical**

   - Problem: First version focused too much on code and not enough on practical insights
   - Solution: Requested: "Rewrite the summary for someone who understands ML basics but needs practical guidance on which approach to choose"
   - Result: More accessible, decision-focused summary

4. **Challenge: Missing Edge Cases**
   - Problem: Initial time-series implementation didn't handle unequal fold sizes
   - Solution: Explicitly asked: "What happens if the data doesn't divide evenly into 5 folds? Handle this case."
   - Result: More robust implementation with proper handling of remainder data

### How is using Windsurf changing your learning process?

**Positive Changes:**

1. **Faster Experimentation:**

   - Can quickly test multiple approaches (different split ratios, visualization styles) without getting bogged down in syntax
   - Allows more time for understanding concepts rather than debugging code

2. **Better Documentation Habits:**

   - Windsurf generates comprehensive explanations alongside code
   - This encourages thinking about "why" not just "how"
   - Makes notebooks more shareable and useful for future reference

3. **Deeper Understanding Through Visualization:**

   - Easy to request multiple visualization approaches
   - Seeing concepts visualized in different ways deepens understanding
   - Can focus on interpretation rather than matplotlib syntax

4. **Learning Best Practices:**
   - Windsurf often suggests industry-standard approaches
   - Exposed to better coding patterns and organization
   - Learn proper structure for educational notebooks

**Areas for Mindful Use:**

- Need to verify that generated code actually demonstrates the intended concept
- Important to understand the underlying logic, not just run generated code
- Should modify and experiment with Windsurf's output to make it truly your own

---

## 2. Key Takeaways (5 pts)

### What was the most surprising thing you learned?

**The Data Leakage Problem in Time-Series is More Subtle Than Expected**

The most eye-opening discovery was how easy it is to accidentally introduce data leakage in time-series problems, and how dramatically it affects results. I knew theoretically that you shouldn't use random splits for temporal data, but seeing it in action was shocking:

- Random split showed **2.15% accuracy** on test data
- Temporal split showed **20.50% accuracy** on the same data
- The random split gave a **false sense of good performance** by training on future data to predict the past

**Key Insight:** The random split wasn't just slightly wrong—it fundamentally violated the causality assumption. In production, a model trained with random splits would completely fail because it can't access future information to make predictions about the present.

**Another Surprise:** Walk-forward validation revealed that model performance can vary significantly across different time periods (22.89% to 28.92% in our example). This variability is hidden when you just do a single train-test split, leading to overconfident performance estimates.

### What preprocessing technique do you think you'll use most?

**Stratified Splitting for Classification Tasks**

While all techniques have their place, I believe **stratified splitting** will be the most frequently used in real-world scenarios because:

1. **Default Choice for Classification:**

   - Most classification datasets have some degree of class imbalance
   - Stratification is a low-cost insurance policy against evaluation problems
   - Very easy to implement: just add `stratify=y` parameter

2. **Prevents Subtle Issues:**

   - Even if classes aren't heavily imbalanced (e.g., 60/40 split), stratification ensures minority classes appear in both train and test sets
   - Reduces variance in performance estimates across different random seeds
   - Makes results more reproducible and reliable

3. **No Downside:**
   - Doesn't add computational cost
   - Doesn't require additional hyperparameters
   - Works well even when not strictly necessary

**When I'll Use Other Techniques:**

- **Temporal splitting:** Whenever working with time-series, sequential data, or any data with temporal ordering
- **Walk-forward validation:** For critical time-series applications where robust evaluation is essential
- **Regular splits:** Only for regression or perfectly balanced classification (rare)

### What would you do differently in a real-world project?

**Key Differences for Production Scenarios:**

1. **Multiple Validation Strategies:**

   - Wouldn't rely on just one split
   - Would use cross-validation (k-fold for standard data, time-series CV for temporal data)
   - Would create a separate hold-out test set that's never touched until final evaluation

2. **More Rigorous Time-Series Handling:**

   - Would implement multiple time-based splits (not just one temporal cutoff)
   - Would test on multiple future periods to assess temporal stability
   - Would explicitly check for concept drift
   - Would consider sliding window vs expanding window based on problem characteristics

3. **Stratification with Additional Considerations:**

   - Would stratify not just on target variable but also on important subgroups
   - Example: If building a medical model, stratify by both diagnosis AND demographic factors
   - Would ensure rare but important classes have sufficient representation

4. **Data Leakage Auditing:**

   - Would conduct thorough feature-target leakage checks
   - Would verify that no information from test set leaks into training
   - Would create automated tests to prevent accidental leakage

5. **Documentation:**

   - Would document the rationale for choosing a specific splitting strategy
   - Would record the date cutoff for temporal splits (critical for reproducibility)
   - Would track performance across multiple random seeds to assess stability

6. **Business Context Integration:**
   - Would align split strategy with deployment scenario
   - Example: If model will be retrained monthly, would simulate that in validation
   - Would consider operational constraints (data availability, retraining frequency)

### One question or topic you want to explore further

**Question: How do you handle the time-series splitting when you have multiple related time-series?**

**The Scenario:**
Imagine you're building a model to predict customer churn, and you have multiple customers, each with their own time-series of activity (purchases, logins, etc.). How do you split this data?

**Specific Sub-Questions:**

1. **Split by Time or by Customer?**

   - Do you put all customers in training set up to a certain date, then all customers in test set after that date?
   - Or do you split customers into train/test groups, and use all their data in respective sets?
   - What are the trade-offs?

2. **Nested Time-Series:**

   - What if you have hierarchical time-series (e.g., sales by product by store by day)?
   - How do you ensure proper splitting at multiple levels?

3. **Irregular Time-Series:**

   - What if different customers have different date ranges?
   - Some customers might have data from 2020-2023, others only 2022-2023
   - How do you create fair train-test splits?

4. **Panel Data Considerations:**
   - Should you stratify by customer characteristics while also respecting temporal order?
   - How do you balance individual-level and time-level variation?

**Why This Matters:**
Most real-world time-series problems aren't simple single sequences. They involve multiple entities tracked over time (customers, sensors, stores, etc.). Understanding the right splitting strategy for these complex scenarios is crucial for building reliable models.

**Next Steps to Explore:**

- Research panel data splitting methods
- Investigate time-series cross-validation for grouped data
- Study how to handle imbalanced temporal data (some entities with much more data than others)
- Look into hierarchical time-series forecasting approaches

---

## Project Files

- `train_test_split_experimentation.ipynb` - Main Jupyter notebook with all experiments and analysis
- `README.md` - This file

## Key Concepts Covered

1. **Basic Train-Test Splitting**

   - Different split ratios (60/40, 70/30, 80/20, 90/10)
   - Bias-variance tradeoff analysis
   - Overfitting detection through train-test gap

2. **Stratified Splitting**

   - Maintaining class proportions
   - Handling imbalanced datasets
   - Comparison with random splitting

3. **Time-Series Splitting**
   - Temporal ordering importance
   - Data leakage demonstration
   - Walk-forward validation
   - Expanding window approach

## Requirements

```python
numpy
pandas
scikit-learn
matplotlib
seaborn
```

## How to Use This Notebook

1. Run cells sequentially to see progressive complexity
2. Each section builds on previous concepts
3. Visualizations demonstrate key differences between approaches
4. Final summary provides decision-making framework

## Key Visualizations

- Train vs Test accuracy across different split ratios
- Class distribution comparisons (random vs stratified)
- Temporal split visualization showing data leakage
- Walk-forward validation windows
- Performance stability across time periods

## Conclusion

This project demonstrates that the way you split your data is just as important as the model you choose. A sophisticated deep learning model with wrong data splitting will perform worse in production than a simple logistic regression with correct splitting. Always align your validation setup with your production scenario.

---

## Author Notes

This project was created as a comprehensive learning exercise in understanding train-test splitting strategies. The goal was not just to implement different approaches, but to deeply understand when and why each strategy is appropriate. Through extensive use of Windsurf AI assistance, I was able to focus more on conceptual understanding and less on syntax, while still producing production-quality code and documentation.

**Learning Philosophy:**

> "Understanding why is more important than knowing how. The how you can always look up; the why determines whether you solve the problem correctly."

---

_Last Updated: December 2024_
