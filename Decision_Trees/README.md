# Decision Trees

## What is Decision Tree?
A Decision Tree is a supervised machine learning algorithm used for classification and regression. It learns a series of if-else rules by splitting the dataset based on feature values.
- A Decision Tree is simply a series of questions. 
- It keeps asking questions until it reaches a final decision.

## Why is it Called a Decision Tree?
Imagine making decisions like this:
Should I carry an umbrella?

          Rain?

         /     \
      Yes       No
       /
    Temperature?
      /       \
    cold     warm

- Each question creates branches.
- Eventually you reach one decision.
- The structure resembles a tree.

## Components of a Decision Tree
- **Node** --> A node is any point in the tree where information is stored. Example:  Age >30
- **Root Node** --> The first node from which the tree begins.
- **Leaf Node** --> A leaf node represents the final outcome, decision, or prediction of the machine learning model.
- **Internal Node** --> Any node that is not the root or a leaf and still performs a split.
- **Branch/Edge** --> Connection between nodes.
- **Splitting Criterion** --> The mathematical rule used to decide the best split. Example: Gini Index, Entropy
- **Pruning** --> Pruning removes redundant decision tree branches to stop overfitting and improve generalization.
- **Impurity** --> A measure of how mixed the classes are in a node.

## How does it work?
1. Start with all data.
2. Find the best feature to split.
3. Divide data.
4. Repeat recursively.
5. Stop when stopping criteria are met.

## Splitting Criteria
`For Classification:`
### Gini Impurity
---
Gini Impurity measures the frequency with which a randomly chosen element from a set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset.

$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$
Where:
* **$p_i$**: The probability of an element belonging to class $i$.
* **$C$**: The total number of classes.
### 🧮 Gini Impurity Step-by-Step Example

Let's calculate the Gini Impurity for a dataset node containing **10 fruits** representing two classes:
* 🍏 **6 Apples**
* 🍌 **4 Bananas**

#### 1. Calculate Probabilities ($p_i$)
* Probability of Apple ($p_1$) = $\frac{6}{10} = 0.6$
* Probability of Banana ($p_2$) = $\frac{4}{10} = 0.4$

#### 2. Square the Probabilities ($p_i^2$)
* $p_1^2 = 0.6^2 = 0.36$
* $p_2^2 = 0.4^2 = 0.16$

#### 3. Sum the Squared Probabilities ($\sum p_i^2$)
$$\sum p_i^2 = 0.36 + 0.16 = 0.52$$

#### 4. Subtract from 1 ($1 - \sum p_i^2$)
$$Gini = 1 - 0.52 = \mathbf{0.48}$$

The **Gini Impurity** for this node is **0.48**.

---

### 💡 Quick Rule of Thumb
* **Gini = 0.0 (Pure Node):** All items belong to a single class (e.g., 10 Apples, 0 Bananas).
* **Gini = 0.5 (Max Impurity):** Items are split perfectly 50/50 between two classes.

```Python
def calculate_gini(labels: list) -> float:
    """
    Calculates the Gini Impurity for a list of target labels.
    
    Example:
        >>> calculate_gini(['apple']*6 + ['banana']*4)
        0.48
    """
    if not labels:
        return 0.0

    total_samples = len(labels)
    
    # 1. Count frequencies of each class
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        
    # 2. Calculate probabilities, square them, and sum them up
    squared_probabilities_sum = 0.0
    for count in counts.values():
        probability = count / total_samples
        squared_probabilities_sum += probability ** 2
        
    # 3. Subtract the sum from 1
    return round(1.0 - squared_probabilities_sum, 4)
    
# 6 Apples, 4 Bananas
node_samples = ['apple'] * 6 + ['banana'] * 4

gini_impurity = calculate_gini(node_samples)
print(f"Gini Impurity: {gini_impurity}")  # Output: 0.48
```

