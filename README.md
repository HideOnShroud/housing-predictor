# Housing Market Analysis

This project analyzes the California Housing Prices dataset using linear regression and Ridge regression models. The analysis includes data preprocessing, feature engineering, visualization, and model evaluation.

## Dataset

The dataset used for this analysis is the California Housing Prices dataset, which can be found [here](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data).

## Prerequisites

Make sure you have the following Python packages installed:
- pandas
- seaborn
- matplotlib
- scikit-learn

You can install these packages using pip:
```bash
pip install pandas seaborn matplotlib scikit-learn
```

## Steps to Run the Analysis

1. **Load the Data**
    ```python
    import pandas as pd
    df = pd.read_csv('housing.csv')
    df.info()
    df.dropna(inplace=True)
    df.info()
    ```

2. **Prepare the Data**
    ```python
    x = df.drop(['median_house_value'], axis=1)
    y = df['median_house_value']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    ```

3. **Explore and Visualize the Data**
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    train_df = X_train.join(y_train)
    train_df.hist(figsize=(15,10))
    train_df.ocean_proximity.value_counts()

    train_df = train_df.join(pd.get_dummies(train_df.ocean_proximity, dtype=float)).drop(['ocean_proximity'], axis=1)
    plt.figure(figsize=(15,10))
    sns.heatmap(train_df.corr(), annot=True, cmap="YlGnBu")
    ```

4. **Feature Engineering**
    ```python
    train_df['bedroom_ratio'] = train_df['total_bedrooms'] / train_df['total_rooms']
    train_df['household_rooms'] = train_df['total_rooms'] / train_df['households']
    plt.figure(figsize=(15,10))
    sns.heatmap(train_df.corr(), annot=True, cmap="YlGnBu")
    ```

5. **Train Linear Regression Model**
    ```python
    from sklearn.linear_model import LinearRegression
    X_train, y_train = train_df.drop(['median_house_value'], axis=1), train_df["median_house_value"]
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    ```

6. **Prepare Test Data**
    ```python
    test_df = X_test.join(y_test)
    test_df = test_df.join(pd.get_dummies(test_df.ocean_proximity, dtype=float)).drop(['ocean_proximity'], axis=1)
    test_df['bedroom_ratio'] = test_df['total_bedrooms'] / test_df['total_rooms']
    test_df['household_rooms'] = test_df['total_rooms'] / test_df['households']
    X_test, y_test = test_df.drop(['median_house_value'], axis=1), test_df["median_house_value"]
    ```

7. **Evaluate Linear Regression Model**
    ```python
    lr_score = lr.score(X_test, y_test)
    print("Linear Regression Score:", lr_score)
    ```

8. **Train and Evaluate Ridge Regression Models**
    ```python
    from sklearn.linear_model import Ridge

    ridge_alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
    for alpha in ridge_alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        ridge_score = ridge.score(X_test, y_test)
        print(f"Ridge Regression Score (alpha={alpha}):", ridge_score)
    ```

## Results

The following scores were obtained for different models:
- **Linear Regression Score**: 0.6513824308446607
- **Ridge Regression Scores**:
  - Alpha 1.0: 0.6520316925844221
  - Alpha 2.0: 0.6523952407118179
  - Alpha 3.0: 0.6526178585667151
  - Alpha 4.0: 0.6527592688417929
  - Alpha 5.0: 0.6528488100151972


## Conclusion

This project demonstrates a comprehensive approach to analyzing housing market data using linear and Ridge regression models. By visualizing data, performing feature engineering, and evaluating multiple models, we can gain insights into the factors influencing house prices.

## Dataset

- The dataset used for this analysis is the California Housing Prices dataset from Kaggle. It contains data about various features of houses in California, including the median house value, total rooms, total bedrooms, population, households, median income, and proximity to the ocean.

For further information and exploration, you can download the dataset from [here](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data).