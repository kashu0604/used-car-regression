import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

df = pd.read_csv("used_car_sales.csv")
print("df.shape:", df.shape)
print("mean Sale_Price:", df["Sale_Price"].mean())

train, test = train_test_split(df, test_size=0.3, random_state=42)
print("train size:", len(train))
print("test size:", len(test))

formula_a = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident) + Num_Owners"
formula_b = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident)"

model_a = smf.ols(formula_a, data=train).fit()
model_b = smf.ols(formula_b, data=train).fit()


def report(name, result):
    print(f"\n===== {name} =====")
    print(result.summary())

    print(f"\n{name} coefficients (1 decimal):")
    for term, coef in result.params.items():
        print(f"  {term}: {coef:.1f}")

    print(f"\n{name} p-values (4 decimals):")
    for term, pval in result.pvalues.items():
        print(f"  {term}: {pval:.4f}")

    print(f"\n{name} R-squared: {result.rsquared:.3f}")
    print(f"{name} adjusted R-squared: {result.rsquared_adj:.3f}")


report("Model A", model_a)
report("Model B", model_b)

v1 = model_b
train_mean = train["Sale_Price"].mean()
print("\ntraining mean Sale_Price:", train_mean)

y_test = test["Sale_Price"]
pred_baseline = np.full(len(test), train_mean)
pred_v1 = v1.predict(test)


def score(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name} R-squared: {r2:.3f}")
    print(f"{name} RMSE: ${rmse:.0f}")


score("baseline", y_test, pred_baseline)
score("v1", y_test, pred_v1)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(pred_v1, y_test, label="v1", alpha=0.85)
ax.scatter(pred_baseline, y_test, label="baseline", alpha=0.85)
low = min(y_test.min(), pred_v1.min(), pred_baseline.min())
high = max(y_test.max(), pred_v1.max(), pred_baseline.max())
ax.plot([low, high], [low, high], linestyle="--", color="gray", label="predicted = actual")
ax.set_xlabel("Predicted sale price ($)")
ax.set_ylabel("Actual sale price ($)")
ax.set_title("Predicted vs actual sale price (test cars)")
ax.legend()
fig.tight_layout()
fig.savefig("pred_vs_actual.png", dpi=150)
plt.close(fig)

lots = pd.read_csv("auction_lots.csv")
COST = 1500


def should_buy(car, auction_price):
    predicted = float(v1.predict(pd.DataFrame([car])).iloc[0])
    if auction_price + COST < predicted:
        return "buy"
    return "pass"


def baseline_should_buy(car, auction_price):
    if auction_price + COST < train_mean:
        return "buy"
    return "pass"


def report_rule(name, decide):
    bought_lots = []
    profits = []
    for row in lots.to_dict(orient="records"):
        car = {
            "Mileage_K": row["Mileage_K"],
            "Age_Years": row["Age_Years"],
            "Horsepower": row["Horsepower"],
            "Accident": row["Accident"],
            "Num_Owners": row["Num_Owners"],
        }
        if decide(car, row["Auction_Price"]) == "buy":
            profit = row["Later_Sale_Price"] - row["Auction_Price"] - COST
            bought_lots.append(int(row["Lot"]))
            profits.append(profit)
    negative = sum(profit < 0 for profit in profits)
    print(f"\n{name} buys: {len(bought_lots)}")
    print(f"{name} negative profit: {negative}")
    print(f"{name} total profit: {sum(profits)}")
    print(f"{name} lots: {bought_lots}")


report_rule("v1", should_buy)
report_rule("baseline", baseline_should_buy)
