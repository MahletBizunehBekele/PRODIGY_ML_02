from src.data_utils import load_train_test, impute_missing
from src.eda import scatter_feature_vs_target
from src.model import train_validate
from src.predict import train_full_and_predict


def main():    
    FEATURES = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
    TARGET = 'SalePrice'
    train_df, test_df = load_train_test('data/train.csv', 'data/test.csv')
    print("xyz")
    scatter_feature_vs_target(train_df, FEATURES, TARGET)
    print("xyz")
    X = impute_missing(train_df[FEATURES])
    y = train_df[TARGET]
    model, rmse, r2, details = train_validate(X, y)
    print(f"Validation RMSE: {rmse:.2f}, R²: {r2:.4f}")
    submission = train_full_and_predict(train_df, test_df, FEATURES, TARGET)

    print(submission.head())
    print(submission.describe())

    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")

if __name__ == "__main__":
    main()
