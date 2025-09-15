import matplotlib.pyplot as plt

def scatter_feature_vs_target(df, features, target):
    for col in features:
        plt.figure()
        plt.scatter(df[col], df[target], alpha=0.5)
        plt.xlabel(col)
        plt.ylabel(target)
        plt.title(f'{col} vs {target}')
    plt.show(block=True)  # <-- keeps plots open until you close manually
