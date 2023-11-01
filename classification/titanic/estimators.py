from sklearn.linear_model import LogisticRegression

ESTIMATORS = {
    "lgr":
        {
            "estimator": LogisticRegression,
            "hyperparameters_space":
                {
                    "tol": [1e-4, 1e-3, 1e-2],
                    "C": [0.1, 0.25, 0.5, 0.75, 1],
                    "random_state": [42],
                    "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                    "penalty": ["l2"]
                },
            "scale": True
        },

}