Notes on how we handle the intercept in scikit-learn in LinearRegression(fit_intercept=True) and Ridge(fit_intercept=True, alpha=0) when `X.T @ X` is singular.

To build [the PDF with rendered math notation](https://raw.githubusercontent.com/ogrisel/minimum-norm-ols/main/minimum-norm-ols-intercept.pdf) with:

- install [typst](https://github.com/typst/typst)
- then run:

```
typst minimum-norm-ols-intercept.typ
```

