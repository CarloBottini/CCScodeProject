
# Analysis of any Optuna results CSV ( the 100 trial run or the  10 trial focused run)

# Produces:
#   - summary statistics (printed)
#   - top-N trials table (CSV)
#   - per-parameter scatter / box plots vs val_loss
#   - Spearman correlation heatmap
#   - fANOVA-based parameter importance plot

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.importance import get_param_importances



HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_CSV_DIR = os.path.join(PROJECT_ROOT, "CSVresults")


class OptunaResultsAnalyzer:
    """Read an Optuna results CSV and generate plots + tables."""

    PARAM_PREFIX = "params_"
    USER_ATTR_PREFIX = "user_attrs_"
    VALUE_COL = "value"   # this is val_loss

    def __init__(self, csv_path, output_dir, top_n=10):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.top_n = top_n
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None
        self.param_cols = None
        self.numeric_params = None
        self.categorical_params = None

    def load(self):
        self.df = pd.read_csv(self.csv_path)
        # Keep only completed trials 
        if "state" in self.df.columns:
            self.df = self.df[self.df["state"] == "COMPLETE"].copy()
        # Drop rows with no value (failed)
        self.df = self.df.dropna(subset=[self.VALUE_COL]).reset_index(drop=True)

        self.param_cols = [c for c in self.df.columns if c.startswith(self.PARAM_PREFIX)]
        # Split numeric vs categorical for plotting
        # Treat booleans as categorical, not numeric, so they get a proper box plot
        self.numeric_params = [
            c for c in self.param_cols
            if pd.api.types.is_numeric_dtype(self.df[c])
            and not pd.api.types.is_bool_dtype(self.df[c])
            and self.df[c].dropna().nunique() > 2  # also catches True/False stored as 0/1
        ]
        self.categorical_params = [c for c in self.param_cols if c not in self.numeric_params]
        print(f"Loaded {len(self.df)} completed trials from {self.csv_path}")
        print(f"Numeric params:     {[p.replace(self.PARAM_PREFIX,'') for p in self.numeric_params]}")
        print(f"Categorical params: {[p.replace(self.PARAM_PREFIX,'') for p in self.categorical_params]}")
        return self

    def summary(self):
        """Print + save a textual summary and the top-N trials."""
        df = self.df
        print("\n VALIDATION LOSS SUMMARY ")
        print(df[self.VALUE_COL].describe())

        top = df.nsmallest(self.top_n, self.VALUE_COL)
        cols_to_show = ["number", self.VALUE_COL] + self.param_cols
        # Add test metrics if present
        for attr in ["user_attrs_test_mae", "user_attrs_test_rmse", "user_attrs_test_r2"]:
            if attr in df.columns:
                cols_to_show.append(attr)

        top_table = top[cols_to_show].reset_index(drop=True)
        out_csv = os.path.join(self.output_dir, "top_trials.csv")
        top_table.to_csv(out_csv, index=False)
        print(f"\nTop {self.top_n} trials saved to {out_csv}")
        print(top_table.to_string(index=False))
        return top_table

    def plot_param_vs_loss(self):
        """Scatter for each numeric param vs val_loss; box for categoricals.
        Reveals bimodal behaviour and robust zones visually."""
        df = self.df

        for col in self.numeric_params:
            name = col.replace(self.PARAM_PREFIX, "")
            plt.figure(figsize=(7, 4.5))
            sc = plt.scatter(df[col], df[self.VALUE_COL],
                             c=df[self.VALUE_COL], cmap="viridis_r", s=45, alpha=0.85)
            plt.colorbar(sc, label="val_loss")
            plt.xlabel(name)
            plt.ylabel("val_loss")
            plt.title(f"{name} vs validation loss")
            # Use log scale for parameters that span orders of magnitude
            if df[col].max() / max(df[col].min(), 1e-9) > 50:
                plt.xscale("log")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"scatter_{name}.png"), dpi=150)
            plt.close()

        for col in self.categorical_params:
            name = col.replace(self.PARAM_PREFIX, "")
            plt.figure(figsize=(7, 4.5))
            sns.boxplot(x=df[col], y=df[self.VALUE_COL])
            sns.stripplot(x=df[col], y=df[self.VALUE_COL], color="black", alpha=0.4, size=3)
            plt.xlabel(name); plt.ylabel("val_loss")
            plt.title(f"{name} vs validation loss")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"box_{name}.png"), dpi=150)
            plt.close()

        print(f"Per-parameter plots saved to {self.output_dir}/")

    def plot_correlation_heatmap(self):
        """Spearman correlation between numeric params and val_loss.
        Spearman is preferred over Pearson here because it captures
        monotonic (not only linear) relationships."""
        cols = self.numeric_params + [self.VALUE_COL]
        corr = self.df[cols].corr(method="spearman")
        nice_labels = [c.replace(self.PARAM_PREFIX, "") for c in cols]
        corr.index = nice_labels; corr.columns = nice_labels

        plt.figure(figsize=(8, 6.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
        plt.title("Spearman correlation: hyperparameters vs val_loss")
        plt.tight_layout()
        out = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"Correlation heatmap saved to {out}")
        return corr

    def plot_param_importance(self):
        """Optuna's built in fANOVA-based parameter importance.
        Tells us which hyperparameters EXPLAIN the most variance in val_loss
        much more informative than raw correlation for non monotonic effects."""
        # Rebuild a study from the dataframe so we can use Optuna's importance API
        study = optuna.create_study(direction="minimize")
        for _, row in self.df.iterrows():
            params = {
                c.replace(self.PARAM_PREFIX, ""): row[c]
                for c in self.param_cols if pd.notna(row[c])
            }
            distributions = {}
            for c in self.param_cols:
                name = c.replace(self.PARAM_PREFIX, "")
                if pd.isna(row[c]):
                    continue
                if c in self.numeric_params:
                    lo, hi = self.df[c].min(), self.df[c].max()
                    if lo == hi:
                        hi = lo + 1e-6  # avoid degenerate distribution
                    distributions[name] = optuna.distributions.FloatDistribution(lo, hi)
                else:
                    distributions[name] = optuna.distributions.CategoricalDistribution(
                        sorted(self.df[c].dropna().unique().tolist())
                    )
            study.add_trial(optuna.trial.create_trial(
                params=params, distributions=distributions,
                value=float(row[self.VALUE_COL]),
            ))

        try:
            importances = get_param_importances(study)
        except Exception as e:
            print(f"[warning] Could not compute importance: {e}")
            return None

        names = list(importances.keys())
        values = list(importances.values())
        plt.figure(figsize=(7, 4.5))
        plt.barh(names[::-1], values[::-1], color="steelblue")
        plt.xlabel("Relative importance (fANOVA)")
        plt.title("Hyperparameter importance for val_loss")
        plt.tight_layout()
        out = os.path.join(self.output_dir, "param_importance.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"Parameter importance plot saved to {out}")
        print("Importances:", {k: round(v, 3) for k, v in importances.items()})
        return importances

    def run_all(self):
        self.load()
        self.summary()
        self.plot_param_vs_loss()
        self.plot_correlation_heatmap()
        self.plot_param_importance()
        print(f"\nAll outputs in: {self.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze an Optuna results CSV and generate plots."
    )
    # Default: the 100 trial study
    parser.add_argument(
        "--csv", default="optuna_results_fullDataset.csv",
        help="CSV filename inside CSVresults/, OR a full path to any CSV.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Subfolder name for outputs (default: derived from CSV filename).",
    )
    parser.add_argument("--top", type=int, default=10, help="How many top trials to show")
    args = parser.parse_args()

    # Resolve input CSV: accept either a bare filename (looked up in CSVresults/)
    # or a full/relative path provided by the user.
    if os.path.isabs(args.csv) or os.path.dirname(args.csv):
        csv_path = args.csv
    else:
        csv_path = os.path.join(DEFAULT_CSV_DIR, args.csv)

    # Build the output folder name
    if args.name is not None:
        out_name = args.name
    else:
        # example "optuna_results_fullDataset.csv" -> "plots_fullDataset"
        base = os.path.splitext(os.path.basename(csv_path))[0]
        base = base.replace("optuna_results_", "")
        out_name = f"plots_{base}"

    output_dir = os.path.join(HERE, out_name)

    OptunaResultsAnalyzer(csv_path, output_dir, args.top).run_all()



