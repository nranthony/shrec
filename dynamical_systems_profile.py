from shrec.models import RecurrenceManifold
from benchmarks.dynamical_systems import load_data
import seaborn as sns

def main():
    X, y_true = load_data()
    model = RecurrenceManifold(verbose=True)
    y_recon = model.fit_predict(X)
    sns.lineplot(y_recon)

if __name__ == "__main__":
    main()
