from credit_risk.pipeline import run_all


def main() -> None:
    run_all(save=True, verbose=True)


if __name__ == "__main__":
    main()
