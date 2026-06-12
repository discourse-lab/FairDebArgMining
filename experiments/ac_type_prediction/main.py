# Entry point for experiments
# from ex import llm_main
from ex import llm_crossval_main


def main():
    data_dir = "../../data/v2/gold_data/ac_types"
    ex_outdir = "ex"
    # llm_main.run_experiment(data_dir, ex_outdir)
    llm_crossval_main.run_experiment(data_dir, ex_outdir)


if __name__ == '__main__':
    main()
