import argparse
import json
import os

import numpy as np
import jax

import config
import data_generation
import evaluation
import methods


def select_default_scenario():
    for scenario in config.SCENARIOS:
        if scenario.get("num_interventions_true_K", 0) == 2:
            return scenario
    return config.SCENARIOS[0]


def summarize_samples(samples, lower_q=2.5, upper_q=97.5):
    mean = np.mean(samples, axis=0)
    lower = np.percentile(samples, lower_q, axis=0)
    upper = np.percentile(samples, upper_q, axis=0)
    return mean, lower, upper


def build_benchmark_cis(benchmarks):
    cis = {}
    for key in ["cCFR_model", "aCFR_model"]:
        if key in benchmarks:
            cis[f"{key}_lower"] = benchmarks[key]
            cis[f"{key}_upper"] = benchmarks[key]
    return cis


def build_fscfr_results_with_ci(fscfr_results):
    out = dict(fscfr_results)
    for key in ["fsCFR_factual_mean", "fsCFR_counterfactual_mean"]:
        if key in fscfr_results:
            suffix = "factual" if "factual" in key else "counterfactual"
            out[f"fsCFR_{suffix}_lower"] = fscfr_results[key]
            out[f"fsCFR_{suffix}_upper"] = fscfr_results[key]
    return out


def main():
    parser = argparse.ArgumentParser(description="Demo: evaluate methods on a single scenario.")
    parser.add_argument("--scenario-id", default=None, help="Scenario ID (e.g., S03).")
    parser.add_argument("--seed", type=int, default=config.GLOBAL_BASE_SEED, help="Random seed.")
    parser.add_argument("--num-warmup", type=int, default=None, help="Override warmup steps.")
    parser.add_argument("--num-samples", type=int, default=None, help="Override samples.")
    parser.add_argument("--num-chains", type=int, default=None, help="Override chains.")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR_PLOTS, help="Output directory for plots.")
    args = parser.parse_args()

    scenario = None
    if args.scenario_id:
        for s in config.SCENARIOS:
            if s["id"] == args.scenario_id:
                scenario = s
                break
        if scenario is None:
            raise ValueError(f"Scenario ID not found: {args.scenario_id}")
    else:
        scenario = select_default_scenario()

    if args.num_warmup is not None:
        config.NUM_WARMUP = args.num_warmup
    if args.num_samples is not None:
        config.NUM_SAMPLES = args.num_samples
    if args.num_chains is not None:
        config.NUM_CHAINS = args.num_chains

    sim_data = data_generation.simulate_scenario_data(scenario, run_seed=args.seed)

    benchmarks = methods.run_all_benchmarks(sim_data)
    fscfr_results = build_fscfr_results_with_ci({
        "fsCFR_factual_mean": benchmarks.get("fsCFR_factual_mean", np.array([])),
        "fsCFR_counterfactual_mean": benchmarks.get("fsCFR_counterfactual_mean", np.array([])),
    })
    benchmarks_r_t = {
        "cCFR_model": benchmarks.get("cCFR_model", np.array([])),
        "aCFR_model": benchmarks.get("aCFR_model", np.array([])),
    }
    benchmark_cis = build_benchmark_cis(benchmarks_r_t)

    posterior_samples, _ = methods.fit_proposed_model(sim_data, jax.random.PRNGKey(args.seed))
    r_t_key = "r_t" if "r_t" in posterior_samples else "p"
    r_cf_key = "r_cf" if "r_cf" in posterior_samples else "p_cf"
    r_t_mean, r_t_lower, r_t_upper = summarize_samples(posterior_samples[r_t_key])
    r_cf_mean, r_cf_lower, r_cf_upper = summarize_samples(posterior_samples[r_cf_key])

    plot_data = {
        "true_r_t": sim_data["true_r_0_t"][:config.T_ANALYSIS_LENGTH],
        "true_rcf_0_t": sim_data["true_rcf_0_t"][:config.T_ANALYSIS_LENGTH],
        "true_intervention_times_0_abs": sim_data["true_intervention_times_0_abs"],
        "estimated_r_t_dict": {
            "sCFR": {
                "mean": r_t_mean[:config.T_ANALYSIS_LENGTH],
                "lower": r_t_lower[:config.T_ANALYSIS_LENGTH],
                "upper": r_t_upper[:config.T_ANALYSIS_LENGTH],
                "cf_mean": r_cf_mean[:config.T_ANALYSIS_LENGTH],
                "cf_lower": r_cf_lower[:config.T_ANALYSIS_LENGTH],
                "cf_upper": r_cf_upper[:config.T_ANALYSIS_LENGTH],
            },
            "cCFR_model": {"mean": benchmarks_r_t["cCFR_model"][:config.T_ANALYSIS_LENGTH]},
            "aCFR_model": {"mean": benchmarks_r_t["aCFR_model"][:config.T_ANALYSIS_LENGTH]},
            "fsCFR_model": {
                "factual_mean": fscfr_results["fsCFR_factual_mean"][:config.T_ANALYSIS_LENGTH],
                "cf_mean": fscfr_results["fsCFR_counterfactual_mean"][:config.T_ANALYSIS_LENGTH],
            },
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    evaluation.plot_cfr_timeseries_from_data(
        scenario_id=scenario["id"],
        mc_run_idx=0,
        plot_data=plot_data,
        output_dir=args.output_dir
    )

    evaluator = evaluation.CFREvaluatorVisualizer(output_dir=args.output_dir)
    result = evaluator.evaluate_single_run(
        sim_data=sim_data,
        posterior_scfr=posterior_samples,
        benchmarks_r_t=benchmarks_r_t,
        benchmark_cis=benchmark_cis,
        its_results=fscfr_results
    )

    metrics_summary = {
        name: res.metrics for name, res in result.model_results.items()
    }
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
