import argparse
import numpy as np
import pybamm


def build_blended_model(blend_ratio=0.5):
    """Create DFN model with blended positive electrode."""
    options = {
        "particle": "Fickian diffusion",
        "positive electrode": "composite",  # requires PyBaMM >=23
    }
    model = pybamm.lithium_ion.DFN(options=options)

    # Set volume fraction of active material for each component
    model.param.update({
        "Positive electrode active material volume fraction [s.s-1]": blend_ratio,
        "Positive electrode 2 active material volume fraction [s.s-1]": 1 - blend_ratio,
    })
    return model


def load_params():
    ncm = pybamm.ParameterValues("Chen2020")
    lfp = pybamm.ParameterValues("Xu2019")
    # Merge parameter sets; LFP parameters for second positive material
    ncm.update({
        k: lfp[k] for k in lfp if "positive electrode 2" in k
    }, check_already_exists=False)
    return ncm


def run_simulation(blend_ratio=0.5):
    model = build_blended_model(blend_ratio)
    params = load_params()
    experiment = pybamm.Experiment([
        'Charge at 1C for 1 hour',
        'Rest for 10 minutes',
        'Discharge at 1C to 2.5 V'
    ])
    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
    sim.solve()
    sim.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', type=float, default=0.5,
                        help='fraction of NCM in the positive electrode (0-1)')
    args = parser.parse_args()
    run_simulation(args.blend)
