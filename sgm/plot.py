import argparse
import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def returns_v_cleanup_steps(df):
    # Hack for current planner names
    df.loc[df["Planner"] == "SGM", "Planner"] = "Sparse Graphical Memory (ours)"
    df.loc[df["Planner"] == "SoRB", "Planner"] = "SoRB + Our Proposed Cleanup"

    ax = sns.lineplot(x="Cleanup Steps", y="Success Rate", hue="Planner", style="Planner", data=df, palette="viridis")
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.8))
    ax.set_title("How Performance Improves with Cleaning")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_directory = os.path.join(os.getcwd(), "plots", f"returns_v_cleanup_steps_{timestamp}")
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    output_file = os.path.join(output_directory, "returns_v_cleanup_steps.png")

    fig = ax.get_figure()
    fig.savefig(output_file, dpi=200)
    plt.close()


def cleanup_timing(df, max_cleanup_steps=100000):
    df["Time per Action Choice"] = df["Time to Choose Action"] / df["Action Attempts"]
    df = df.loc[
        (0 < df["Cleanup Steps"]) & (df["Cleanup Steps"] <= max_cleanup_steps), ["Planner", "Time to Clean Graph",
                                                                                 "Time per Action Choice"]]
    print(df.groupby(["Planner"]).describe())


def get_combined_dataframe(directories):
    df = pd.DataFrame()

    for directory in directories:
        filename = os.path.join(os.getcwd(), directory, "evaluation.csv")
        to_append = pd.read_csv(filename)
        df = df.append(to_append, ignore_index=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experimental results")
    parser.add_argument("logdirs", type=str, nargs="+", help="The directories of the experiment logs")
    parser.add_argument("--returns_v_cleanup_steps", dest="returns_v_cleanup_steps", action="store_true",
                        help="Plot the evaluation return versus the number of environment cleanup steps")
    parser.add_argument("--cleanup_timing", dest="cleanup_timing", action="store_true",
                        help="Report the amount of time to clean the graph")
    args = parser.parse_args()

    df = get_combined_dataframe(args.logdirs)
    if args.returns_v_cleanup_steps:
        returns_v_cleanup_steps(df)
    if args.cleanup_timing:
        cleanup_timing(df)
