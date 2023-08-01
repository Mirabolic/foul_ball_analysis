#!/usr/bin/env python

# Grab a litte data for teams and year ranges of interest
# (in particular, for 2005-2012 Baltimore Orioles)

import scrape_and_analyze
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare foul ball data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--team", type=str, help="Three letter abbreviation for team.", required=True
    )
    parser.add_argument(
        "--first_year",
        type=int,
        help="First year of interest (e.g., 2005).",
        required=True,
    )
    parser.add_argument(
        "--last_year",
        type=int,
        help="Last year of interest (e.g., 2012).",
        required=True,
    )

    args = parser.parse_args()
    assert args.first_year >= 1876  # First MLB game, in Philadelphia :)
    assert args.first_year <= args.last_year

    suffix = "_%s_%d_%d" % (args.team, args.first_year, args.last_year)
    scrape_and_analyze.results_file_name = os.path.join(
        scrape_and_analyze.data_dir, "results%s.txt" % suffix
    )

    df = scrape_and_analyze.grab_one_team_stats(
        team=args.team, first_year=args.first_year, last_year=args.last_year
    )
    file_name = os.path.join(
        scrape_and_analyze.data_dir, "basic_MLB_stats_%s.csv" % suffix
    )
    df.to_csv(file_name, index=False)
