#!/usr/bin/env python

# Scripts for web scraping, data ingest and data cleaning.

import warnings
from bs4 import BeautifulSoup
import numpy as np
import nbinom_fit
import pandas as pd
import os
import subprocess
import argparse
import datetime
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()


data_dir = 'data'
pix_dir = 'pix'
results_file_name = os.path.join(data_dir, 'results.txt')
teams_file_name = os.path.join(data_dir, 'teams.csv')
extracted_file_name = os.path.join(data_dir, 'extracted_raw.csv')
mlb_stats_file_name = os.path.join(data_dir, 'basic_MLB_stats.csv')
merged_file_name = os.path.join(data_dir, 'merged.csv')
missing_game_file_name = os.path.join(data_dir, 'missing_game_estimates.csv')
missing_summary_file_name = os.path.join(data_dir, 'missing_summary.csv')
neg_binom_params_file_name = os.path.join(data_dir, 'neg_binom.csv')
pure_med_file_name = os.path.join(data_dir, 'pure_med_stats.csv')


#####################
# Utility functions #
#####################
results_fp = None


def lprint(x, suppress_stdout=False):
    if not suppress_stdout:
        print(x)
    global results_fp
    if results_fp is None:
        results_fp = open(results_file_name, 'w')
        results_fp.write(str(datetime.datetime.now()))
    results_fp.write('%s\n' % x)


def prettyprint(func):
    def pp(*args, **kwargs):
        lprint('')
        lprint(60 * '#')
        lprint('##       %s      %s' % (
            func.__name__, datetime.datetime.now()))
        lprint(60 * '#')

        return(func(*args, **kwargs))
    return(pp)


def check_cache(file_name_list):
    if type(file_name_list) == str:
        file_name_list = [file_name_list]
    lprint('Check if cached files exist: %s' % (', '.join(file_name_list)))

    for file_name in file_name_list:
        if not(os.path.exists(file_name)):
            lprint('Cache files missing; processing.')
            return(False)
    lprint('Cache files present; skipping...')
    return(True)

##################
# Data functions #
##################


def grab_teams():
    if not os.path.exists(teams_file_name):
        print('  User must provide file %f' % teams_file_name)
        print('  See README.md for a description and sample_teams.csv')
        print('  for an example.')
        raise ValueError('Missing teams.csv file')
    teams_df = pd.read_csv(teams_file_name)
    required_columns = ['team', 'team_full_name',
                        'excel_sheet_name', 'anonymized_name', 'event_type']
    for c in required_columns:
        if c not in teams_df.columns:
            raise ValueError('teams.csv missing column %s' % c)
    if len(teams_df) == 0:
        raise ValueError('teams.csv has no rows!')
    return(teams_df)


@prettyprint
def parse_raw_excel(args):
    print('Parsing raw Excel medical data')

    # If we have previously parsed the raw excel data, just
    # skip this step
    if check_cache(extracted_file_name):
        return

    # Otherwise, extract the part of the Excel file we will use
    if args.raw is None:
        raise ValueError('Must specify input Excel file with --raw')
    if not os.path.exists(args.raw):
        raise ValueError('Cannot find file %s' % args.raw)

    teams_df = grab_teams()
    xls = pd.ExcelFile(args.raw)
    combined_df = pd.DataFrame()
    for index, row in teams_df.iterrows():
        with warnings.catch_warnings():
            # Suppress irrelevant "Data Validation extension" warning
            warnings.simplefilter("ignore")
            accidents_df = pd.read_excel(xls, row.excel_sheet_name)
        if len(accidents_df) == 0:
            raise ValueError('Excel sheet %f is empty!' % row.excel_sheet_name)
        assert len(accidents_df) > 0

        core_columns = ['Date', 'Age', 'Gender', 'Mechanism', 'Primary Dx']
        # Restrict to columns of interest
        accidents_df = accidents_df[core_columns]
        # If all data is missing, we should certainly skip the row!
        accidents_df.dropna(axis='index', how='all', inplace=True)
        # If crucial columns are missing, we also need to drop row
        accidents_df.dropna(axis='index', subset=['Date', 'Mechanism'],
                            inplace=True)
        accidents_df.reset_index(drop=True, inplace=True)

        # Standardize labeling of "Gender" column
        #   Strip whitespace (e.g., "F" instead of "  F")
        accidents_df['Gender'] = accidents_df['Gender'].str.strip()
        #   Capitalize (e.g., "F" instead of "f")
        accidents_df['Gender'] = accidents_df['Gender'].str.upper()

        # Construct alternate representations of "Date"
        accidents_df[['epoch_second', 'year', 'month', 'day']] = 0
        ns = 1e-9
        for i in range(len(accidents_df)):
            d = accidents_df['Date'][i]
            for c in ['year', 'month', 'day']:
                accidents_df[c].values[i] = getattr(d, c)
            epoch = int(accidents_df['Date'].values[i].item()*ns)
            accidents_df['epoch_second'].values[i] = epoch

        # Broadcast the teams.csv info to the remaining rows
        for c in teams_df.columns:
            accidents_df[c] = row[c]

        combined_df = combined_df.append(accidents_df, ignore_index=True)
    combined_df.to_csv(extracted_file_name, index=False)


@prettyprint
def grab_basic_game_stats():
    print('Downloading basic baseball stats (like which days had games')
    print('during the years of interest.)')

    if check_cache(mlb_stats_file_name):
        return

    # Figure out which teams and time ranges we need
    extracted_df = pd.read_csv(extracted_file_name)
    combined_df = pd.DataFrame()
    MLB_columns = []
    for team, team_df in extracted_df.groupby('team'):
        first_year = int(team_df.year.min())
        last_year = int(team_df.year.max())
        print('Downloading basic stats for %s for years %d-%d' % (
            team, first_year, last_year))

        for year in range(first_year, last_year+1):
            out_filename = os.path.join(data_dir, '%s_%d.shtml' % (team, year))
            if not os.path.exists(out_filename):
                cmd = 'curl -o %s https://www.baseball-reference.com/teams/%s/%d-schedule-scores.shtml' % (  # noqa
                    out_filename, team, year)
                subprocess.call(cmd, shell=True)

        output_rows = []
        for year in range(first_year, last_year+1):

            html = open('data/%s_%d.shtml' % (team, year)).read()
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find("table")

            # Figure out data fields on first pass through.
            # Note that the website may change these fields,
            # so we need to be somewhat paranoid about handling them.
            if MLB_columns == []:
                for table_header in table.findAll('thead'):
                    for t in table_header.findAll('th'):
                        MLB_columns.append(t.string)

                # As of March 2021, this reads:
                # Gm#, Date, None, Tm, \xa0, Opp, W/L, R, RA, Inn, W-L,
                # Rank, GB, Win, Loss, Save, Time, D/N, Attendance, cLI,
                # Streak, Orig. Scheduled
                #  (Note that "None" is the Python None, not the string
                #   "None".)

                # Need to overwrite some weirdnesses.  Hope that the
                # ordering of these fields doesn't change.
                MLB_columns[0] = 'Year'  # Weird, but correct
                MLB_columns[2] = 'Boxscore'
                MLB_columns[4] = 'Home_game'

                # Relabel with saner names when possible
                relabels = {
                    'Tm': 'Team', 'Opp': 'Opposing_team',
                    'W/L': 'Win_loss_tie',
                    'R': 'Runs_scored_allowed', 'RA': 'Runs_allowed',
                    'Inn': 'Innings', 'W-L': 'Win_loss_record_after_game',
                    'GB': 'Games_back', 'DN': 'Daytime', 'D/N': 'Daytime',
                    'CLI': 'Championship_Leverage_Index',
                    'Orig. Scheduled': 'Orig_Scheduled'}
                MLB_columns = [relabels.get(c, c) for c in MLB_columns]

            # Extract data
            for table_row in table.findAll('tr'):
                columns = table_row.findAll('td')
                output_row = [year]
                for column in columns:
                    output_row.append(column.text)
                if len(output_row) == 1:
                    continue
                output_rows.append(output_row)

        df = pd.DataFrame(output_rows, columns=MLB_columns)

        # Represent data in cleaner ways
        df.Home_game = (df.Home_game.values != '@')
        df.Innings.values[df.Innings.values == ''] = '9'
        df.Innings = df.Innings.astype(int)

        # Attendance may have a few missing values.  Check that it's
        # not out of hand, and drop them.
        NaN_attendance = np.sum(df.Attendance.isna())
        if NaN_attendance > 10:
            raise ValueError(
                'Suspiciously many null attendance entries! (%d of %d)' %
                (NaN_attendance, len(df)))
        df['Attendance'].values[df.Attendance.values == ''] = np.nan
        df.dropna(axis='index', subset=['Attendance'], inplace=True)
        df['Attendance'] = df.Attendance.str.replace(',', '')
        df['Attendance'] = df['Attendance'].astype(int)
        df.drop(columns=['Boxscore'], axis='columns', inplace=True)
        df['Daytime'] = (df['Daytime'].values == 'D')
        df['walkoff'] = False
        df['epoch_second'] = 0
        month_to_int = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5,
                        'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                        'Nov': 11, 'Dec': 12}
        df['double_header'] = False
        df['double_header_game_count'] = 0
        df['game_length_minutes'] = 0
        df['day_of_week'] = ''
        for i in range(len(df)):
            # Observed values:
            # ['W', 'L-wo', 'L', 'W-wo', 'L &V']
            if df['Win_loss_tie'].values[i].endswith('-wo'):
                df['walkoff'].values[i] = True
                df['Win_loss_tie'].values[i] = df[
                    'Win_loss_tie'].values[i][:-3]
            if df['Win_loss_tie'].values[i].endswith(' &V'):
                df['Win_loss_tie'].values[i] = df[
                    'Win_loss_tie'].values[i][:-3]

            hours, minutes = df['Time'].values[i].split(':')
            df['game_length_minutes'].values[i] = 60 * \
                int(hours) + int(minutes)

            year = df['Year'].values[i]
            splitted = df['Date'].values[i].split()
            if len(splitted) == 3:
                (_, month, day_of_month) = splitted
            elif len(splitted) == 4:
                (_, month, day_of_month, game_count) = splitted
                assert game_count[0] == '(' and game_count[2] == ')'
                game_count = int(game_count[1]) - 1
                df['double_header'].values[i] = True
                df['double_header_game_count'].values[i] = game_count
            else:
                assert False

            month = month_to_int[month]
            day_of_month = int(day_of_month)
            dt = datetime.datetime(year=year, month=month, day=day_of_month)
            epoch_second = int(calendar.timegm(dt.timetuple()))
            df['epoch_second'].values[i] = epoch_second
            df['day_of_week'].values[i] = (
                datetime.datetime.fromtimestamp(epoch_second).strftime("%A"))

        # Restrict to only home games
        df = df[df['Home_game'].values]

        # Figure out what fraction of the season corresponds to each game.
        # There are 183 days in a standard season (prior to 2018):
        starting_day = {}
        for year in range(first_year, 1+last_year):
            starting_day[year] = (
                df[df['Year'].values == year]['epoch_second'].min())
        df['fraction_of_season'] = 0.0
        season_length_seconds = 183 * 86400
        for i in range(len(df)):
            sd = starting_day[df['Year'].values[i]]
            frac = 1.0 * (df['epoch_second'].values[i] -
                          sd) / season_length_seconds
            frac = min(frac, 1.0)
            assert frac >= 0.0
            df['fraction_of_season'].values[i] = frac

        df.reset_index(drop=True, inplace=True)
        combined_df = combined_df.append(df, ignore_index=True)
    combined_df.to_csv(mlb_stats_file_name, index=False)


@prettyprint
def merge_mlb_and_medical_data():
    print()

    if check_cache([merged_file_name, pure_med_file_name]):
        return

    # First, aggregate medical data by day.  That essentially means
    # "by game", except in the case of double-headers (although
    # we'll be discarding those below anyway.)
    med_df = pd.read_csv(extracted_file_name)
    med_df['foul_ball_injuries'] = 0
    med_df['non_foul_ball_injuries'] = 0
    for index, row in med_df.iterrows():
        # Looking for string "foul ball", but let's be a little paranoid
        if 'foul' in row['Mechanism']:
            med_df['foul_ball_injuries'].values[index] = 1
        else:
            med_df['non_foul_ball_injuries'].values[index] = 1
    raw_df = med_df.copy()
    keep_columns = ['Date', 'epoch_second', 'team',
                    'foul_ball_injuries', 'non_foul_ball_injuries',
                    'anonymized_name']
    med_df = med_df[keep_columns].groupby(
        ['Date', 'epoch_second', 'team']).sum().reset_index()
    med_df.rename(columns={'team': 'Team'}, inplace=True)

    # Next, clean up MLB data
    mlb_df = pd.read_csv(mlb_stats_file_name)
    # Drop double-headers
    mlb_df = mlb_df[~(mlb_df.double_header.values)].reset_index(drop=True)
    # Drop unlikely-to-be-used columns
    keep_columns = [
        'Year', 'Team', 'Win_loss_tie', 'Innings', 'Daytime', 'Attendance',
        'epoch_second', 'game_length_minutes', 'fraction_of_season']
    mlb_df = mlb_df[keep_columns]

    # Finally, join on game date.
    joint = ['Team', 'epoch_second']
    combined_df = mlb_df.merge(med_df, on=joint)

    # There should only be one match for each date; let's double-check.
    assert len(combined_df) == len(combined_df.groupby(joint).groups)

    combined_df.to_csv(merged_file_name, index=False)

    lprint('Total accident counts (only MLB games, no double-headers)')
    for team, team_df in combined_df.groupby(['Team']):
        lprint('%s:  FB=%d    non-FB=%d' % (
            team,
            team_df.foul_ball_injuries.sum(),
            team_df.non_foul_ball_injuries.sum()))

    ###################################
    # Next, take the raw extracted medical records and restrict to the dates of
    # MLB ball games
    good_epoch_set = set(combined_df.epoch_second.values)
    pure_med_df = raw_df[raw_df.epoch_second.isin(good_epoch_set)
                         ].reset_index(drop=True)

    lprint('Ages of victims.  [min < 25th < median < 75th < max]')
    for team, team_df in pure_med_df.groupby(['team']):
        age_dist = team_df.Age.values
        age_dist = age_dist[np.isfinite(age_dist)]
        lprint('%s:    %d  < %d  <  %d  <  %d  <  %3d (N=%5d)   Foul balls' % (
            team,
            np.quantile(age_dist, 0),
            np.quantile(age_dist, .25),
            np.quantile(age_dist, .50),
            np.quantile(age_dist, .75),
            np.quantile(age_dist, 1),
            len(age_dist)))
    pure_med_df.to_csv(pure_med_file_name, index=False)


@ prettyprint
def estimate_missing_games():
    if check_cache([missing_game_file_name, neg_binom_params_file_name,
                    missing_summary_file_name]):
        return

    df_key = pd.read_csv(teams_file_name)
    df_key.set_index('team', inplace=True)

    merged_df = pd.read_csv(merged_file_name)
    mlb_df = pd.read_csv(mlb_stats_file_name)
    # Drop double-headers.  (We have already restricted to home games.)
    mlb_df = mlb_df[~(mlb_df.double_header.values)].reset_index(drop=True)

    # The fraction of missing games probably differs for each team,
    # so we estimate each team's rate independently.
    mlb_grouped = mlb_df.groupby(['Team', 'Year'])

    game_count = {'present': {}, 'missing': {}}
    mean_missing_attendance = {}
    neg_binom_params_df = pd.DataFrame(columns=['Team', 'r', 'p'])
    missing_game_df = pd.DataFrame(columns=['Team', 'Year', 'missing_games',
                                            'present_games'])
    missing_summary_df = pd.DataFrame(
        columns=['Team', 'missing_games',
                 'present_games', 'present_attendance',
                 'corrected_games', 'corrected_attendance',
                 'first_year', 'last_year', 'event_type'])
    for team, team_df in merged_df.groupby(['Team']):
        lprint(40 * '=')
        lprint('Team: %s' % team)
        # Count how many (non-double-header) games show up and are missing
        # per year
        game_count['missing'][team] = {}
        game_count['present'][team] = {}
        mean_missing_attendance[team] = {}

        for year, year_df in team_df.groupby('Year'):
            game_count['present'][team][year] = len(year_df)
            game_count['missing'][team][year] = (
                len(mlb_grouped.groups[(team, year)]) - len(year_df))
            missing_game_df.loc[len(missing_game_df)] = (
                team, year, game_count['missing'][team][year],
                game_count['present'][team])

            # Figure out average per-game attendance, per year, across the
            # missing games.
            all_games_this_year_df = mlb_df.loc[
                mlb_grouped.groups[(team, year)]]
            total_attendance_missing_games = 0
            for index, row in all_games_this_year_df.iterrows():
                if row.epoch_second not in year_df.epoch_second.values:
                    total_attendance_missing_games += row.Attendance
            if game_count['missing'][team][year] > 0:
                mean_missing_attendance[team][year] = (
                    total_attendance_missing_games /
                    game_count['missing'][team][year])
            else:
                mean_missing_attendance[team][year] = 0

        lprint('      missing  present   E[attendance @ missing games]')
        years_in_order = sorted(game_count['present'][team].keys())
        for year in years_in_order:
            lprint('%d:  %3d    %3d         %3d' % (
                year,
                game_count['missing'][team][year],
                game_count['present'][team][year],
                mean_missing_attendance[team][year]))
        lprint('Totals:')
        lprint('       %3d    %3d         %3d' % (
            sum(game_count['missing'][team].values()),
            sum(game_count['present'][team].values()),
            sum(mean_missing_attendance[team].values())))
        lprint('')

        # Figure out weighted estimate of per-game attendance
        # during missing games
        tmp_num, tmp_denom = 0, 0
        for year in years_in_order:
            tmp_num += (mean_missing_attendance[team][year] *
                        game_count['present'][team][year])
            tmp_denom += game_count['present'][team][year]
        reweighted_missing_attendance_per_game = tmp_num / tmp_denom
        lprint('Reweighted missing attendance per game: %d' %
               reweighted_missing_attendance_per_game)

        # The merged data corresponds to all the games with at least one
        # medical event (i.e., we observe the zero-censored count of events).
        # Fit the distribution (i.e., minimize the negative log-likelihood);
        # we model with a negative binomial (as it fits vastly better
        # than a Poisson).
        data = (team_df.foul_ball_injuries.values +
                team_df.non_foul_ball_injuries.values)
        # Initialize log-likelihood
        best_neg_llh = np.inf
        # Guess a reasonable value for a warm start
        c1 = np.sum(data == 1)
        c2 = np.sum(data == 2)
        guessed_num_zeros = max(int(2*c1-c2), 0)

        for num_zeros in range(2 * (1 + guessed_num_zeros)):
            uncensored = np.append(data, np.zeros(num_zeros))
            nbinom_params = nbinom_fit.nbinom_fit(uncensored)
            (r, p) = (nbinom_params['size'], nbinom_params['prob'])
            # We don't actually have to divide by len(data), since
            # everything's the same length, but let's keep
            # things apples-to-apples.
            neg_loglikelihood = (nbinom_fit.zero_censored_nllh(
                r=r, p=p, data=data) / len(data))

            if neg_loglikelihood < best_neg_llh:
                best_neg_llh = neg_loglikelihood
                best_num_zeros = num_zeros
                best_r, best_p = r, p
        r, p, num_zeros = best_r, best_p, best_num_zeros
        neg_loglikelihood = best_neg_llh

        neg_binom_params_df.loc[len(neg_binom_params_df)] = [team, r, p]

        lprint('r=%f, p=%f, neg_llh=%f' % (r, p, neg_loglikelihood))
        # Plot up to the second biggest value
        MM = np.sort(data)[-2]
        neg_binom_fit = nbinom_fit.nbinom_values(r=r, p=p, N=MM+1)
        # Scale up to data size
        lprint('Probability = 0: %f' % neg_binom_fit[0])
        # rescale = len(data) + best_num_zeros
        rescale = len(data)/(1-neg_binom_fit[0])
        neg_binom_fit *= rescale
        plt.plot(np.arange(len(neg_binom_fit)), neg_binom_fit,
                 label='Neg Binom (r=%.3f, p=%.3f)' % (
            r, p))
        lprint('Expecting %.2f (%d%%) games with no accidents!' % (
            neg_binom_fit[0], neg_binom_fit[0] * 100.0 / rescale))
        lprint('Expected attendance of those missing games: %d' %
               (reweighted_missing_attendance_per_game * (neg_binom_fit[0])))
        plt.xlim([0, MM])
        plt.xlabel('Number of %s events (FB + non-FB)' %
                   df_key['event_type'][team])
        plt.ylabel('Number of games')
        plt.title(df_key['anonymized_name'][team])
        file_name = os.path.join(pix_dir, 'missing_games_%s_%s.png' % (
            team, df_key['event_type'][team]))

        hist_y, hist_x = np.histogram(data,
                                      bins=np.max(data)-1)
        hist_x = hist_x[:-1]
        plt.plot(hist_x, hist_y, '.', label='Observed')
        plt.legend()
        plt.savefig(file_name, dpi=400)
        plt.close()

        # missing_summary_df = pd.DataFrame(
        #     columns=['Team', 'missing_games', 'present_games',
        #              'corrected_games', 'corrected_attendance',
        #               'first_year', 'last_year', 'event_type'])

        corrected_estimate_of_total_games = neg_binom_fit[0] + len(team_df)
        corrected_estimate_of_total_attendance = (
            team_df.Attendance.sum() +
            (reweighted_missing_attendance_per_game *
                neg_binom_fit[0]))
        missing_summary_df.loc[len(missing_summary_df)] = (
            team,
            sum(game_count['missing'][team].values()),
            sum(game_count['present'][team].values()),
            team_df.Attendance.sum(),
            corrected_estimate_of_total_games,
            corrected_estimate_of_total_attendance,
            years_in_order[0],
            years_in_order[-1],
            df_key['event_type'][team],
        )

    neg_binom_params_df.to_csv(neg_binom_params_file_name, index=False)
    missing_game_df.to_csv(missing_game_file_name, index=False)
    missing_summary_df.to_csv(missing_summary_file_name, index=False)


def twinned_plot_across_time(
        df=None,
        title=None,
        data1=None, data2=None,
        ylabel1=None, ylabel2=None,
        plotlabel1=None, plotlabel2=None):
    x = []
    y1 = []
    y1_low = []
    y1_high = []

    y2 = []
    y2_low = []
    y2_high = []
    for year in range(df.Year.min(), df.Year.max()+1):
        x.append(int(year))
        index = (df.Year.values == year)

        # data1
        num_injuries = data1[index].sum()
        num_games = np.sum(index)
        denom = num_games

        z = num_injuries / denom
        z_low = (num_injuries - np.sqrt(num_injuries)) / denom
        z_high = (num_injuries + np.sqrt(num_injuries)) / denom

        y1.append(z)
        y1_low.append(z_low)
        y1_high.append(z_high)

        # data2
        num_injuries = data2[index].sum()
        num_games = np.sum(index)
        denom = num_games

        z = num_injuries / denom
        z_low = (num_injuries - np.sqrt(num_injuries)) / denom
        z_high = (num_injuries + np.sqrt(num_injuries)) / denom

        y2.append(z)
        y2_low.append(z_low)
        y2_high.append(z_high)

    fig, ax1 = plt.subplots()
    # plt.figure()
    color = 'tab:blue'
    ax1.plot(x, y1, marker='o', label=plotlabel1, color=color)
    ax1.fill_between(x, y1_low, y1_high, alpha=.2,
                     label='%s $\\pm\\sigma$=1' % plotlabel1)
    ax1.set_xlabel('Year')
    ax1.set_ylabel(ylabel1, color=color)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes with same x-axis
    color = 'tab:green'
    ax2.plot(x, y2, marker='x', label=plotlabel2, color=color)
    ax2.fill_between(x, y2_low, y2_high, alpha=.2,
                     label='%s $\\pm\\sigma$=1' % plotlabel2, color=color)
    ax2.set_ylabel(ylabel2, color=color)
    ax2.grid(False)

    plt.title(title)
    fig.legend(loc=8, bbox_to_anchor=(.7, .15))
    plt.xticks(x)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', labelcolor=color)


def plot_smartphone_era():
    lprint('Plotting risk across time (smart phone era)')
    df_key = pd.read_csv(teams_file_name)
    df_key.set_index('team', inplace=True)

    merged_df = pd.read_csv(merged_file_name)

    for team, team_df in merged_df.groupby('Team'):
        team_anonymized = df_key.anonymized_name[team]

        twinned_plot_across_time(
            df=team_df,
            data1=team_df['foul_ball_injuries'],
            ylabel1='FB presentations per game',
            plotlabel1='FB',
            data2=team_df['non_foul_ball_injuries'],
            ylabel2='Non-FB presentations per game',
            plotlabel2='Non-FB',
            title='%s annual FB and non-FB injury rates' % team_anonymized
        )
        file_name = os.path.join(pix_dir, 'smartphone_%s_%s.png' % (
            team, df_key['event_type'][team]))
        plt.savefig(file_name, dpi=400)
        plt.close()


@prettyprint
def fit_glms():
    df_key = pd.read_csv(teams_file_name)
    df_key.set_index('team', inplace=True)

    merged_df = pd.read_csv(merged_file_name)

    merged_df['win'] = (merged_df['Win_loss_tie'].values == 'W')
    merged_df.drop(columns=['Date', 'Win_loss_tie'], inplace=True)
    for c in merged_df.columns:
        if merged_df[c].dtype in [bool, int]:
            merged_df[c] = merged_df[c].astype(float)

    for team, team_df in merged_df.groupby('Team'):
        team_anonymized = df_key.anonymized_name[team]
        lprint(5 * '\n', suppress_stdout=True)
        lprint(80 * '#', suppress_stdout=True)
        lprint('    %s    (%s)' % (team, team_anonymized))
        lprint(80 * '#', suppress_stdout=True)

        df = team_df.copy()
        target_FB = df['foul_ball_injuries']
        target_non_FB = df['non_foul_ball_injuries']
        df.drop(columns=['Team', 'foul_ball_injuries',
                         'non_foul_ball_injuries'],
                inplace=True)
        # Standardize
        df = (df - df.mean())/df.std()
        # We need to add a constant term to the dataframe for the
        # GLM fit
        df['constant'] = 1.0

        # Which sets of features should we examine?
        #   Year  Innings  Daytime  Attendance  epoch_second
        #  game_length_minutes  fraction_of_season constant win

        # Include all features and no features...
        experiment_list = [df.columns, ['constant']]
        # Add some favorites:
        experiment_list.append(['Year', 'game_length_minutes', 'constant'])
        # Include each feature individually...
        for c in df.columns:
            if c == 'constant':
                next
            experiment_list.append([c, 'constant'])

        for fields in experiment_list:
            for target in [target_FB, target_non_FB]:
                family = sm.families.Poisson()
                model = sm.GLM(target, df[fields], family=family)
                fitted_model = model.fit()
                lprint('\n\n', suppress_stdout=True)
                lprint(fitted_model.summary(), suppress_stdout=True)


@prettyprint
def summarize_data():
    df_key = pd.read_csv(teams_file_name)
    df_key.set_index('team', inplace=True)
    merged_df = pd.read_csv(merged_file_name)
    missing_summary_df = pd.read_csv(missing_summary_file_name)
    missing_summary_df.set_index('Team', inplace=True)
    pure_med_df = pd.read_csv(pure_med_file_name)
    mlb_stats_df = pd.read_csv(mlb_stats_file_name)
    neg_binom_df = pd.read_csv(neg_binom_params_file_name)
    neg_binom_df.set_index('Team', inplace=True)

    lprint('Key to team names:')
    lprint(df_key)
    lprint('')

    for team in df_key.index.unique():
        lprint('')
        lprint(30 * '-')
        lprint('    %s    (%s)' % (team, df_key.anonymized_name[team]))
        lprint(30 * '-')

        df = mlb_stats_df[mlb_stats_df.Team.values == team]
        lprint('Years: %d - %d' % (df.Year.min(), df.Year.max()))
        lprint('Totals across all MLB games for %s:' % team)
        lprint('   Total games:                    %d' % (len(df)))
        lprint('   Total attendance:               %d' % (df.Attendance.sum()))

        lprint('Totals across all MLB single-headers for %s:' % team)
        lprint('   Total games:                    %d' % (
            len(df) - df.double_header.sum()))
        lprint('   Total attendance:               %d' % (
            df[~(df.double_header.values)].Attendance.sum()))

        merged_team_df = merged_df[merged_df.Team.values == team]
        pure_med_team_df = pure_med_df[pure_med_df.team.values == team]
        lprint('Totals across MLB single-header games with medical data:')
        lprint('   Total games:                    %d' % (
            missing_summary_df.present_games[team]))
        lprint('   Total attendance:               %d' % (
            missing_summary_df.present_attendance[team]))
        lprint('   Total fans treated:             %d' % (
            len(pure_med_team_df)))
        lprint('   Total FB injuries:              %d' % (
            merged_team_df.foul_ball_injuries.sum()))
        lprint('   Total non-FB injuries:          %d' % (
            merged_team_df.non_foul_ball_injuries.sum()))
        lprint('   FB as %% of total injuries:      %.1f%%' % (
            100.0 * (merged_team_df.foul_ball_injuries.sum()) /
            (merged_team_df.foul_ball_injuries.sum() +
             merged_team_df.non_foul_ball_injuries.sum())))

        lprint('Patient characteristics (FB & non-FB, single-headers, only '
               'MLB games)')
        ages = pure_med_team_df.Age.values
        ages = ages[np.isfinite(ages)]
        lprint('   Age range=[%d - %d]  (median=%d)   N=%d' % (
            np.min(ages), np.max(ages), np.median(ages), len(ages)))
        lprint('   Gender: M=%.1f%%  F=%.1f%%          N=%d' % (
            (100.0 * np.sum(pure_med_team_df.Gender.values == 'M') /
             len(pure_med_team_df)),
            (100.0 * np.sum(pure_med_team_df.Gender.values == 'F') /
             len(pure_med_team_df)),
            len(pure_med_team_df)))

        lprint('FB Patient characteristics (single-headers, only MLB games)')
        pure_fb_df = pure_med_team_df[
            pure_med_team_df.foul_ball_injuries.values > 0]
        ages = pure_fb_df.Age.values
        ages = ages[np.isfinite(ages)]
        lprint('   Age range=[%d - %d]  (median=%d)   N=%d' % (
            np.min(ages), np.max(ages), np.median(ages), len(ages)))
        lprint('   Gender: M=%.1f%%  F=%.1f%%          N=%d' % (
            (100.0 * np.sum(pure_fb_df.Gender.values == 'M') /
             len(pure_fb_df)),
            (100.0 * np.sum(pure_fb_df.Gender.values == 'F') /
             len(pure_fb_df)),
            len(pure_fb_df)))

        lprint('Corrected totals across MLB single-header games with '
               'medical data:')
        lprint('   Corrected total games:          %.1f' % (
            missing_summary_df.corrected_games[team]))
        lprint('   Corrected total attendance:     %d' % (
            missing_summary_df.corrected_attendance[team]))
        lprint('   FB injuries per game (%s):      %.4f' % (
            df_key.event_type[team],
            ((merged_team_df.foul_ball_injuries.sum()) /
                (missing_summary_df.corrected_games[team]))))
        lprint('   FB injuries per 10K fans (%s):  %.4f' % (
            df_key.event_type[team],
            (10000.0 *
                (merged_team_df.foul_ball_injuries.sum()) /
                (missing_summary_df.corrected_attendance[team]))))

        lprint('Coefficients of maximum-likelihood fit of negative binomial '
               'distribution:')
        lprint('   r=%.6f   p=%.6f' %
               (neg_binom_df.r[team], neg_binom_df.p[team]))
    lprint('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare foul ball data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--raw', help='Excel file with raw medical diagnostic info')
    args = parser.parse_args()

    parse_raw_excel(args)
    grab_basic_game_stats()
    merge_mlb_and_medical_data()
    estimate_missing_games()
    plot_smartphone_era()
    summarize_data()
    fit_glms()
    lprint('Done!')
