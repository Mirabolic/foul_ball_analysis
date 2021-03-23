# Foul Ball Analysis
In the forthcoming paper "Foul Ball Rates and Injuries at Major League Baseball Games" by Milsten et al., the authors collected injury data from first aid stations and ambulance run sheets at MLB stadiums to estimate the risk of foul ball injury.  This repository contains the corresponding tools used for web scraping, data cleaning and statistical analysis.  We have elected to share the code publicly to facilitate transparency and reproducibility of the analysis.

The software development and statistical analyses were performed during 2019-2021 by [Bill Bradley](https://www.linkedin.com/in/bill-bradley-a1614314/) in consultation with [Andrew Milsten](https://www.umassmed.edu/emed/faculty-old/milsten/) and [Karl Knaub](https://www.linkedin.com/in/karl-knaub-ab10701/).

  * [Installation](#installation)
     * [System-level Installation](#system-level-installation)
     * [Virtual-env Installation](#virtual-env-installation)
  * [Quickstart Guide](#quickstart-guide)
  * [Processing Details](#processing-details)
     * [Provide the Raw Data](#provide-the-raw-data)
     * [Specify the Teams](#specify-the-teams)
     * [Scrape the Data](#scrape-the-data)
     * [Results](#results)
  * [Future Extensions](#future-extensions)

## Installation

We assume you have already installed `python` and `pip`. This code has been tested with `python 3.8.5` but may work with other versions.

### System-level Installation
`cd` into the directory where this `README.md` file lives and run:
```
pip install -r requirements.txt --user
```
 
### Virtual-env Installation

You may prefer to install modules within a virtual environment to keep the Python module versions separate from the rest of your system.  If so, `cd` into the directory where this `README.md` file lives and run:
```
virtualenv -p python venv
source ./venv/bin/activate
```
Next, install the required dependencies:
```
pip install -r requirements.txt
```

## Quickstart Guide

You need to provide two files:
* An Excel file with worksheets for each MLB team; each row corresponds to a single medical event for one person.  You probably need to contact the authors of the paper to get a copy of this file.  For the sake of argument, let's call the file `raw_medical.xlsx`.
* A CSV file specifying additional information about the MLB teams.  An example file can be found at `sample_teams.csv`; copy your version to `data/teams.csv`.

Once those files are in place, run:
```
scrape_and_analyze.py --raw raw_medical.xlsx
```
Running that script takes us about 12 seconds.  A set of human-readable results will be written to `results.txt` and figures will be saved to the `pix/` directory.

## Processing Details

This analysis requires two data sources:

* Records of injuries associated with MLB games (e.g., EMS patient care run sheets or first-aid station logs). 
* A collection of historical baseball statistics (e.g., game dates, foul ball counts, etc.)  We found https://www.baseball-reference.com to be a careful and comprehensive resource for this purpose.  We include a set of Python scripts for web scraping the necessary data from the website.

The rest of this section provides detailed instructions for replicating the web scraping, data cleaning and statistical analysis.

### Provide the Raw Data
You, the user, need to provide a file containing the medical data. Unfortunately, because of the sensitive nature of this data, we cannot include this file in a public repository; if you need access, please contact the authors of the paper.

The data should consist of a single Excel file, with one (or more) worksheet for each MLB team of interest.  For clarity, we will refer to this file in the following as `raw_medical.xlsx`.  The data should include (at least) columns
```
'Date', 'Age', 'Gender', 'Mechanism', 'Primary Dx'
```
In the interest of reproducibility, we include the `MD5` sum of the actual data file used in the paper:
```
MD5 (data/foul_ball_cleaned_2020_09_10.xlsx) = 8b4e9b3d8c8562a565edc28af774f217
```

### Specify the Teams
The `raw_medical.xlsx` file may have data from multiple teams, some not necessarily in the analysis.  Additionally, there may have multiple worksheets corresponding to the same team.  We need a way to keep track of which teams are of interest and anonymize them appropriately.

This is accomplished by writing a master file `data/teams.csv`.  The file has a row for each worksheet referenced in the `raw_medical.xlsx` file. It contains the following columns:
* `team`: This is the standard abbreviation for the MLB team.  C.f., for example, https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Baseball/Team_abbreviations
* `team_full_name`: The full name of the baseball team.  This field is only included to make the CSV file slightly more readable.
* `excel_sheet_name`: The name of the Worksheet in the Excel file corresponding to the team.
* `anonymized_name`: We anonymize the team names for the paper; provide the anonymized name here.  We used "MLB #1" and "MLB #2".
* `event_type`: Was the medical event a patient presentation ("PP") or a hospital tranportation ("TH")?

Note that if the data from a single team is spread across multiple Excel worksheets, use multiple rows in this file to specify all the data.

In case of ambiguity, we provide a hypothetical `sample_teams.csv`, using some MLB teams that have been defunct for the past century as examples.

### Scrape the Data

Now that you have provided the two files above (`data/teams.csv` and, for example, `raw_medical.xlsx`), the rest of the process is automated.  Run
```
./scrape_and_analyze --raw raw_medical.xlsx
```
The script will attempt to reuse information from previous runs.  For example, after it scrapes the website, it will cache the results and avoid re-downloading the files.  If, for some reason, you wish to rerun everything from scratch, you can do something like:
```
cp data/teams.csv .   # Save the "teams.csv" file; you'll still need that
rm data/*.csv data/*.shtml data/results.txt   # Clobber everything else
cp teams.csv data/    # Copy the "teams.csv" file back into place
./scrape_and_analyze --raw raw_medical.xlsx
```

### Results

A set of human-readable summaries are written to `data/results.txt`.  The output is fairly copious (including results of fitting numerous GLMs).  To get to the relevant part, you probably want to search for the string `summarize_data` and read the sections immediately following that.

Graphs and figures are written to `pix/` as `PNG` files.


## Future Extensions

This code is translated from a more fully-featured private repository.  (Because of the desire to anonymize the MLB teams, we cannot us tha code directly, but needed to rewrite it for this public repository.)

In the private repository, we extract pitch-by-pitch information, which provides a count of the actual number of foul balls per game, in addition to the pitchers and batters.  The web scraping and processing is much larger, so running the scripts can easily take over 10 hours to run. Because of the complexity, and because the results were largely irrelevant to paper, we have not included these extensions at the moment.
