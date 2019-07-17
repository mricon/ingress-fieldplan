# Introduction

This is for Ingress. If you don't know what that is, you're lost.

This is a heavily modified original [maxfield](https://github.com/tvwenger/maxfield)
software that will generate an easy-to-follow fielding plan using Google Spreadsheets.
The benefits over the original maxfield program are:

1. Works on Python 3
2. Generates more efficient solutions requiring fewer iterations
3. Generates an efficient capture plan in addition to the fielding plan
4. Uses Google Directions API for precise distances (recommended, requires an API key)
5. Supports walking, biking, and driving plans (mostly relevant with Google Directions)

The main perk of this implementation is the Google Spreadsheet fielding plan,
which provides easy-to-follow step-by-step instructions for the agent. This is
how it looks on a mobile phone:

<img src="https://raw.githubusercontent.com/mricon/ingress-fieldplan/master/screenshots/spreadsheet-view.jpg" width="250">

Here are a few examples of annotated spreadsheets generated with fieldplan:

- [Walking](https://docs.google.com/spreadsheets/d/1TbwOCNpsvA7CjOTPv_98Iirjt_siOoAgoTKa0PTglgU)
- [Bicycling](https://docs.google.com/spreadsheets/d/1PXawfbKaOVKJ4PUo8qvOGHi6Xw1ldNy-Qsyer6-E1-Y)

Open those links on your phone and choose "Use the App" -- then switch to the
plan tab and zoom in for best readability.

# Why use spreadsheets

1. they are easy to edit to input portals
2. they come preinstalled on all android phones
3. the generated plans are easy to tweak and reorder manually if you find an improvement
4. it's easy to mark on which step of the plan you are

The main reason why I hacked on maxfield is to make it more convenient for
biking, as having a simple plan to follow allows me to concentrate more on
biking and less on figuring out what to do next.

# Will this get me banned?

Fieldplan does not touch any of the Niantic's servers, so it is perfectly
within the Terms of Service. All of the data comes from the spreadsheet and
from the Google Directions API.

# Prerequisites

This is a console python application. It expects a POSIX-compatible system
(Linux, OS X) and a virtualenv-3 setup. After initializing the environment,
you can install the required libraries using:

    pip install -r requirements.txt

If you've never used a console, python and pip before, then you'll have a bit
of a hard time at first, but it's not that hard to learn.

# Obtaining Google Spreadsheets credentials

To start generating fieldplans, you will need to first get a credentials.json
file and then generate a token. It's a bit annoying and complicated, but you
only have to do this once.

1. follow instructions on the [Python Quickstart page](https://developers.google.com/sheets/api/quickstart/python')
2. once you have credentials.json, save that file in the same directory as fieldplan
3. run ./obtainGSToken.py
4. copy the authorization link and open in your browser
5. allow access
6. copy the long string and paste it into the terminal where the script tells you
7. the access token will be saved in the ingress-fieldplan cache folder

# Obtaining Google Directions API key

This is also annoying and complicated, and you also have to only do this once.

CAUTION: Google will require you to set up billing for your project, so if
you're not in a position to put in a credit card, then you shouldn't bother
with this.  You should not get charged unless you're making many thousands of
API calls daily. Using your Directions API key with ingress-fieldplan should
be effectively free for you if you're not calculating hundreds of plans every
hour. Fieldplan also relies heavily on caching, so if we've looked up the
walking/biking/driving distance between two portals once, it will be stored in
local cache for all future lookups.

If you do set up your Google Directions API key, then you will greatly benefit
from much more accurate distances, especially for portals that are in close
proximity but require long detours.

1. go to the [Instructions page](https://developers.google.com/maps/documentation/directions/get-api-key)
2. click "Get Started" and go through the process
3. run the fieldplan command with the -g {yourkey} switch once
4. fieldplan will cache the key and use it automatically next time

# Testing it out

Once you have your credentials.json (and your Directions API key, if you
choose), you can run the following command to test out if it's working:

    ./fieldplan -n -s https://docs.google.com/spreadsheets/d/1TbwOCNpsvA7CjOTPv_98Iirjt_siOoAgoTKa0PTglgU/edit

If it didn't crash horribly, then you're in business!

# Creating your own plans

1. Go to the Intel Map and find the portals you want to field
2. A good number is around 15 portals, good for about 1 hour of gameplay plus getting around
3. Start a new Google Spreadsheet
4. Column A is portal names
5. Column B is for Intel Map portal links
6. To get a portal link, click on the portal, then click on the "Link" button at the top-right
7. We only need pll=x,y bits, but easiest is to copy-paste the whole URL
8. Blank rows will be ignored

Once you have all the portals entered, copy the spreadsheet URL and run the command:

    ./fieldplan -s https://docs.google.com/spreadsheets/d/xxx/edit#gid=0

Depending on how many portals you have in the spreadsheet, it will take
anywhere from a few seconds to 10-15 minutes to run with the default set of
iterations. The results will be saved as a new sheet and cached on your
computer, so if you run the same spreadsheet again, it will continue from the
previous best plan.

# Other commandline switches

Look at the output of

    ./fieldplan --help

to find all the knobs and levers you can tweak. Here are a few pointers:

## How many iterations to use?

The default is 10,000 random iterations to find the best fielding plan. The
bisecting and fielding is done randomly, largely because finding efficient
movement plans for a set of geographical coordinates is one of those "NP hard"
problems (look up the "Travelling Salesman Problem"). There is an optimization
step after each random plan to fix the worst inefficiencies, so the results
after each iteration are already tweaked. In my personal experience using it,
I've found 10,000 a good number of iterations to generate decent plans.

*Note:* the actual number of iterations is actually higher, because we reset the
counter after each new best plan is found. It's possible to find a new best plan
on iteration 9,999 and thus reset the counter back to 0 again.

Generally:

- 10 iterations: probably bad plans
- 1,000 iterations: okayish plans
- 10,000 iterations: decent plans
- 20,000 iterations: very good plans
- 100,000 iterations: total overkill

Since iterations are largely random, it's entirely possible to find the best
possible plan on your first run, and to only find terrible plans even after
100,000 iterations. YMMV.

## Using the -k switch to limit the number of keys required

The optimization steps will try to reduce how much you travel between portals,
so it's possible that the final plan will require crazy amounts of keys (see
more on that below). If you're in an area where you've never been before and
want to make sure that you don't need too many keys from each portal, you can
make sure by passing the `-k` switch.

## Getting lots of keys from portals

Getting lots of keys used to be difficult, but really isn't any longer. If
you're good at glyphing, then you can expect to get 2 keys each time you hack
a portal. If you speed-hack (with "Complex") on a 3+ glyph portal, then you
can get as many as 3 keys from a single hack.

Your strategy, therefore, should be to always glyph-hack with "More" and
"Complex". If you're capturing by yourself, that should limit you to 3-glyph
portals, and that's not too hard to do at "Complex" speeds.

## Plans that want you to get 6+ keys from a portal

Because of the optimization routines, you may end up with plans that require
you to get lots and lots of keys from a single portal. You may be tempted to
redo those with -k, but I find it is actually more convenient to have a single
portal requiring lots of keys than to have lots of portals needing 3-4 keys.
If you have a portal requiring 8 keys, you can:

1. Speed-hack to get 2 keys (1st hack)
2. Add a rare/very rare Heat Sink mod
3. Immediately speed-hack for 2 more keys (free hack)
4. Wait 1:30, speed-hack for 2 more keys (2nd hack)
5. Wait 1:30, speed-hack for 2 more keys (3rd hack)
6. Wait 1:30, speed-hack again if previous hacks didn't always get you 2 keys (4th hack)
7. Install a Multihack if you're unlucky and you don't yet have 8 keys

As you see, getting as many as 8 keys usually requires a single VRHS mod and
less than 5 minutes of waiting for portal cooldown. This ends up much faster
than installing common Heat Sinks at every other portal needing you to ensure
3-4 keys.

Note, that the plan instructions will always tell you how many keys you need
for the portal before you leave. Often, even if a portal requires 5-6 keys,
you may not need to get them all at once.

## Start and End waypoints

You will probably be planning your field ops either from home, on the way
from home to work/school, or from a parking/transit stop location. To generate
plans that are more efficient with those locations, you should add
them as waypoints to your spreadsheet.

- First, find the waypoint location on the intel map and zoom in as far in
  as possible for the most accurate result.
- With your waypoint at the center of the map, click "Link" and copy the
  URL in the pop-up box (just like with portal URLs).
- Use special name indicators at the start of the portal names:

  - `#!s Location Name` for your start waypoint
  - `#!e Location Name` for your end waypoint

For example:

  - Column A1: `#!s My Home`
  - Column B1: `https://intel.ingress.com/intel?ll=45.498803,-73.598872&z=21`
  - Column A2: `#!e My School`
  - Column B2: `https://intel.ingress.com/intel?ll=45.504427,-73.574309&z=21`
  
Waypoints can be either at the start or at the end of the portal list.

## Blocker waypoints

You can also add portals you need to visit to destroy blockers by using the
same logic as with start/end waypoints. Use the `#!b Portal Name` indicator
in the left column to mark that a portal is a blocker and not part of the
fielding plan.

*Note:* The software has no idea where the blocking links are, so you will
need to review the plan to make sure that you are not throwing early links
before destroying the blockers that would be in the way.
    
## Prioritizing MU capture (-u)

By default, fieldplan will try to maximize AP per minute of gameplay, but 
using the `-u` switch you can tell it to consider field sizes as well, in an
attempt to find plans that would also give you highest area capture per
minute of gameplay. 

*Note:* the software has no way of knowing the actual in-game MU density,
so it will simply give higher priority to larger fields.

## Generating plots

Passing the `-p` switch will generate a set of step-by-step PNG files that
allows you to preview the plan in action. Here's what it is for the Biking
example above:

![Plot example](https://raw.githubusercontent.com/mricon/ingress-fieldplan/master/screenshots/plotting.gif "Plot example")

You may need to install python-tkinter for it to work.

## I have an hour to play, find me a plan that works

*Note: This is an experimental feature.*

This probably happened to you -- you found an area with lots of uncaptured
portals, but you only have a limited amount of time to play. You can give
Fieldplan that large list of portals and ask it to find you a plan that would
give you maximum AP (or MU, with `-u`) within the time constraints specified.
The software will try various subsets of portals until it finds something that
satisfies the parameters.

For example, there's a historical site with 25 portals, but fielding them all
would take over 3 hours:

- Create the spreadsheet with all 25 portals
- Run fieldplan with `--maxtime 130 --mintime 110` flags

Fieldplan will try to find the most efficient plan that will take roughly
2 hours to execute.

*Note:* This is an experimental feature and currently requires significantly
more iterations to find efficient plans, so run it with `-i 50000` and higher.

### Cooling

While estimating the time to play, the software will assume that you will get
1.5 keys per hack and use Rare Heat Sinks to speed up portal cooldown.
If you only have regular Heat Sinks, you can specify that with `--cooling hs`.

Other options are:

- `rhs`: Rare Heat Sink (default)
- `hs`: Heat Sink
- `vrhs`: Very Rare Heat Sink
- `none`: don't use Heat Sinks at all
- `idkfa`: you have all the keys and hack/cooldown times should not be counted

## Copy-pasting portal lists from IITC

*Note: This is an experimental feature.*

Manually inputting portals can be tedious, so there is a way to copy and paste
the list from IITC. You will need:

- [IITC](https://iitc.me/desktop/), obviously
- The "portals-list" plugin

Navigate to the area you want to field in IITC, and try to zoom in so that
only the portals you are interested in are shown.

- Click on the "Portals List" link
- Select and copy all rows in the table
- Start a new spreadsheet
- In the `A1` cell, type: `#!iitc`
- Put the cursor into the `B1` cell and paste the portal list
- Remove any rows with portals you don't want
- You can add the waypoints below the IITC paste if you need

Here is an [example
spreadsheet](https://docs.google.com/spreadsheets/d/1D5dqZwWyZRwxdgy1OjmtB3md6uAO_yEd0hiF8C4QL4s/edit#gid=0)
to illustrate the format.

*CAUTION*: IITC is not an official resource provided by Niantic, and your use
of it may be against their Terms of Service.

## If something is not working

You can open a GitHub issue if something is not working for you, but please
keep in mind that this is entirely a hobby project and I may not have a chance
to help you out.

Happy fielding!

ENL agent: mricon
