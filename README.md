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

### Will this get me banned?

Fieldplan does not touch any of the Niantic's servers, so it is perfectly
within the Terms of Service. All of the data comes from the spreadsheet and
from the Google Directions API.


# Why use spreadsheets

1. they are easy to edit to input portals
2. they come preinstalled on all android phones
3. the generated plans are easy to tweak and reorder manually if you find an improvement
4. it's easy to mark on which step of the plan you are

The main reason why I hacked on maxfield is to make it more convenient for
biking, as having a simple plan to follow allows me to concentrate more on
biking and less on figuring out what to do next.

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
3. run the fieldplan command with the `-g {yourkey}` switch once
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

The default is 5,000 random iterations to find the best fielding plan. The
bisecting and fielding is done randomly, largely because finding efficient
movement plans for a set of geographical coordinates is one of those "NP hard"
problems (look up the "Travelling Salesman Problem"). There is an optimization
step after each random plan to fix the worst inefficiencies, so the results
after each iteration are already tweaked. In my personal experience using it,
I've found 10,000 a good number of iterations to generate decent plans.

Generally:

- 500 iterations: not very good plans
- 5,000 iterations: good plans
- 10,000 iterations: very good plans

Since iterations are largely random, it's entirely possible to find the best
possible plan on your first run, and to only find terrible plans even after
10,000 iterations. YMMV.

## Getting lots of keys from portals

Getting lots of keys used to be difficult, but really isn't any longer. If
you're good at glyphing, then you can expect to get 2 keys almost each time
you hack a portal.

Your strategy, therefore, should be to always glyph-hack with "More". If you're
capturing by yourself, that should limit you to 3-glyph portals, and that's not
too hard to do -- you can even throw in a "Complex" to speed things up.

### Plans that want you to get 6+ keys from a portal

Because of the optimization routines, you may end up with plans that require
you to get lots and lots of keys from a single portal. This may seem crazy, 
but if you are using Heat Sinks, this actually results in faster gameplay
than plans where you need 3-4 keys from every portal.

For example, if you have a portal requiring 8 keys, you would:

1. Speed-hack to get 2 keys (1st hack)
2. Add a Rare Heat Sink mod
3. Immediately speed-hack for 2 more keys (free hack)
4. Wait 2 minutes, speed-hack for 2 more keys (2nd hack)
5. Wait 2 minutes, speed-hack for 2 more keys (3rd hack)
6. Wait 2 minutes, speed-hack again if previous hacks didn't always get you 2 keys (4th hack)
7. Install a Multihack if you were unlucky and didn't get 8 keys

As you see, getting as many as 8 keys usually requires a single RHS mod and
about 6 minutes of waiting for portal cooldown. This ends up much faster
than installing common Heat Sinks at every other portal to get 3-4 keys, and you
end up using fewer Heat Sink mods.

Note, that the plan instructions will always tell you how many keys you need
for the portal before you leave. Often, even if a portal requires 5-6 keys,
you may not need to get them all at once.

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

Running with `--cooling none` is recommended if you have lots of time, don't
mind extra moving around, or don't want to spend your Heat Sink mods.

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

*Note:* This is an experimental feature and currently requires significantly
more iterations to find efficient plans, so run it with `-i 50000` and higher.

This probably happened to you -- you found an area with lots of uncaptured
portals, but you only have a limited amount of time to play. You can give
Fieldplan that large list of portals and ask it to find you a plan that would
give you maximum AP (or MU, with `-u`) within the time constraint specified.
The software will try various subsets of portals until it finds something that
satisfies the parameters.

For example, there's a historical site with 25 portals, but fielding them all
would take over 3 hours:

- Create the spreadsheet with all 25 portals
- Run fieldplan with `--maxtime 120`

Fieldplan will try to find the most efficient plan that will take no more than
2 hours to execute.

### Setting minimal AP

Since Fieldplan prioritizes plans with highest AP (or MU) per minute of
gameplay, it's possible that the most efficient plan it finds will contain only
a few portals from the list. This especially tends to happen when prioritizing
MU over AP.

You can pass `--minap` to tell Fieldplan to not consider plans resulting in
too few total AP points. For example, to get plans with at least 50,000 AP, run
`--minap 50000`.

## Copy-pasting portal lists from IITC

Manually inputting portals can be tedious, so there is a way to copy and paste
the list from IITC. You will need:

- [IITC](https://iitc.me/desktop/), obviously
- [Multi-Export Plugin](https://github.com/modkin/Ingress-IITC-Multi-Export/raw/master/multi_export.user.js)

Here's how to use it:

- Draw a polygon around the portals you are interested in
- Click on "Multi-Export"
- Click on `XXX` in the "Polygon/TSV" column
- Copy all entries in the text area
- Start a new spreadsheet
- Paste in the `A1` cell

Fieldplan needs Portal names in the column `A` and Intel URLs in the column `B`,
so you will need to either:

- delete columns `D`, `C`, `A`, or
- rearrange the columns to be in the expected order

Fieldplan will ignore anything not in columns `A` and `B`.

*CAUTION: IITC is not an official resource provided by Niantic, and your use
of it [may be against their Terms of Service](https://iitc.me/faq/#ban).*

## If something is not working

You can open a GitHub issue if something is not working for you, but please
keep in mind that this is entirely a hobby project and I may not have a chance
to help you out.

Happy fielding!

ENL agent: mricon
