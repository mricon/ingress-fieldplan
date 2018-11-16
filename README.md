# Introduction

This is for Ingress. If you don't know what that is, you're lost.

This is a heavily modified original [maxfield](https://github.com/tvwenger/maxfield)
software that will generate an easy-to-follow fielding plan using Google Spreadsheets.
The benefits over the original maxfield program are:

1. Works on Python 3
2. Generates more efficient solutions requiring fewer iterations
3. Generates an efficient capture plan in addition to the fielding plan
4. Uses Google Directions API for precise distances (optional, requires an API key)
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
4. Rename the first sheet "Portals"
5. Column A is portal names
6. Column B is for Intel Map portal links
7. To get a portal link, click on the portal, then click on the "Link" button at the top-right
8. We only need pll=x,y bits, but easiest is to copy-paste the whole URL
9. Blank rows or rows where the value in column A starts with "#" will be ignored

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

Note: the actual number of iterations is actually higher, because we reset the
count after each new best plan is found. It's possible to find a new best plan
on iteration 9,999 and thus reset the counter back to 0 again.

Generally:

- 10 iterations: probably bad plans
- 1,000 iterations: okayish plans
- 10,000 iterations: decent plans
- 25,000 iterations: very good plans
- 100,000 iterations: total overkill

Since iterations are largely random, it's entirely possible to find the best
possible plan on your first run, and to only find terrible plans even after
100,000 iterations. YMMV.

## Using the -k switch to limit the number of keys required

The optimization steps will try to reduce how much you travel between portals,
so it's possible that the final plan will require crazy amounts of keys (see
more on that below). If you're in an area where you've never been before and
want to make sure that you don't need too many keys from each portal, you can
make sure by passing the -k switch.

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

## Generating plots

Passing the -p switch will generate a set of step-by-step PNG files that
allows you to preview the plan in action. Here's what it is for the Biking
example above:

![Plot example](https://raw.githubusercontent.com/mricon/ingress-fieldplan/master/screenshots/plotting.gif "Plot example")

You may need to install python-tkinter for it to work.

## If something is not working

You can open a GitHub issue if something is not working for you, but please
keep in mind that this is entirely a hobby project and I may not have a chance
to help you out.

Happy fielding!

ENL agent: mricon
