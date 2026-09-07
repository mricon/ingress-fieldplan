# Introduction

This is for Ingress. If you don't know what that is, you're lost.

This is a heavily modified original [maxfield](https://github.com/tvwenger/maxfield)
software that will generate an easy-to-follow fielding plan. The benefits over
the original maxfield program are:

1. Works on Python 3
2. Generates more efficient solutions requiring fewer iterations
3. Generates an efficient capture plan in addition to the fielding plan
4. Estimates how long the plan will actually take to play, including hacking
   and portal cooldown, and optimizes for AP per minute rather than raw AP
5. Uses Google Directions API for precise distances (optional, requires an API key)
6. Supports walking, biking, and driving plans (mostly relevant with Google Directions)

There are three ways to get your plan out of it:

- **a plain text file** -- no accounts, no API keys, nothing to set up
- **a single HTML page** (`-w`) -- a step-by-step checklist for your phone that
  needs no account and no network, and remembers how far you have got
- **a Google Spreadsheet** -- more work to set up, but easy to edit and reorder
  by hand, and it syncs across your devices

This is how the HTML plan and the spreadsheet look on a mobile phone:

<img src="https://raw.githubusercontent.com/mricon/ingress-fieldplan/master/screenshots/html-plan.jpg" width="250"> <img src="https://raw.githubusercontent.com/mricon/ingress-fieldplan/master/screenshots/spreadsheet-view.jpg" width="250">

Here are a few examples of annotated spreadsheets generated with fieldplan:

- [Walking](https://docs.google.com/spreadsheets/d/1TbwOCNpsvA7CjOTPv_98Iirjt_siOoAgoTKa0PTglgU)
- [Bicycling](https://docs.google.com/spreadsheets/d/1PXawfbKaOVKJ4PUo8qvOGHi6Xw1ldNy-Qsyer6-E1-Y)

Open those links on your phone and choose "Use the App" -- then switch to the
plan tab and zoom in for best readability.

### Will this get me banned?

Fieldplan does not touch any of the game servers, so it is perfectly within the
Terms of Service. All of the data comes from the portal list you give it, and
from the Google Directions API if you choose to use one.

# Installing

This is a console python application requiring **Python 3.12 or newer**. It
expects a POSIX-compatible system (Linux, OS X).

The easy way, using [uv](https://docs.astral.sh/uv/):

    uv sync

That's it -- uv reads `pyproject.toml`, fetches the right Python, and creates
the virtualenv for you. Prefix commands with `uv run` and you're done.

If you'd rather do it by hand:

    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt

If you've never used a console, python and pip before, then you'll have a bit
of a hard time at first, but it's not that hard to learn.

# Quick start

Put your portals in a text file, one per line, `Name; portal link`:

    Mount Royal Cross; https://intel.ingress.com/intel?pll=45.5088,-73.5878
    Chalet du Mont-Royal; https://intel.ingress.com/intel?pll=45.5041,-73.5872
    Beaver Lake Pavilion; https://intel.ingress.com/intel?pll=45.4996,-73.5951

To get a portal link, find the portal on the [Intel Map](https://intel.ingress.com/intel),
click on it, then click the "Link" button at the top right. Only the `pll=x,y`
part matters, but pasting the whole URL is easiest.

Then run:

    uv run python fieldplan.py --textfile portals.txt

Depending on how many portals you have, this will take anywhere from a few
seconds to 10-15 minutes with the default number of iterations. The plan is
written to `portals_plan.txt` next to your input file, and it looks like this:

    Start at Home Base
    [W] Waypoint: Home Base
    🚶 Move to Saint Joseph Oratory (3.0 km, 37 min)
    [P] At Saint Joseph Oratory
      [H] Ensure 5 keys here
      [S] Shields ON (6 links)
      [L] Link to Mount Royal Cross

A good number of portals is around 15, which is good for about an hour of
gameplay plus getting around.

## The text file format

- One portal per line: `Name; link`
- The link can be a full Intel URL, a bare `pll=45.5,-73.5` fragment, or just
  `45.5,-73.5` coordinates
- An optional number after a second `;` is how many keys you **already have
  in hand** for that portal, matching the maxfield `Name;URL;keys` format
  (see below)
- Lines starting with `#` are comments, blank lines are ignored
- Lines starting with `#!s`, `#!e` or `#!b` are waypoints (see below)
- Two entries at the same coordinates are the same portal in game, so the
  later one is dropped with a warning

## Keys you already have

Most of the time in a fielding run is spent hacking portals for keys and
waiting out the cooldown, so if you have already farmed some keys the plan
should know about it. Put the count after the link:

    Mount Royal Cross; https://intel.ingress.com/intel?pll=45.5088,-73.5878; 5

Those keys are a budget for the whole run, not per visit, and they are spent
against the earliest visits that need them. Fewer hacks means less cooldown
and often fewer heat sinks, so the plan may reorder as a result. On the
example portal list, marking three portals as already keyed cuts about three
minutes and drops the heat sinks needed from three to one.

Portals you have no keys for need no annotation -- the default is zero.

All three output formats count them, so a stop you can already cover says so
("keys already in hand") instead of telling you to farm keys that are in your
pocket, and a partly covered stop asks only for the difference.

*Note:* `-n`/`--nosave` has no effect in text file mode -- the plan file is
always written.

If you would rather follow the plan on your phone than off a printout, add
`-w` and read on.

# Taking the plan on your phone

Add `-w` and you also get a single HTML file you can open in your phone's
browser and follow while you walk or ride:

    uv run python fieldplan.py --textfile portals.txt -w

<img src="https://raw.githubusercontent.com/mricon/ingress-fieldplan/master/screenshots/html-plan.jpg" width="250">

It writes `portals_plan.html` next to your input file, or you can give it a
path of your own: `-w ~/Downloads/tonight.html`. Mail it to yourself, drop it
in your cloud folder of choice, or plug the phone in and copy it across.

One stop per card. Swipe sideways to move between them, with the next card
peeking in at the edge so you can always see what is coming. Tap an action to
tick it off; the card outlines itself when everything at that stop is done and
the bar at the bottom turns into a "go to" button for the next one. The strip
above the buttons always names the next stop and what waits there, so you know
whether to expect a hack, a link or a blocker before you get there.

The rest of it:

- **Navigate** hands the portal to Google Maps for turn-by-turn directions, in
  the travel mode you generated the plan with
- each link shows how far away its target is and roughly which way to face,
  which helps when you are picking it out of a crowded scanner
- **All stops** opens the whole run: a map of the route with the fields filling
  in as you complete them, the totals, and every stop with its progress. Tap
  any of them to jump there
- the header counts down the distance and time still to travel
- the cup icon asks the browser to keep the screen awake, so it stops locking
  every thirty seconds while you walk. Not every browser offers this, and the
  button is hidden when yours does not
- it follows your phone's light or dark setting; the moon icon overrides it

**No account, no network.** Everything is in the one file -- no fonts, scripts
or styles are fetched from anywhere, so the plan works in a park with no
signal. Your progress is saved in that browser's local storage, keyed to the
plan, so closing the tab and coming back resumes where you were, and a plan you
generate later starts clean instead of inheriting old ticks. Nothing is sent
anywhere, which also means progress does not follow you to another phone, and
clearing site data clears it. "Clear all progress" at the bottom of **All
stops** resets it deliberately.

One thing it does differently from the spreadsheet: it lists shields after the
links you make from that portal rather than before, which is the order you
actually play.

# Using Google Spreadsheets instead

Spreadsheets are more setup, but worth it:

1. they are easy to edit to input portals
2. they come preinstalled on all android phones
3. the generated plans are easy to tweak and reorder manually if you find an improvement
4. it's easy to mark on which step of the plan you are

If it is only the last two you are after, `-w` gets you those without a Google
account -- see above. Spreadsheet mode accepts `-w` too, if you like entering
portals in a sheet but would rather follow the plan in a browser.

The main reason why I hacked on maxfield is to make it more convenient for
biking, as having a simple plan to follow allows me to concentrate more on
biking and less on figuring out what to do next.

Set up a sheet with portal names in column `A` and Intel Map portal links in
column `B`. Blank rows are ignored. Then pass the spreadsheet URL:

    uv run python fieldplan.py -s https://docs.google.com/spreadsheets/d/xxx/edit#gid=0

The results are saved as a new sheet in that same spreadsheet, and cached on
your computer, so if you run the same spreadsheet again it will continue from
the previous best plan.

Use `-n` to calculate a plan without writing anything back to the sheet.

## Obtaining Google Spreadsheets credentials

To use spreadsheet mode you need a credentials.json file and a token. It's a
bit annoying and complicated, but you only have to do this once.

1. follow instructions on the [Python Quickstart page](https://developers.google.com/sheets/api/quickstart/python)
2. create the OAuth client as an application type of **Desktop app**
3. once you have credentials.json, save that file in the same directory as fieldplan
4. run `uv run python obtainGSToken.py`
5. a browser window will open -- allow access
6. the access token will be saved in the cache folder

*CAUTION:* this uses the long-deprecated `oauth2client` library. The
`--noauth_local_webserver` option, where you copy a link and paste a code back
into the terminal, relied on Google's "out-of-band" OAuth flow, which Google
switched off in October 2022 and which will no longer work. You need the
default flow, which spins up a local web server on `localhost:8080` and
requires a browser on the same machine.

If spreadsheet auth gives you grief, just use `--textfile` -- it needs none of
this.

# Obtaining Google Directions API key

This is optional, and also annoying and complicated, and you also have to only
do this once.

CAUTION: Google will require you to set up billing for your project, so if
you're not in a position to put in a credit card, then you shouldn't bother
with this. You should not get charged unless you're making many thousands of
API calls daily. Using your Directions API key with ingress-fieldplan should
be effectively free for you if you're not calculating hundreds of plans every
hour. Fieldplan also relies heavily on caching, so if we've looked up the
walking/biking/driving distance between two portals once, it will be stored in
local cache for all future lookups.

If you do set up your Google Directions API key, then you will greatly benefit
from much more accurate distances, especially for portals that are in close
proximity but require long detours. Without it, fieldplan falls back to
straight-line distances and a fixed speed per travel mode.

1. go to the [Instructions page](https://developers.google.com/maps/documentation/directions/get-api-key)
2. click "Get Started" and go through the process
3. run the fieldplan command with the `-g {yourkey}` switch once
4. fieldplan will cache the key and use it automatically next time

# Other commandline switches

Look at the output of

    uv run python fieldplan.py --help

to find all the knobs and levers you can tweak. Here are a few pointers:

## How many iterations to use?

Since 2026-09-07 you do not have to choose: leaving `-i` off scales the
iteration count to the size of your portal list, and the run prints what it
picked. The rest of this section is for when you want to override it.

The floor is 5,000 random iterations, and small lists get more (up to 20,000)
because iterations are cheap when there are few portals. The
bisecting and fielding is done randomly, largely because finding efficient
movement plans for a set of geographical coordinates is one of those "NP hard"
problems (look up the "Travelling Salesman Problem"). There is an optimization
step after each random plan to fix the worst inefficiencies, so the results
after each iteration are already tweaked.

Measured over 30 burn-in runs on random portal lists (2026-09-07), as a
percentage of the best a very long run finds:

| iterations | 10 portals | 15 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| 500 | 97% | 90% | 88% | 94% | 96% |
| 1,000 | 97% | 91% | 90% | 98% | 100% |
| 5,000 | 99% | 99% | 98% | 100% | 100% |

The perhaps surprising part is that **bigger lists need fewer iterations**,
not more. Each iteration does more optimizing work when there are more
portals, so the search settles sooner: the last improvement arrived at
iteration 14,339 for 10 portals but only 2,132 for 40. What grows with the
list is the cost of a single iteration, which is why the automatic default
gives small lists many more of them and large lists the 5,000 floor.

So there is little point pushing a 30- or 40-portal list past 5,000, while a
10-portal list genuinely keeps improving out to 15,000 or so -- and can
afford to, since those iterations are around 45x cheaper.

Since iterations are largely random, it's entirely possible to find the best
possible plan on your first run, and to only find terrible plans even after
10,000 iterations. YMMV.

You can hit Ctrl-C at any point to stop early and use the best plan found so far.

The route used to capture all portals before linking starts is solved as a
travelling-salesman problem. By default the solver spends 200 ms of local
search improving each route on top of the quick greedy one, which typically
shortens the capture walk by a few percent. Routes are cached, so this costs a
few seconds per run in total. Pass `--capture-search-ms 0` to skip it, or a
larger value to search harder on big portal sets. In `--maxtime` mode routes
are solved far more often, so the budget is additionally capped at a few
milliseconds per portal in the subset, which is where the gains level off.

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

You can also tune `--keys-per-hack` if 1.5 doesn't match your glyphing, and
`--cool-if-longer-than` to change when a Heat Sink is considered worth using.

## Start and End waypoints

You will probably be planning your field ops either from home, on the way
from home to work/school, or from a parking/transit stop location. To generate
plans that are more efficient with those locations, you should add
them as waypoints to your portal list.

- First, find the waypoint location on the intel map and zoom in as far in
  as possible for the most accurate result.
- With your waypoint at the center of the map, click "Link" and copy the
  URL in the pop-up box (just like with portal URLs).
- Use special name indicators at the start of the portal names:

  - `#!s Location Name` for your start waypoint
  - `#!e Location Name` for your end waypoint

In a text file:

    #!s My Home; https://intel.ingress.com/intel?ll=45.498803,-73.598872&z=21
    #!e My School; https://intel.ingress.com/intel?ll=45.504427,-73.574309&z=21

In a spreadsheet, put the `#!s My Home` part in column `A` and the URL in
column `B`.

Waypoints can be either at the start or at the end of the portal list.

## Blocker waypoints

You can also add portals you need to visit to destroy blockers by using the
same logic as with start/end waypoints. Use the `#!b Portal Name` indicator
to mark that a portal is a blocker and not part of the fielding plan.

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

Passing the `-p somedir` switch will generate a set of step-by-step PNG files
that allow you to preview the plan in action, plus an animated
`plan_movie.gif` in the same directory. Here's what it is for the Biking
example above:

![Plot example](https://raw.githubusercontent.com/mricon/ingress-fieldplan/master/screenshots/plotting.gif "Plot example")

You may need to install python-tkinter for it to work. GIF file size
optimization additionally requires the `gifsicle` binary; without it you just
get a warning and a slightly larger GIF.

Use `--plotdpi 144` if you're on a high-dpi screen.

## Exporting the plan to IITC

Passing `-j plan.json` writes the resulting fields as IITC DrawTools JSON,
which you can paste into the DrawTools plugin to see the plan on the map.

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

- Create the portal list with all 25 portals
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

- [IITC](https://iitc.app/), obviously (IITC-CE is the maintained version these days)
- [Multi-Export Plugin](https://github.com/modkin/Ingress-IITC-Multi-Export/raw/master/multi_export.user.js)

Here's how to use it:

- Draw a polygon around the portals you are interested in
- Click on "Multi-Export"
- Click on `XXX` in the "Polygon/TSV" column
- Copy all entries in the text area

For a text file, keep the portal name and the Intel URL and separate them with
a `;` -- anything after that is ignored, so the maxfield export format works
directly.

For a spreadsheet, paste in the `A1` cell. Fieldplan needs portal names in
column `A` and Intel URLs in column `B`, so you will need to either delete
columns `D`, `C`, `A`, or rearrange the columns to be in the expected order.
Fieldplan will ignore anything not in columns `A` and `B`.

*CAUTION: IITC is not an official resource, and your use of it
[may be against the Terms of Service](https://iitc.me/faq/#ban).*

# Where things are cached

Everything persistent lives in `~/.cache/ingress-fieldmap/`:

- `token.json` -- your Google Spreadsheets auth token
- `distcache` -- cached Google Directions results and your Maps API key
- `plans/` -- the best plan found so far for a given set of portals, so
  re-running the same list picks up where it left off

Pass `--no-plan-cache` to ignore and not update the stored best plan.

*Note:* these are `shelve` databases, and the on-disk format depends on which
Python built them. If you switch interpreters (say between a system Python 3.14
and uv's 3.12) you may get `dbm.error: db type could not be determined`. Delete
the cache directory and carry on.

# If something is not working

You can open a GitHub issue if something is not working for you, but please
keep in mind that this is entirely a hobby project and I may not have a chance
to help you out.

Happy fielding!

ENL agent: mricon

## Hacking on the solver

`tests/test_golden.py` runs a fixed number of seeded solver iterations
against the portal lists in `tests/fixtures/` and compares every resulting
workplan and its stats against the recorded JSON in `tests/golden/`:

    uv run python tests/test_golden.py

Any change to the algorithm that alters results will fail it. If the change
is intentional, review the difference and re-record with `--record`.
