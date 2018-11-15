# Introduction

This is for Ingress. If you don't know what that is, you're lost.

This is a heavily modified original maxfield solution that will generate an
easy-to-follow fielding plans using Google Spreadsheets. The benefits over the
original maxfield program are:

1. Ported to Python 3
2. Generates more efficient solutions requiring fewer iterations
3. Generates an efficient capture plan in addition to the fielding plan
4. Uses Google Directions API for precise distances
5. Supports walking, biking, and driving plans

# Prerequisites

This is a console python application. It expects a POSIX-compatible system
(Linux, OS X) and a virtualenv-3 setup. After initializing the environment,
you can install the required libraries using:

    pip install -r requirements.txt

