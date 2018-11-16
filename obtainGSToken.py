#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import argparse

from oauth2client import file, client, tools

from pathlib import Path

SCOPES = 'https://www.googleapis.com/auth/spreadsheets'


def main():
    description = (
            'Obtain a Google Spreadsheets authorization token using credentials.json.'
            'To obtain the credentials.json file, follow instructions on this page:'
            'https://developers.google.com/sheets/api/quickstart/python'
            'Save credentials.json in the same directory with this script.'
            )

    parser = argparse.ArgumentParser(
                description=description,
                formatter_class=argparse.RawDescriptionHelpFormatter,
                parents=[tools.argparser])
    flags = parser.parse_args()

    home = str(Path.home())
    cachedir = os.path.join(home, '.cache', 'ingress-fieldmap')
    Path(cachedir).mkdir(parents=True, exist_ok=True)
    tokenfile = os.path.join(cachedir, 'token.json')

    store = file.Storage(tokenfile)
    flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
    creds = tools.run_flow(flow, store, flags)

    if creds:
        print('Token saved in %s' % tokenfile)


if __name__ == '__main__':
    main()
