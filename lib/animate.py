#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os

from matplotlib.patches import Polygon

import logging
logger = logging.getLogger('maxfield3')

from lib import maxfield

def shrink(a):
    centroid = a.mean(1).reshape([2,1])
    return  centroid + .9*(a-centroid)


def draw_edge(a, s, t, fig, marker, directional=False):
    eart = list()
    edge = np.array([a.node[s]['xy'], a.node[t]['xy']]).T
    eart += fig.plot(edge[0], edge[1], marker, lw=2)

    if directional:
        # Imitate an arrow to show which way direction is going
        x0 = edge[0][0]
        x1 = edge[0][1]
        y0 = edge[1][0]
        y1 = edge[1][1]

        eart += fig.plot([x1-0.05*(x1-x0), x1-0.4*(x1-x0)],
                         [y1-0.05*(y1-y0), y1-0.4*(y1-y0)], marker, lw=6)

    return eart


def make_png_steps(a, workplan, outdir):
    logger.info('Generating step-by-step pngs of the workplan')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    GREEN     = ( 0.0 , 1.0 , 0.0 , 0.3)
    BLUE      = ( 0.0 , 0.0 , 1.0 , 0.3)
    RED       = ( 1.0 , 0.0 , 0.0 , 0.5)
    INVISIBLE = ( 0.0 , 0.0 , 0.0 , 0.0 )

    portals = np.array([a.node[i]['xy'] for i in a.nodes()]).T

    fig = plt.figure(figsize=(11, 8), dpi=80)
    frames = list()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    ax.plot(portals[0], portals[1], 'ko')
    ax.set_title('Portals before capture', ha='center')

    filen = os.path.join(outdir, 'step_{0:03d}.png'.format(len(frames)))
    fig.savefig(filen)
    frames.append(filen)

    prev_p = None
    seen_p = list()
    for p, q, f in workplan:
        if p != prev_p:
            torm = list()
            # Colour this node green
            p_coords = np.array(a.node[p]['xy']).T
            ax.plot(p_coords[0], p_coords[1], 'go')
            if prev_p is not None:
                # Show travel edge
                dist = maxfield.getPortalDistance(a, prev_p, p)
                torm = draw_edge(a, prev_p, p, ax, 'm:', directional=True)
                if 'blocker' in a.node[p]:
                    action = 'Destroy blocker'
                else:
                    action = 'Capture'

                if p not in seen_p:
                    if dist > 80:
                        ax.set_title('%s %s (%s m)' % (action, a.node[p]['name'], dist), ha='center')
                    else:
                        ax.set_title('%s %s' % (action, a.node[p]['name']), ha='center')
                else:
                    if dist > 80:
                        ax.set_title('Travel to %s (%s m)' % (a.node[p]['name'], dist), ha='center')
                    else:
                        ax.set_title('Move to %s' % a.node[p]['name'], ha='center')
            else:
                ax.set_title('Start at %s' % a.node[p]['name'], ha='center')

            filen = os.path.join(outdir, 'step_{0:03d}.png'.format(len(frames)))
            fig.savefig(filen)
            frames.append(filen)
            for art in torm:
                art.remove()

        prev_p = p
        if p not in seen_p:
            seen_p.append(p)

        if q is None:
            continue

        ax.set_title('Link to %s' % a.node[q]['name'], ha='center')

        # Draw the link edge
        torm = draw_edge(a, p, q, ax, 'k-', directional=True)

        if f:
            # We'll display the new fields in red
            fields = list()
            for tri in a.edges[p, q]['fields']:
                coords = np.array([a.node[v]['xy'] for v in tri])
                fields.append(Polygon(shrink(coords.T).T, facecolor=RED,
                                      edgecolor=INVISIBLE))

            for field in fields:
                torm.append(ax.add_patch(field))

        filen = os.path.join(outdir, 'step_{0:03d}.png'.format(len(frames)))
        fig.savefig(filen)
        frames.append(filen)

        # remove the newly added edges and triangles from the graph
        for art in torm:
            art.remove()
        # redraw the new edges and fields in the final colour
        draw_edge(a, p, q, ax, 'g-')
        if f:
            # reset fields to green
            for field in fields:
                field.set_facecolor(GREEN)
                ax.add_patch(field)

    # Save final frame with all completed fields
    ax.set_title('Finish at %s' % a.node[p]['name'], ha='center')
    filen = os.path.join(outdir, 'step_{0:03d}.png'.format(len(frames)))
    fig.savefig(filen)
    frames.append(filen)

    logger.info('Saved step-by-step pngs into %s', outdir)

