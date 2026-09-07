import logging
from urllib.parse import urlparse, parse_qs
from lib import plansteps

logger = logging.getLogger('fieldplan')

def _get_qp_from_url(url, qp='pll'):
    # Basic cleanup
    url = url.strip()
    
    # If it's a simple coordinate pair "lat,lon"
    if ',' in url and 'ingress.com' not in url and 'http' not in url:
        try:
            parts = url.split(',')
            float(parts[0])
            float(parts[1])
            return url
        except ValueError:
            pass

    try:
        p_url = urlparse(url)
        q_parts = parse_qs(p_url.query)
        if qp in q_parts:
            return q_parts[qp][0]
    except Exception:
        pass
        
    # Fallback for simpler string manipulation if parse_qs fails or URL is malformed
    if qp + '=' in url:
        try:
            return url.split(qp + '=')[1].split('&')[0]
        except IndexError:
            pass

    return None

def get_portals_from_file(filename):
    portals = []
    waypoints = []
    
    logger.info(f"Reading portals from {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    startpoint_loc = None
    endpoint_loc = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for waypoint markers (mimicking gsheets logic)
        # Format: #!s Name ; URL
        if line.startswith('#!'):
            parts = line.split(';')
            marker_part = parts[0].strip()
            
            # marker_part looks like "#!s Name"
            if len(marker_part) < 4:
                continue
                
            w_type = marker_part[2] # 's', 'e', 'b'
            name = marker_part[3:].strip()
            
            url = None
            for part in parts[1:]:
                if 'll=' in part or 'pll=' in part or ',' in part:
                     url = part.strip()
                     break
            
            if not url:
                continue

            coords = _get_qp_from_url(url, qp='ll')
            if not coords:
                 coords = _get_qp_from_url(url, qp='pll')
            
            if not coords:
                 continue

            if w_type == 's':
                if startpoint_loc is not None:
                     logger.critical('Multiple start waypoints found!')
                waypoints.append((name, coords, '_w_start'))
                logger.info('Adding start waypoint: %s', name)
                startpoint_loc = len(waypoints) - 1
            elif w_type == 'e':
                if endpoint_loc is not None:
                     logger.critical('Multiple end waypoints found!')
                waypoints.append((name, coords, '_w_end'))
                logger.info('Adding end waypoint: %s', name)
                endpoint_loc = len(waypoints) - 1
            elif w_type == 'b':
                waypoints.append((name, coords, '_w_blocker'))
                logger.info('Adding blocker waypoint: %s', name)
            
            continue

        if line.startswith('#'):
            continue
            
        # Normal portal
        # Format: Name ; URL ; ...
        parts = line.split(';')
        name = parts[0].strip()
        
        url = None
        # Try to find the URL part. It's usually the second part, but let's be flexible
        # The maxfield format is Name;URL;Keys
        if len(parts) > 1:
            potential_url = parts[1].strip()
            if 'http' in potential_url or 'pll=' in potential_url or ',' in potential_url:
                url = potential_url
        
        if not url:
            # Maybe it's in other parts?
            for part in parts[1:]:
                 if 'pll=' in part or 'http' in part:
                     url = part.strip()
                     break
        
        if not url:
            logger.debug(f"Skipping line, no URL found: {line}")
            continue
            
        coords = _get_qp_from_url(url, qp='pll')
        if not coords:
            logger.warning(f"Could not extract coordinates from URL for portal {name}")
            continue

        # Any remaining bare number is how many keys we already hold for
        # this portal (maxfield uses the same convention)
        keys = 0
        for part in parts[1:]:
            part = part.strip()
            if part == url or not part:
                continue
            try:
                keys = int(part)
            except ValueError:
                logger.warning(f"Ignoring unrecognized field '{part}' for portal {name}")
                continue
            if keys < 0:
                logger.warning(f"Ignoring negative key count for portal {name}")
                keys = 0
            break

        if keys:
            logger.info('Adding portal: %s (%d keys in hand)', name, keys)
        else:
            logger.info('Adding portal: %s', name)
        portals.append((name, coords, None, keys))

    # make sure end waypoint is always last in the waypoint list
    if endpoint_loc is not None and endpoint_loc != len(waypoints)-1:
        _ep = waypoints.pop(endpoint_loc)
        waypoints.append(_ep)
            
    return portals, waypoints

def write_workplan(filename, a, workplan, stats, faction, travelmode='walking'):
    import os

    base, ext = os.path.splitext(filename)
    outfile = f"{base}_plan.txt"
    
    logger.info(f"Writing workplan to {outfile}")
    
    with open(outfile, 'w', encoding='utf-8') as f:
        # Header
        totalkm = stats['dist']/float(1000)
        f.write(f"Plan for: {filename}\n")
        f.write(f"Total distance: {totalkm:.2f} km\n")
        f.write(f"Total time: {stats['nicetime']} (Travel: {stats['nicetraveltime']} {travelmode})\n")
        f.write(f"Total AP: {stats['ap']:,}\n")
        f.write("-" * 40 + "\n\n")

        travelmoji = {
            'walking': u"\U0001F6B6",
            'bicycling': u"\U0001F6B2",
            'transit': u"\U0001F68D",
            'driving': u"\U0001F697",
        }

        for stop in plansteps.build_stops(a, workplan, travelmode):
            travel = stop['travel']
            if travel is None:
                f.write(f"Start at {stop['name']}\n")
            elif travel['moved']:
                f.write(f"{travelmoji.get(travelmode, '')} Move to {stop['name']} "
                        f"({travel['nicedist']}, {travel['time']} min)\n")
            else:
                f.write(f"\u25bc Move to {stop['name']}\n")

            if stop['is_waypoint']:
                f.write(f"[W] Waypoint: {stop['name']}\n")
                continue

            if stop['is_blocker']:
                f.write(f"[X] DESTROY BLOCKER at {stop['name']}\n")
                continue

            f.write(f"[P] At {stop['name']}\n")

            keys = stop['keys']
            if keys:
                # 'hack' nets off the keys already in hand; 'total' is the
                # raw number of links into this portal still to be made
                if not keys['want']:
                    f.write(f"  [H] Need {keys['total']} max keys later\n")
                elif not keys['hack']:
                    held = plansteps.plural(keys['want'], 'key')
                    f.write(f"  [H] Already carrying the {held} needed here\n")
                elif keys['lastvisit']:
                    f.write(f"  [H] Ensure {keys['hack']} keys here\n")
                else:
                    f.write(f"  [H] Ensure {keys['hack']} keys here "
                            f"(will need {keys['total']} total)\n")

            if stop['shields']:
                f.write(f"  [S] Shields ON ({stop['shields']['links']} links)\n")

            for link in stop['links']:
                f.write(f"  [{link['action']}] Link to {link['name']}\n")

    logger.info("Text plan generation done.")

