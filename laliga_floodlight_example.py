#!/usr/bin/env python3
"""
La Liga Floodlight Example: Comprehensive Pitch Maps
====================================================

This script demonstrates what Floodlight can do by creating a series of
pitch map visualizations for various metrics from a La Liga 2023 match.

It loads OPTA F24 XML data and creates visualizations for:
- Pass maps (successful/unsuccessful)
- Key passes
- Shots and goals
- Crosses
- Tackles
- Heat maps (pass density)
- Shot locations

Author: Module 10 - Collaborative Activity
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from collections import defaultdict
import os
import glob
from datetime import datetime


def parse_opta_xml(file_path: str) -> dict:
    """
    Parse a single OPTA F24 XML file and extract all event data.
    
    Args:
        file_path: Path to XML file
        
    Returns:
        Dictionary with match info and events
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Get game info
    game = root.find('Game')
    if game is None:
        games = root.find('Games')
        if games is not None:
            game = games.find('Game')
    
    if game is None:
        return None
    
    match_id = game.get('id')
    home_team_id = game.get('home_team_id')
    away_team_id = game.get('away_team_id')
    home_team_name = game.get('home_team_name', f'Team {home_team_id}')
    away_team_name = game.get('away_team_name', f'Team {away_team_id}')
    
    # Extract match date
    game_date = game.get('game_date', '')
    match_date = ''
    if game_date:
        # Parse date from format like "2022-08-15T20:30:00" to "2022-08-15"
        try:
            from datetime import datetime
            date_obj = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
            match_date = date_obj.strftime('%Y-%m-%d')
        except:
            # Fallback: try to extract just the date part
            if 'T' in game_date:
                match_date = game_date.split('T')[0]
            else:
                match_date = game_date[:10] if len(game_date) >= 10 else ''
    
    # Track scores
    home_score = 0
    away_score = 0
    
    # Extract all events
    events = []
    for event in game.findall('Event'):
        event_type = int(event.get('type_id', 0))
        
        # Track goals
        if event_type == 16:  # Goal
            team_id = event.get('team_id')
            if team_id == home_team_id:
                home_score += 1
            else:
                away_score += 1
        
        # Extract event data
        x = float(event.get('x', 0)) if event.get('x') else None
        y = float(event.get('y', 0)) if event.get('y') else None
        outcome = int(event.get('outcome', 0)) if event.get('outcome') else None
        player_id = event.get('player_id')
        team_id = event.get('team_id')
        minute = int(event.get('min', 0))
        period = int(event.get('period_id', 0))
        
        # Parse qualifiers
        qualifiers = {}
        is_cross = False
        is_through_ball = False
        is_assist = False
        is_long_ball = False
        is_progressive = False
        end_x = None  # Pass end coordinates (qualifiers 140, 141)
        end_y = None
        
        for qualifier in event.findall('Q'):
            qual_id = int(qualifier.get('qualifier_id', 0))
            value = qualifier.get('value')
            qualifiers[qual_id] = value
            
            if qual_id == 2:  # Cross
                is_cross = True
            elif qual_id == 4:  # Through Ball
                is_through_ball = True
            elif qual_id == 210:  # Assist
                is_assist = True
            elif qual_id == 1:  # Long ball
                is_long_ball = True
            elif qual_id == 106:  # Attacking Pass (progressive)
                is_progressive = True
            elif qual_id == 140:  # End X coordinate
                try:
                    end_x = float(value)
                except (ValueError, TypeError):
                    end_x = None
            elif qual_id == 141:  # End Y coordinate
                try:
                    end_y = float(value)
                except (ValueError, TypeError):
                    end_y = None
        
        # Calculate total seconds for easier time-based matching
        second = int(event.get('sec', 0))
        total_seconds = minute * 60 + second
        
        events.append({
            'type_id': event_type,
            'x': x,
            'y': y,
            'end_x': end_x,  # Pass end coordinates if available
            'end_y': end_y,
            'outcome': outcome,
            'player_id': player_id,
            'team_id': team_id,
            'minute': minute,
            'second': second,
            'total_seconds': total_seconds,
            'period': period,
            'qualifiers': qualifiers,
            'is_key_pass': False,  # Will be determined below
            'is_cross': is_cross,
            'is_through_ball': is_through_ball,
            'is_assist': is_assist,
            'is_long_ball': is_long_ball,
            'is_progressive': is_progressive,
            'event_id': event.get('id')  # Store event ID for reference
        })
    
    # Now identify key passes according to OPTA's definition:
    # "The final pass from a player to their teammate who then makes an attempt 
    # on Goal without scoring."
    # This means: pass (type_id=1) followed by SHOT (type_id=13), NOT goal (type_id=16)
    # If a pass leads to a goal, it's an ASSIST, not a key pass!
    
    # Approach: For each SHOT (not goal), find the final pass by the same team before it
    for i, event in enumerate(events):
        if event['type_id'] == 13:  # It's a SHOT (not a goal)
            shot_team = event['team_id']
            shot_period = event['period']
            shot_time = event['total_seconds']
            
            # Look backwards for the FINAL pass by the same team before this shot
            for j in range(i - 1, -1, -1):  # Go backwards from the shot
                prev_event = events[j]
                
                # Must be same team, same period
                if prev_event['team_id'] != shot_team or prev_event['period'] != shot_period:
                    continue
                
                # Must be a pass
                if prev_event['type_id'] == 1:
                    time_diff = shot_time - prev_event['total_seconds']
                    
                    # Pass must occur within 10 seconds before the shot
                    if 0 < time_diff <= 10:
                        # This is the final pass before the shot = KEY PASS
                        prev_event['is_key_pass'] = True
                        break  # Found the final pass, stop looking
    
    return {
        'match_id': match_id,
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_team_name': home_team_name,
        'away_team_name': away_team_name,
        'home_score': home_score,
        'away_score': away_score,
        'match_date': match_date,
        'events': events
    }


def draw_pitch(ax):
    """Draw a professional football pitch on the axes."""
    # Professional pitch colors
    pitch_green = '#3a5f3a'  # Darker, more realistic green
    line_color = '#ffffff'    # Pure white for visibility
    goal_color = '#1a1a1a'   # Dark for contrast
    
    # Pitch background with gradient effect (darker edges)
    pitch_rect = mpatches.Rectangle((0, 0), 100, 100, 
                                     linewidth=3, edgecolor='#1a1a1a', 
                                     facecolor=pitch_green, alpha=0.85)
    ax.add_patch(pitch_rect)
    
    # Add subtle grid pattern for grass texture
    for i in range(0, 101, 5):
        ax.plot([i, i], [0, 100], color=pitch_green, linewidth=0.3, alpha=0.2)
        ax.plot([0, 100], [i, i], color=pitch_green, linewidth=0.3, alpha=0.2)
    
    # Center line - thicker and more visible
    ax.plot([50, 50], [0, 100], color=line_color, linewidth=2.5, alpha=0.9, linestyle='-')
    
    # Center circle - more prominent
    center_circle = plt.Circle((50, 50), 9.15, fill=False, 
                               edgecolor=line_color, linewidth=2, alpha=0.9)
    ax.add_patch(center_circle)
    # Center dot
    center_dot = plt.Circle((50, 50), 0.5, fill=True, 
                           facecolor=line_color, edgecolor=line_color, alpha=0.9)
    ax.add_patch(center_dot)
    
    # Penalty areas - more defined
    # Defensive penalty area
    def_penalty = mpatches.Rectangle((0, 20.16), 16.5, 59.68, 
                                      fill=False, edgecolor=line_color, 
                                      linewidth=2, alpha=0.9)
    ax.add_patch(def_penalty)
    # Penalty arc
    penalty_arc_left = mpatches.Arc((16.5, 50), 9.15*2, 9.15*2, 
                                    angle=0, theta1=270, theta2=90, 
                                    color=line_color, linewidth=2, alpha=0.9)
    ax.add_patch(penalty_arc_left)
    # Penalty spot
    penalty_spot_left = plt.Circle((11, 50), 0.5, fill=True, 
                                   facecolor=line_color, edgecolor=line_color, alpha=0.9)
    ax.add_patch(penalty_spot_left)
    
    # Attacking penalty area
    att_penalty = mpatches.Rectangle((83.5, 20.16), 16.5, 59.68, 
                                      fill=False, edgecolor=line_color, 
                                      linewidth=2, alpha=0.9)
    ax.add_patch(att_penalty)
    # Penalty arc
    penalty_arc_right = mpatches.Arc((83.5, 50), 9.15*2, 9.15*2, 
                                     angle=0, theta1=90, theta2=270, 
                                     color=line_color, linewidth=2, alpha=0.9)
    ax.add_patch(penalty_arc_right)
    # Penalty spot
    penalty_spot_right = plt.Circle((89, 50), 0.5, fill=True, 
                                    facecolor=line_color, edgecolor=line_color, alpha=0.9)
    ax.add_patch(penalty_spot_right)
    
    # Goals - more prominent
    goal_left = mpatches.Rectangle((-1, 36.84), 1, 26.32, 
                                    fill=True, facecolor=goal_color, 
                                    edgecolor=line_color, linewidth=2.5, alpha=0.95)
    ax.add_patch(goal_left)
    goal_right = mpatches.Rectangle((100, 36.84), 1, 26.32, 
                                     fill=True, facecolor=goal_color, 
                                     edgecolor=line_color, linewidth=2.5, alpha=0.95)
    ax.add_patch(goal_right)
    
    # Goal areas (6-yard boxes)
    goal_area_left = mpatches.Rectangle((0, 40.32), 5.5, 19.36, 
                                         fill=False, edgecolor=line_color, 
                                         linewidth=1.5, alpha=0.8)
    ax.add_patch(goal_area_left)
    goal_area_right = mpatches.Rectangle((94.5, 40.32), 5.5, 19.36, 
                                          fill=False, edgecolor=line_color, 
                                          linewidth=1.5, alpha=0.8)
    ax.add_patch(goal_area_right)
    
    # Final third lines - subtle but visible
    ax.plot([67, 67], [0, 100], color=line_color, linewidth=1.5, 
           alpha=0.4, linestyle='--')
    ax.plot([33, 33], [0, 100], color=line_color, linewidth=1.5, 
           alpha=0.4, linestyle='--')
    
    # Corner arcs
    corner_arcs = [
        ((0, 0), 2, 0, 90),      # Bottom left
        ((0, 100), 2, 270, 360), # Top left
        ((100, 0), 2, 90, 180),  # Bottom right
        ((100, 100), 2, 180, 270) # Top right
    ]
    for (x, y), radius, theta1, theta2 in corner_arcs:
        corner_arc = mpatches.Arc((x, y), radius*2, radius*2, 
                                  angle=0, theta1=theta1, theta2=theta2, 
                                  color=line_color, linewidth=2, alpha=0.9)
        ax.add_patch(corner_arc)
    
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_aspect('equal')
    ax.axis('off')


def create_pass_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create pass map showing successful and unsuccessful passes."""
    team_events = [e for e in match_data['events'] 
                   if e['team_id'] == team_id and e['type_id'] == 1]
    
    if not team_events:
        return None
    
    successful = [e for e in team_events if e['outcome'] == 1]
    unsuccessful = [e for e in team_events if e['outcome'] == 0]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    # Professional color scheme
    success_color = '#28a745'  # Professional green
    fail_color = '#dc3545'     # Professional red
    
    # Plot unsuccessful passes - more visible
    if unsuccessful:
        x_fail = [e['x'] for e in unsuccessful if e['x'] is not None]
        y_fail = [e['y'] for e in unsuccessful if e['x'] is not None]
        ax.scatter(x_fail, y_fail, c=fail_color, s=60, alpha=0.6, 
                  marker='x', linewidths=2, label='Unsuccessful', zorder=3)
    
    # Plot successful passes - enhanced styling
    if successful:
        x_success = [e['x'] for e in successful if e['x'] is not None]
        y_success = [e['y'] for e in successful if e['x'] is not None]
        ax.scatter(x_success, y_success, c=success_color, s=70, alpha=0.7, 
                  marker='o', edgecolors='#1e7e34', linewidths=1.5, 
                  label='Successful', zorder=4)
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    
    # Enhanced title styling
    ax.set_title(f'Pass Map - {team_name}\n'
                f'Successful: {len(successful)} | Unsuccessful: {len(unsuccessful)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def get_pass_end_location(pass_event: dict):
    """
    Get the end location of a pass from OPTA F24 data.
    
    OPTA F24 stores pass end coordinates in qualifiers:
    - Qualifier 140: End X coordinate
    - Qualifier 141: End Y coordinate
    
    Args:
        pass_event: The pass event dictionary
        
    Returns:
        Tuple of (end_x, end_y) or (None, None) if not available
    """
    # First try to get from qualifiers (most accurate)
    if pass_event.get('end_x') is not None and pass_event.get('end_y') is not None:
        return pass_event['end_x'], pass_event['end_y']
    
    # Fallback: look for next event by same team (if qualifiers not available)
    # This is a backup method
    return None, None


def create_key_pass_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing key passes with arrows indicating direction."""
    key_passes = [e for e in match_data['events'] 
                  if e['team_id'] == team_id and e['is_key_pass']]
    
    if not key_passes:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    x_coords = [e['x'] for e in key_passes if e['x'] is not None]
    y_coords = [e['y'] for e in key_passes if e['x'] is not None]
    
    # Create heat map
    if x_coords:
        H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=15,
                                          range=[[0, 100], [0, 100]])
        H = H.T
        im = ax.imshow(H, extent=[0, 100, 0, 100], origin='lower',
                      cmap='hot', alpha=0.5, interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Key Pass Density')
    
    # Find end locations and draw arrows
    arrows_drawn = 0
    for key_pass in key_passes:
        if key_pass['x'] is None or key_pass['y'] is None:
            continue
        
        # Get end location from qualifiers
        end_x, end_y = get_pass_end_location(key_pass)
        
        if end_x is not None and end_y is not None:
            # Draw arrow from start to end
            dx = end_x - key_pass['x']
            dy = end_y - key_pass['y']
            
            # Only draw arrow if there's meaningful distance (at least 2 units)
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 2:
                # Calculate arrow properties
                arrow_length = distance * 0.9  # Make arrow slightly shorter to avoid overlap
                head_width = max(1.5, distance * 0.08)  # Proportional head size
                head_length = max(1.5, distance * 0.1)
                
                # Normalize direction
                dx_norm = dx / distance
                dy_norm = dy / distance
                
                # Draw arrow
                ax.arrow(key_pass['x'], key_pass['y'], 
                        dx_norm * arrow_length, dy_norm * arrow_length,
                        head_width=head_width, head_length=head_length, 
                        fc='#ffd700', ec='#1a1a1a',
                        linewidth=2.5, alpha=0.85, zorder=4, 
                        length_includes_head=True)
                arrows_drawn += 1
    
    # Overlay individual key passes (circles at start locations)
    ax.scatter(x_coords, y_coords, c='#ffd700', s=180, alpha=0.95,
              marker='o', edgecolors='#1a1a1a', linewidths=2.5, zorder=5,
              label=f'Key Passes ({arrows_drawn} with arrows)')
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Key Pass Map - {team_name}\n'
                f'Total Key Passes: {len(key_passes)} | Arrows: {arrows_drawn}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_shot_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create shot map showing shots and goals."""
    shots = [e for e in match_data['events'] 
             if e['team_id'] == team_id and e['type_id'] == 13]  # Shot
    goals = [e for e in match_data['events'] 
             if e['team_id'] == team_id and e['type_id'] == 16]  # Goal
    
    if not shots and not goals:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    # Plot shots (non-goals)
    shots_no_goal = [s for s in shots if s not in goals]
    if shots_no_goal:
        x_shots = [e['x'] for e in shots_no_goal if e['x'] is not None]
        y_shots = [e['y'] for e in shots_no_goal if e['x'] is not None]
        outcomes = [e['outcome'] for e in shots_no_goal if e['x'] is not None]
        
        # Color by outcome: 1 = on target, 0 = off target
        for i, (x, y, outcome) in enumerate(zip(x_shots, y_shots, outcomes)):
            color = 'orange' if outcome == 1 else 'red'
            marker = 'o' if outcome == 1 else 'x'
            ax.scatter(x, y, c=color, s=100, alpha=0.7, marker=marker, zorder=4)
    
    # Plot goals
    if goals:
        x_goals = [e['x'] for e in goals if e['x'] is not None]
        y_goals = [e['y'] for e in goals if e['x'] is not None]
        ax.scatter(x_goals, y_goals, c='#ffd700', s=400, alpha=0.95,
                  marker='o', edgecolors='#1a1a1a', linewidths=3, zorder=5,
                  label='Goals')
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Shot Map - {team_name}\n'
                f'Shots: {len(shots)} | Goals: {len(goals)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    # Enhanced custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffd700', 
               markersize=16, label='Goals', markeredgecolor='#1a1a1a', 
               markeredgewidth=3, linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff8c00', 
               markersize=12, label='On Target', markeredgecolor='#cc6600', 
               markeredgewidth=1.5, linestyle='None', alpha=0.8),
        Line2D([0], [0], marker='x', color='#dc3545', markersize=12, 
               label='Off Target', markeredgewidth=2.5, linestyle='None', alpha=0.8)
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                      framealpha=0.95, edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_cross_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing crosses."""
    crosses = [e for e in match_data['events'] 
               if e['team_id'] == team_id and e['is_cross']]
    
    if not crosses:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    successful_crosses = [e for e in crosses if e['outcome'] == 1]
    unsuccessful_crosses = [e for e in crosses if e['outcome'] == 0]
    
    # Professional color scheme for crosses
    cross_success_color = '#0066cc'  # Professional blue
    cross_fail_color = '#dc3545'     # Professional red
    
    # Plot unsuccessful crosses
    if unsuccessful_crosses:
        x_fail = [e['x'] for e in unsuccessful_crosses if e['x'] is not None]
        y_fail = [e['y'] for e in unsuccessful_crosses if e['x'] is not None]
        ax.scatter(x_fail, y_fail, c=cross_fail_color, s=100, alpha=0.6, 
                  marker='x', linewidths=2.5, label='Unsuccessful', zorder=3)
    
    # Plot successful crosses
    if successful_crosses:
        x_success = [e['x'] for e in successful_crosses if e['x'] is not None]
        y_success = [e['y'] for e in successful_crosses if e['x'] is not None]
        ax.scatter(x_success, y_success, c=cross_success_color, s=120, alpha=0.8, 
                  marker='o', edgecolors='#004499', linewidths=2.5,
                  label='Successful', zorder=4)
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Cross Map - {team_name}\n'
                f'Successful: {len(successful_crosses)} | Unsuccessful: {len(unsuccessful_crosses)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_tackle_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing tackles."""
    tackles = [e for e in match_data['events'] 
               if e['team_id'] == team_id and e['type_id'] == 7]  # Tackle
    
    if not tackles:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    successful_tackles = [e for e in tackles if e['outcome'] == 1]
    unsuccessful_tackles = [e for e in tackles if e['outcome'] == 0]
    
    # Professional color scheme for tackles
    tackle_success_color = '#155724'  # Dark green
    tackle_fail_color = '#dc3545'    # Red
    
    # Plot unsuccessful tackles
    if unsuccessful_tackles:
        x_fail = [e['x'] for e in unsuccessful_tackles if e['x'] is not None]
        y_fail = [e['y'] for e in unsuccessful_tackles if e['x'] is not None]
        ax.scatter(x_fail, y_fail, c=tackle_fail_color, s=100, alpha=0.6, 
                  marker='x', linewidths=2.5, label='Unsuccessful', zorder=3)
    
    # Plot successful tackles
    if successful_tackles:
        x_success = [e['x'] for e in successful_tackles if e['x'] is not None]
        y_success = [e['y'] for e in successful_tackles if e['x'] is not None]
        ax.scatter(x_success, y_success, c=tackle_success_color, s=120, alpha=0.8, 
                  marker='s', edgecolors='#0d3e1a', linewidths=2,
                  label='Successful', zorder=4)
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Tackle Map - {team_name}\n'
                f'Successful: {len(successful_tackles)} | Unsuccessful: {len(unsuccessful_tackles)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_pass_density_heatmap(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create heat map showing pass density."""
    passes = [e for e in match_data['events'] 
              if e['team_id'] == team_id and e['type_id'] == 1]
    
    if not passes:
        return None
    
    successful_passes = [e for e in passes if e['outcome'] == 1]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    if successful_passes:
        x_coords = [e['x'] for e in successful_passes if e['x'] is not None]
        y_coords = [e['y'] for e in successful_passes if e['x'] is not None]
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=20,
                                          range=[[0, 100], [0, 100]])
        H = H.T
        
        # Plot heat map
        im = ax.imshow(H, extent=[0, 100, 0, 100], origin='lower',
                      cmap='YlOrRd', alpha=0.7, interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Pass Density')
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Pass Density Heat Map - {team_name}\n'
                f'Total Successful Passes: {len(successful_passes)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_combined_metrics_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create a combined map showing multiple metrics."""
    passes = [e for e in match_data['events'] 
              if e['team_id'] == team_id and e['type_id'] == 1]
    shots = [e for e in match_data['events'] 
             if e['team_id'] == team_id and e['type_id'] == 13]
    goals = [e for e in match_data['events'] 
             if e['team_id'] == team_id and e['type_id'] == 16]
    key_passes = [e for e in match_data['events'] 
                  if e['team_id'] == team_id and e['is_key_pass']]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    # Professional color scheme for combined map
    pass_color = '#90ee90'      # Light green
    keypass_color = '#0066cc'  # Blue
    shot_color = '#ff8c00'     # Orange
    goal_color = '#ffd700'    # Gold
    
    # Plot passes (light green, small)
    if passes:
        successful_passes = [e for e in passes if e['outcome'] == 1]
        if successful_passes:
            x_passes = [e['x'] for e in successful_passes if e['x'] is not None]
            y_passes = [e['y'] for e in successful_passes if e['x'] is not None]
            ax.scatter(x_passes, y_passes, c=pass_color, s=25, alpha=0.4, 
                      marker='o', edgecolors='#5cb85c', linewidths=0.5,
                      zorder=1, label='Passes')
    
    # Plot key passes (blue)
    if key_passes:
        x_kp = [e['x'] for e in key_passes if e['x'] is not None]
        y_kp = [e['y'] for e in key_passes if e['x'] is not None]
        ax.scatter(x_kp, y_kp, c=keypass_color, s=140, alpha=0.8, 
                  marker='o', edgecolors='#004499', linewidths=2,
                  zorder=3, label='Key Passes')
    
    # Plot shots (orange)
    if shots:
        shots_no_goal = [s for s in shots if s not in goals]
        if shots_no_goal:
            x_shots = [e['x'] for e in shots_no_goal if e['x'] is not None]
            y_shots = [e['y'] for e in shots_no_goal if e['x'] is not None]
            ax.scatter(x_shots, y_shots, c=shot_color, s=180, alpha=0.85, 
                      marker='o', edgecolors='#cc6600', linewidths=2.5,
                      zorder=4, label='Shots')
    
    # Plot goals (gold circles)
    if goals:
        x_goals = [e['x'] for e in goals if e['x'] is not None]
        y_goals = [e['y'] for e in goals if e['x'] is not None]
        ax.scatter(x_goals, y_goals, c=goal_color, s=450, alpha=0.95,
                  marker='o', edgecolors='#1a1a1a', linewidths=3,
                  zorder=5, label='Goals')
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Combined Metrics Map - {team_name}\n'
                f'Passes: {len(passes)} | Key Passes: {len(key_passes)} | '
                f'Shots: {len(shots)} | Goals: {len(goals)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_interception_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing interceptions."""
    interceptions = [e for e in match_data['events'] 
                     if e['team_id'] == team_id and e['type_id'] == 8]  # Interception
    
    if not interceptions:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    successful_interceptions = [e for e in interceptions if e['outcome'] == 1]
    unsuccessful_interceptions = [e for e in interceptions if e['outcome'] == 0]
    
    # Professional color scheme
    success_color = '#28a745'  # Green
    fail_color = '#dc3545'     # Red
    
    # Plot unsuccessful interceptions
    if unsuccessful_interceptions:
        x_fail = [e['x'] for e in unsuccessful_interceptions if e['x'] is not None]
        y_fail = [e['y'] for e in unsuccessful_interceptions if e['x'] is not None]
        ax.scatter(x_fail, y_fail, c=fail_color, s=80, alpha=0.6, 
                  marker='x', linewidths=2, label='Unsuccessful', zorder=3)
    
    # Plot successful interceptions
    if successful_interceptions:
        x_success = [e['x'] for e in successful_interceptions if e['x'] is not None]
        y_success = [e['y'] for e in successful_interceptions if e['x'] is not None]
        ax.scatter(x_success, y_success, c=success_color, s=100, alpha=0.8, 
                  marker='o', edgecolors='#1e7e34', linewidths=2, 
                  label='Successful', zorder=4)
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Interception Map - {team_name}\n'
                f'Successful: {len(successful_interceptions)} | Unsuccessful: {len(unsuccessful_interceptions)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_takeon_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing take-ons/dribbles."""
    takeons = [e for e in match_data['events'] 
               if e['team_id'] == team_id and e['type_id'] == 3]  # Take On
    
    if not takeons:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    successful_takeons = [e for e in takeons if e['outcome'] == 1]
    unsuccessful_takeons = [e for e in takeons if e['outcome'] == 0]
    
    # Professional color scheme
    success_color = '#9b59b6'  # Purple
    fail_color = '#dc3545'     # Red
    
    # Plot unsuccessful take-ons
    if unsuccessful_takeons:
        x_fail = [e['x'] for e in unsuccessful_takeons if e['x'] is not None]
        y_fail = [e['y'] for e in unsuccessful_takeons if e['x'] is not None]
        ax.scatter(x_fail, y_fail, c=fail_color, s=80, alpha=0.6, 
                  marker='x', linewidths=2, label='Unsuccessful', zorder=3)
    
    # Plot successful take-ons
    if successful_takeons:
        x_success = [e['x'] for e in successful_takeons if e['x'] is not None]
        y_success = [e['y'] for e in successful_takeons if e['x'] is not None]
        ax.scatter(x_success, y_success, c=success_color, s=120, alpha=0.8, 
                  marker='o', edgecolors='#7d3c98', linewidths=2, 
                  label='Successful', zorder=4)
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Take-On / Dribble Map - {team_name}\n'
                f'Successful: {len(successful_takeons)} | Unsuccessful: {len(unsuccessful_takeons)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_aerial_duel_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing aerial duels."""
    aerials = [e for e in match_data['events'] 
               if e['team_id'] == team_id and e['type_id'] == 44]  # Aerial
    
    if not aerials:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    successful_aerials = [e for e in aerials if e['outcome'] == 1]
    unsuccessful_aerials = [e for e in aerials if e['outcome'] == 0]
    
    # Professional color scheme
    success_color = '#17a2b8'  # Teal
    fail_color = '#dc3545'     # Red
    
    # Plot unsuccessful aerials
    if unsuccessful_aerials:
        x_fail = [e['x'] for e in unsuccessful_aerials if e['x'] is not None]
        y_fail = [e['y'] for e in unsuccessful_aerials if e['x'] is not None]
        ax.scatter(x_fail, y_fail, c=fail_color, s=80, alpha=0.6, 
                  marker='x', linewidths=2, label='Lost', zorder=3)
    
    # Plot successful aerials
    if successful_aerials:
        x_success = [e['x'] for e in successful_aerials if e['x'] is not None]
        y_success = [e['y'] for e in successful_aerials if e['x'] is not None]
        ax.scatter(x_success, y_success, c=success_color, s=120, alpha=0.8, 
                  marker='o', edgecolors='#138496', linewidths=2, 
                  label='Won', zorder=4)
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Aerial Duel Map - {team_name}\n'
                f'Won: {len(successful_aerials)} | Lost: {len(unsuccessful_aerials)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_clearance_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing clearances."""
    clearances = [e for e in match_data['events'] 
                  if e['team_id'] == team_id and e['type_id'] == 12]  # Clearance
    
    if not clearances:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    x_coords = [e['x'] for e in clearances if e['x'] is not None]
    y_coords = [e['y'] for e in clearances if e['x'] is not None]
    
    # Professional color scheme
    clearance_color = '#6c757d'  # Gray
    
    # Plot clearances
    ax.scatter(x_coords, y_coords, c=clearance_color, s=100, alpha=0.7, 
              marker='s', edgecolors='#495057', linewidths=2, 
              label='Clearances', zorder=4)
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Clearance Map - {team_name}\n'
                f'Total Clearances: {len(clearances)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_through_ball_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing through balls with arrows."""
    through_balls = [e for e in match_data['events'] 
                     if e['team_id'] == team_id and e['type_id'] == 1 and e['is_through_ball']]
    
    if not through_balls:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    successful = [e for e in through_balls if e['outcome'] == 1]
    unsuccessful = [e for e in through_balls if e['outcome'] == 0]
    
    # Professional color scheme
    success_color = '#00bcd4'  # Cyan
    fail_color = '#dc3545'     # Red
    
    # Plot unsuccessful through balls
    if unsuccessful:
        x_fail = [e['x'] for e in unsuccessful if e['x'] is not None]
        y_fail = [e['y'] for e in unsuccessful if e['x'] is not None]
        ax.scatter(x_fail, y_fail, c=fail_color, s=80, alpha=0.6, 
                  marker='x', linewidths=2, label='Unsuccessful', zorder=3)
    
    # Plot successful through balls with arrows
    arrows_drawn = 0
    if successful:
        x_success = [e['x'] for e in successful if e['x'] is not None]
        y_success = [e['y'] for e in successful if e['x'] is not None]
        ax.scatter(x_success, y_success, c=success_color, s=120, alpha=0.8, 
                  marker='o', edgecolors='#0097a7', linewidths=2, 
                  label='Successful', zorder=4)
        
        # Draw arrows for successful through balls
        for tb in successful:
            if tb['x'] is None or tb['y'] is None:
                continue
            end_x, end_y = get_pass_end_location(tb)
            if end_x is not None and end_y is not None:
                dx = end_x - tb['x']
                dy = end_y - tb['y']
                distance = np.sqrt(dx**2 + dy**2)
                if distance > 2:
                    dx_norm = dx / distance
                    dy_norm = dy / distance
                    arrow_length = distance * 0.9
                    ax.arrow(tb['x'], tb['y'], dx_norm * arrow_length, dy_norm * arrow_length,
                            head_width=2, head_length=2, fc=success_color, ec='#0097a7',
                            linewidth=2, alpha=0.7, zorder=3, length_includes_head=True)
                    arrows_drawn += 1
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Through Ball Map - {team_name}\n'
                f'Successful: {len(successful)} | Unsuccessful: {len(unsuccessful)} | Arrows: {arrows_drawn}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_assist_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing assists (passes that led to goals)."""
    assists = [e for e in match_data['events'] 
               if e['team_id'] == team_id and e['is_assist']]
    
    if not assists:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    x_coords = [e['x'] for e in assists if e['x'] is not None]
    y_coords = [e['y'] for e in assists if e['x'] is not None]
    
    # Professional color scheme - gold for assists
    assist_color = '#ffd700'  # Gold
    
    # Draw arrows from assist location to goal location
    arrows_drawn = 0
    for assist in assists:
        if assist['x'] is None or assist['y'] is None:
            continue
        end_x, end_y = get_pass_end_location(assist)
        if end_x is not None and end_y is not None:
            dx = end_x - assist['x']
            dy = end_y - assist['y']
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 2:
                dx_norm = dx / distance
                dy_norm = dy / distance
                arrow_length = distance * 0.9
                ax.arrow(assist['x'], assist['y'], dx_norm * arrow_length, dy_norm * arrow_length,
                        head_width=3, head_length=3, fc=assist_color, ec='#1a1a1a',
                        linewidth=3, alpha=0.9, zorder=4, length_includes_head=True)
                arrows_drawn += 1
    
    # Plot assist locations
    ax.scatter(x_coords, y_coords, c=assist_color, s=300, alpha=0.95,
              marker='o', edgecolors='#1a1a1a', linewidths=3, zorder=5,
              label=f'Assists ({arrows_drawn} with arrows)')
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Assist Map - {team_name}\n'
                f'Total Assists: {len(assists)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_ball_recovery_map(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create map showing ball recoveries."""
    recoveries = [e for e in match_data['events'] 
                  if e['team_id'] == team_id and e['type_id'] == 49]  # Ball recovery
    
    if not recoveries:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    x_coords = [e['x'] for e in recoveries if e['x'] is not None]
    y_coords = [e['y'] for e in recoveries if e['x'] is not None]
    
    # Professional color scheme
    recovery_color = '#20c997'  # Teal green
    
    # Create heat map
    if x_coords:
        H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=15,
                                          range=[[0, 100], [0, 100]])
        H = H.T
        im = ax.imshow(H, extent=[0, 100, 0, 100], origin='lower',
                      cmap='Greens', alpha=0.6, interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Recovery Density')
    
    # Plot individual recoveries
    ax.scatter(x_coords, y_coords, c=recovery_color, s=80, alpha=0.7,
              marker='o', edgecolors='#17a2b8', linewidths=1.5, zorder=4,
              label='Ball Recoveries')
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Ball Recovery Map - {team_name}\n'
                f'Total Recoveries: {len(recoveries)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='#1a1a1a', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('#ffffff')
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_defensive_actions_heatmap(match_data: dict, team_id: str, output_path: str = None, pdf_page=None):
    """Create heatmap of all defensive actions (tackles, interceptions, clearances, aerial duels)."""
    tackles = [e for e in match_data['events'] 
               if e['team_id'] == team_id and e['type_id'] == 7]
    interceptions = [e for e in match_data['events'] 
                     if e['team_id'] == team_id and e['type_id'] == 8]
    clearances = [e for e in match_data['events'] 
                  if e['team_id'] == team_id and e['type_id'] == 12]
    aerials = [e for e in match_data['events'] 
               if e['team_id'] == team_id and e['type_id'] == 44]
    
    all_defensive = tackles + interceptions + clearances + aerials
    
    if not all_defensive:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax)
    
    x_coords = [e['x'] for e in all_defensive if e['x'] is not None]
    y_coords = [e['y'] for e in all_defensive if e['x'] is not None]
    
    # Create heat map
    if x_coords:
        H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=20,
                                          range=[[0, 100], [0, 100]])
        H = H.T
        im = ax.imshow(H, extent=[0, 100, 0, 100], origin='lower',
                      cmap='Reds', alpha=0.7, interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Defensive Action Density')
    
    team_name = match_data['home_team_name'] if team_id == match_data['home_team_id'] else match_data['away_team_name']
    ax.set_title(f'Defensive Actions Heat Map - {team_name}\n'
                f'Tackles: {len(tackles)} | Interceptions: {len(interceptions)} | '
                f'Clearances: {len(clearances)} | Aerials: {len(aerials)}',
                fontsize=16, fontweight='bold', color='#1a1a1a', pad=20)
    plt.tight_layout()
    
    if pdf_page:
        pdf_page.savefig(fig, bbox_inches='tight')
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def main():
    """Main function to create all pitch map visualizations."""
    print("="*70)
    print("La Liga Floodlight Example - Comprehensive Pitch Maps")
    print("="*70)
    
    # Find XML files
    # Data directory - modify this to point to your OPTA F24 XML files
    # Default path assumes data is in: ../Task 2/F24 La Liga 2023/
    data_dir = "../Task 2/F24 La Liga 2023"
    
    # Alternative: Use a relative path from script location
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(script_dir, "data")
    
    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))
    
    if not xml_files:
        print(f"No XML files found in {data_dir}")
        print("\nPlease ensure:")
        print("1. OPTA F24 XML files are in the specified directory")
        print("2. The data_dir path in the script is correct")
        print("3. Files have .xml extension")
        print("\nSee DATA_README.md for information on obtaining data.")
        return
    
    # Use first file as example
    sample_file = xml_files[0]
    print(f"\nLoading match: {os.path.basename(sample_file)}")
    
    # Parse the XML file
    match_data = parse_opta_xml(sample_file)
    
    if not match_data:
        print("Failed to parse XML file")
        return
    
    print(f"\nMatch ID: {match_data['match_id']}")
    print(f"Home Team: {match_data['home_team_name']} (ID: {match_data['home_team_id']})")
    print(f"Away Team: {match_data['away_team_name']} (ID: {match_data['away_team_id']})")
    print(f"Score: {match_data['home_score']}-{match_data['away_score']}")
    print(f"Total Events: {len(match_data['events'])}")
    
    # Create output directory
    output_dir = "pitch_maps"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Creating Pitch Maps...")
    print(f"{'='*70}\n")
    
    # Helper function to sanitize team names for filenames
    def sanitize_filename(name):
        """Remove/replace characters that aren't safe for filenames."""
        # Replace spaces and special characters
        name = name.replace(' ', '_')
        name = name.replace('/', '_')
        name = name.replace('\\', '_')
        name = name.replace(':', '_')
        name = name.replace('*', '_')
        name = name.replace('?', '_')
        name = name.replace('"', '_')
        name = name.replace('<', '_')
        name = name.replace('>', '_')
        name = name.replace('|', '_')
        return name
    
    # Get sanitized team names and match date
    home_team_safe = sanitize_filename(match_data['home_team_name'])
    away_team_safe = sanitize_filename(match_data['away_team_name'])
    match_date = match_data.get('match_date', datetime.now().strftime('%Y-%m-%d'))
    
    # Create PDF filenames: TeamName_vs_Opponent_Date.pdf
    home_pdf_path = f"{output_dir}/{home_team_safe}_vs_{away_team_safe}_{match_date}.pdf"
    away_pdf_path = f"{output_dir}/{away_team_safe}_vs_{home_team_safe}_{match_date}.pdf"
    
    # Create PDFs and individual PNGs for home team
    print(f"Home Team: {match_data['home_team_name']}")
    with PdfPages(home_pdf_path) as home_pdf:
        # Pass map
        fig = create_pass_map(match_data, match_data['home_team_id'], 
                             f"{output_dir}/home_pass_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Key pass map
        fig = create_key_pass_map(match_data, match_data['home_team_id'], 
                                 f"{output_dir}/home_key_pass_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Shot map
        fig = create_shot_map(match_data, match_data['home_team_id'], 
                            f"{output_dir}/home_shot_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Cross map
        fig = create_cross_map(match_data, match_data['home_team_id'], 
                              f"{output_dir}/home_cross_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Tackle map
        fig = create_tackle_map(match_data, match_data['home_team_id'], 
                               f"{output_dir}/home_tackle_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Pass density heatmap
        fig = create_pass_density_heatmap(match_data, match_data['home_team_id'], 
                                         f"{output_dir}/home_pass_density.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Combined metrics map
        fig = create_combined_metrics_map(match_data, match_data['home_team_id'], 
                                         f"{output_dir}/home_combined_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # NEW MAPS
        # Interception map
        fig = create_interception_map(match_data, match_data['home_team_id'], 
                                     f"{output_dir}/home_interception_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Take-on map
        fig = create_takeon_map(match_data, match_data['home_team_id'], 
                               f"{output_dir}/home_takeon_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Aerial duel map
        fig = create_aerial_duel_map(match_data, match_data['home_team_id'], 
                                     f"{output_dir}/home_aerial_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Clearance map
        fig = create_clearance_map(match_data, match_data['home_team_id'], 
                                   f"{output_dir}/home_clearance_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Through ball map
        fig = create_through_ball_map(match_data, match_data['home_team_id'], 
                                      f"{output_dir}/home_throughball_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Assist map
        fig = create_assist_map(match_data, match_data['home_team_id'], 
                               f"{output_dir}/home_assist_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Ball recovery map
        fig = create_ball_recovery_map(match_data, match_data['home_team_id'], 
                                      f"{output_dir}/home_recovery_map.png", home_pdf)
        if fig:
            plt.close(fig)
        
        # Defensive actions heatmap
        fig = create_defensive_actions_heatmap(match_data, match_data['home_team_id'], 
                                               f"{output_dir}/home_defensive_heatmap.png", home_pdf)
        if fig:
            plt.close(fig)
    
    print(f"  ✓ Created PDF: {home_pdf_path}")
    print()
    
    # Create PDFs and individual PNGs for away team
    print(f"Away Team: {match_data['away_team_name']}")
    with PdfPages(away_pdf_path) as away_pdf:
        # Pass map
        fig = create_pass_map(match_data, match_data['away_team_id'], 
                             f"{output_dir}/away_pass_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Key pass map
        fig = create_key_pass_map(match_data, match_data['away_team_id'], 
                                 f"{output_dir}/away_key_pass_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Shot map
        fig = create_shot_map(match_data, match_data['away_team_id'], 
                            f"{output_dir}/away_shot_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Cross map
        fig = create_cross_map(match_data, match_data['away_team_id'], 
                              f"{output_dir}/away_cross_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Tackle map
        fig = create_tackle_map(match_data, match_data['away_team_id'], 
                               f"{output_dir}/away_tackle_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Pass density heatmap
        fig = create_pass_density_heatmap(match_data, match_data['away_team_id'], 
                                         f"{output_dir}/away_pass_density.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Combined metrics map
        fig = create_combined_metrics_map(match_data, match_data['away_team_id'], 
                                         f"{output_dir}/away_combined_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # NEW MAPS
        # Interception map
        fig = create_interception_map(match_data, match_data['away_team_id'], 
                                     f"{output_dir}/away_interception_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Take-on map
        fig = create_takeon_map(match_data, match_data['away_team_id'], 
                               f"{output_dir}/away_takeon_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Aerial duel map
        fig = create_aerial_duel_map(match_data, match_data['away_team_id'], 
                                     f"{output_dir}/away_aerial_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Clearance map
        fig = create_clearance_map(match_data, match_data['away_team_id'], 
                                   f"{output_dir}/away_clearance_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Through ball map
        fig = create_through_ball_map(match_data, match_data['away_team_id'], 
                                      f"{output_dir}/away_throughball_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Assist map
        fig = create_assist_map(match_data, match_data['away_team_id'], 
                               f"{output_dir}/away_assist_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Ball recovery map
        fig = create_ball_recovery_map(match_data, match_data['away_team_id'], 
                                      f"{output_dir}/away_recovery_map.png", away_pdf)
        if fig:
            plt.close(fig)
        
        # Defensive actions heatmap
        fig = create_defensive_actions_heatmap(match_data, match_data['away_team_id'], 
                                              f"{output_dir}/away_defensive_heatmap.png", away_pdf)
        if fig:
            plt.close(fig)
    
    print(f"  ✓ Created PDF: {away_pdf_path}")
    
    print(f"\n{'='*70}")
    print("All visualizations created successfully!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}/")
    print("\nGenerated files:")
    print(f"  {match_data['home_team_name']}:")
    print(f"    - {os.path.basename(home_pdf_path)} (all 15 maps)")
    print("    - home_pass_map.png")
    print("    - home_key_pass_map.png")
    print("    - home_shot_map.png")
    print("    - home_cross_map.png")
    print("    - home_tackle_map.png")
    print("    - home_pass_density.png")
    print("    - home_combined_map.png")
    print("    - home_interception_map.png")
    print("    - home_takeon_map.png")
    print("    - home_aerial_map.png")
    print("    - home_clearance_map.png")
    print("    - home_throughball_map.png")
    print("    - home_assist_map.png")
    print("    - home_recovery_map.png")
    print("    - home_defensive_heatmap.png")
    print(f"  {match_data['away_team_name']}:")
    print(f"    - {os.path.basename(away_pdf_path)} (all 15 maps)")
    print("    - away_pass_map.png")
    print("    - away_key_pass_map.png")
    print("    - away_shot_map.png")
    print("    - away_cross_map.png")
    print("    - away_tackle_map.png")
    print("    - away_pass_density.png")
    print("    - away_combined_map.png")
    print("    - away_interception_map.png")
    print("    - away_takeon_map.png")
    print("    - away_aerial_map.png")
    print("    - away_clearance_map.png")
    print("    - away_throughball_map.png")
    print("    - away_assist_map.png")
    print("    - away_recovery_map.png")
    print("    - away_defensive_heatmap.png")
    print("\nThese visualizations demonstrate what Floodlight can create")
    print("with its built-in visualization functions.")


if __name__ == "__main__":
    main()
