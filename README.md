# La Liga Floodlight Example - Pitch Map Visualizations

A comprehensive Python script for creating football pitch map visualizations from OPTA F24 XML data.

## Features

- **15 different pitch map visualizations** including:
  - Pass maps (successful/unsuccessful)
  - Key passes (with arrows showing direction)
  - Shots and goals
  - Crosses
  - Tackles
  - Interceptions
  - Take-ons/Dribbles
  - Aerial duels
  - Clearances
  - Through balls
  - Assists
  - Ball recoveries
  - Pass density heatmaps
  - Defensive actions heatmaps
  - Combined metrics overview

- **Professional visualizations** with enhanced pitch design
- **PDF report generation** - One comprehensive PDF per team with all visualizations
- **Individual PNG files** for each visualization
- **Automatic file naming** - PDFs named with team names, opponent, and match date

## Requirements

- Python 3.7+
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- OPTA F24 XML data files

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Obtain OPTA F24 XML data files** (see [Data Requirements](#data-requirements) below)

2. **Place your XML files** in the appropriate directory. By default, the script looks for:
   ```
   ../Task 2/F24 La Liga 2023/*.xml
   ```
   
   You can modify the `data_dir` variable in the script (line ~1200) to point to your data location.

3. **Run the script**:
   ```bash
   python laliga_floodlight_example.py
   ```

4. **Output files** will be generated in the `pitch_maps/` directory:
   - Individual PNG files for each visualization
   - PDF reports named: `{TeamName}_vs_{OpponentName}_{Date}.pdf`

## Data Requirements

This script requires **OPTA F24 XML files** to run.

### How to Obtain Data

1. **Educational Use**: Check if your institution has access to OPTA data
2. **StatsBomb Open Data**: Free alternative - https://github.com/statsbomb/open-data
   - Note: StatsBomb uses JSON format, so you'd need to adapt the parser
3. **Contact OPTA**: For educational licenses, contact OPTA directly
4. **Your Course Materials**: Check if your professor has provided sample data

### Expected File Structure

Place your OPTA F24 XML files in a directory structure like:
```
../Task 2/F24 La Liga 2023/
  ├── f24-23-2022-2301589-eventdetails.xml
  ├── f24-23-2022-2301590-eventdetails.xml
  └── ...
```

Or modify the `data_dir` variable in the script to match your structure.

## Output

The script generates:

- **Individual PNG files** for each visualization type
- **PDF reports** (one per team) containing all 15 visualizations
- Files are automatically named with team names, opponent, and match date

Example output:
```
pitch_maps/
  ├── Real_Betis_vs_Elche_2022-08-15.pdf
  ├── Elche_vs_Real_Betis_2022-08-15.pdf
  ├── home_pass_map.png
  ├── home_key_pass_map.png
  └── ... (30 total files: 15 per team)
```

## How It Works

1. **Parses OPTA F24 XML** files to extract event data
2. **Identifies key passes** using OPTA's definition: "The final pass from a player to their teammate who then makes an attempt on Goal without scoring"
3. **Creates visualizations** for various metrics using matplotlib
4. **Generates PDFs** with all visualizations for easy sharing and presentation

## Key Features Explained

### Key Pass Identification
The script correctly identifies key passes by:
- Finding all shots (type_id=13) that don't result in goals
- Looking backwards to find the final pass by the same team before each shot
- This matches OPTA's official definition

### Visualization Design
- Professional pitch design with enhanced styling
- Color-coded events (successful/unsuccessful)
- Arrows showing pass directions for key passes, through balls, and assists
- Heatmaps for density analysis
- Clear legends and statistics

## Example

The script was tested on a La Liga 2023 match:
- **Real Betis** (home) vs **Elche** (away)
- **Date**: August 15, 2022
- **Score**: 3-0

Generated 15 different visualizations showing:
- 900 total passes
- 6 key passes
- 6 shots (5 home, 1 away)
- 3 goals
- And many more metrics...

## Troubleshooting

### "No XML files found"
- Check that your `data_dir` path is correct
- Ensure XML files are in the specified directory
- Verify file extensions are `.xml`

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.7+)

### Empty visualizations
- Some maps may be empty if the team didn't perform those actions
- This is normal - not all teams will have all event types in every match

## Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new visualization types
- Enhance the design

## License

This educational project is provided as-is for learning purposes. Please respect OPTA's data licensing terms - do not share proprietary OPTA data files.

## Acknowledgments

- Built for Module 10 - Collaborative Activity
- Uses OPTA F24 XML data format
- Inspired by Floodlight library capabilities
- Visualization design inspired by professional sports analytics

## Related Resources

- **Floodlight Library**: https://github.com/floodlight-sports/floodlight
- **OPTA Documentation**: Check with your institution for access
- **StatsBomb Open Data**: https://github.com/statsbomb/open-data
- **mplsoccer**: https://github.com/andrewRowlinson/mplsoccer (alternative visualization library)

---

**Note**: This script is designed for educational purposes. Always ensure you have proper licensing for any data you use.
