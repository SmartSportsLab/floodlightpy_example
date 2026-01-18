# Data Requirements

This script requires **OPTA F24 XML files** to run.

## ⚠️ Important: Data Licensing

**DO NOT upload OPTA XML data files to GitHub or share them publicly.** These files contain proprietary data and sharing them violates OPTA's terms of service.

## How to Obtain Data

### Option 1: Educational Institution Access
- Check if your university/institution has access to OPTA data
- Contact your professor or library for access information
- Many educational institutions have partnerships with OPTA

### Option 2: StatsBomb Open Data (Free Alternative)
- **Website**: https://github.com/statsbomb/open-data
- **Format**: JSON (not XML)
- **Note**: You would need to adapt the parser in this script to work with StatsBomb JSON format
- **Free**: Yes, completely free and open

### Option 3: Contact OPTA Directly
- For educational licenses, contact OPTA
- They may offer educational discounts or free access for academic projects
- **Website**: https://www.optasports.com/

### Option 4: Course Materials
- Check if your professor has provided sample data
- Use the data provided for your course assignments
- Do not share this data outside your course

## Expected File Structure

The script expects XML files in this structure:

```
../Task 2/F24 La Liga 2023/
  ├── f24-23-2022-2301589-eventdetails.xml
  ├── f24-23-2022-2301590-eventdetails.xml
  └── ...
```

### To Use Your Own Data Location

Modify the `data_dir` variable in `laliga_floodlight_actual.py` (around line 1000):

```python
# Change this line:
data_dir = "../Task 2/F24 La Liga 2023"

# To your data location, for example:
data_dir = "path/to/your/opta/data"
# or
data_dir = "./data"
```

## File Format

The script expects OPTA F24 XML files with this structure:

```xml
<Games>
  <Game id="..." home_team_name="..." away_team_name="..." game_date="...">
    <Event type_id="1" ... />  <!-- Pass -->
    <Event type_id="13" ... /> <!-- Shot -->
    <Event type_id="16" ... /> <!-- Goal -->
    ...
  </Game>
</Games>
```

## Testing Without Data

If you want to test the script structure without actual data:

1. Create a minimal test XML file following OPTA F24 format
2. Or use StatsBomb open data and adapt the parser
3. Or wait until you have access to OPTA data

## Data Privacy

- **Never commit data files to Git**
- **Never share data files publicly**
- **Respect data licensing agreements**
- **Use `.gitignore` to prevent accidental uploads**

## Questions?

If you have questions about:
- **Getting data**: Contact your professor or institution
- **Data format**: Check OPTA documentation (if you have access)
- **Alternative data sources**: See StatsBomb open data
- **Script adaptation**: Modify the parser for your data format

---

**Remember**: This script is for educational purposes. Always ensure you have proper licensing for any data you use.
