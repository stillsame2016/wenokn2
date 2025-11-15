
import datacommons_pandas as dc
            
def get_variables_for_dcid(dcid_list, variable_name_list):
    _df = dc.build_multivariate_dataframe(dcid_list, variable_name_list)
    _df.insert(0, 'Name', _df.index.map(dc.get_property_values(_df.index, 'name')))        
    _df['Name'] = _df['Name'].str[0]
    return _df


def get_time_series_dataframe_for_dcid(dcid_list, variable_name):
    _df = dc.build_time_series_dataframe(dcid_list, variable_name)    
    _df.insert(0, 'Name', _df.index.map(dc.get_property_values(_df.index, 'name')))
    _df['Name'] = _df['Name'].str[0]    
    
    # columns = _df.columns.to_list().remove('Name')
    columns = _df.columns.to_list()
    columns.remove('Name')
    _df = _df.melt(
        ['Name'],
        columns,
        'Date',
         variable_name,
    )
    _df = _df.dropna()     
    _df = _df.drop_duplicates(keep='first')
    _df.variable_name = variable_name
    return _df


# def get_dcid_from_county_name(county_name):
#     simple_query = f"""
#                     SELECT ?geoId
#                     WHERE {{
#                       ?county typeOf County .
#                       ?county name '{county_name}' .
#                       ?county dcid ?geoId .
#                     }}
#                     LIMIT 1
#                  """
#     try:
#         # Execute the simple query
#         dcid_dict = dc.query(simple_query)
#         dcid = [ item['?geoId'] for item in dcid_dict ]
#         return dcid[0]
#     except Exception as ex:
#         return None

def get_dcid_from_county_name(county_name):
    """
    Get DCID for a county name.
    
    Args:
        county_name: Can be:
            - 'Ross County' or 'Ross'
            - 'Ross County, Ohio' or 'Ross, Ohio'
            - 'Los Angeles County' or 'Los Angeles'
            
    Returns:
        DCID string if unique match found
        
    Raises:
        ValueError: If multiple counties match or invalid format
        Exception: If no county found or query fails
    """
    # Parse the county name to handle both formats
    parts = [p.strip() for p in county_name.split(',')]
    
    if len(parts) == 1:
        # Format: 'Ross County' or 'Ross'
        county_only = parts[0]
        state_filter = ""
    elif len(parts) == 2:
        # Format: 'Ross County, Ohio' or 'Ross, Ohio'
        county_only = parts[0]
        state_name = parts[1]
        # Add state filter to the query
        state_filter = f"""
          ?county containedInPlace ?state .
          ?state typeOf State .
          ?state name '{state_name}' .
        """
    else:
        raise ValueError(f"Invalid county name format: '{county_name}'. Use 'County Name' or 'County Name, State'")
    
    # Append " County" if not already present
    if not county_only.lower().endswith(' county'):
        county_only = county_only + ' County'
    
    # Build query with optional state filter
    simple_query = f"""
                    SELECT ?geoId
                    WHERE {{
                      ?county typeOf County .
                      ?county name '{county_only}' .
                      ?county dcid ?geoId .
                      {state_filter}
                    }}
                 """
    
    try:
        # Execute the query
        dcid_dict = dc.query(simple_query)
        dcid_list = [item['?geoId'] for item in dcid_dict]
        
        if len(dcid_list) == 0:
            raise Exception(f"No county found with name '{county_name}'")
        elif len(dcid_list) > 1:
            # Extract just the base name for the error message
            base_name = county_only.replace(' County', '')
            raise ValueError(
                f"Multiple counties found with name '{base_name}' ({len(dcid_list)} matches). "
                f"Please specify the state: '{base_name}, State Name'"
            )
        
        return dcid_list[0]
        
    except ValueError:
        # Re-raise ValueError (our custom error for multiple matches)
        raise
    except Exception as ex:
        raise Exception(f"Error querying county '{county_name}': {str(ex)}")


def get_dcid_from_state_name(state_name):
    simple_query = f"""
                    SELECT ?geoId
                    WHERE {{
                      ?state typeOf State .
                      ?state name '{state_name}' .
                      ?state dcid ?geoId .
                    }}
                    LIMIT 1
                 """
    try:
        # Execute the simple query
        dcid_dict = dc.query(simple_query)
        dcid = [ item['?geoId'] for item in dcid_dict ]
        return dcid[0]
    except Exception as ex:
        return None


def get_dcid_from_country_name(country_name):
    simple_query = f"""
                    SELECT ?geoId
                    WHERE {{
                      ?country typeOf Country .
                      ?country name '{country_name}' .
                      ?country dcid ?geoId .
                    }}
                    LIMIT 1
                 """
    try:
        # Execute the simple query
        dcid_dict = dc.query(simple_query)
        dcid = [ item['?geoId'] for item in dcid_dict ]
        return dcid[0]
    except Exception as ex:
        return None
