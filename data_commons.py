
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
    
    columns = _df.columns.to_list().remove('Name')
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


def get_dcid_from_county_name(county_name):
    simple_query = f"""
                    SELECT ?geoId
                    WHERE {{
                      ?county typeOf County .
                      ?county name '{county_name}' .
                      ?county dcid ?geoId .
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
