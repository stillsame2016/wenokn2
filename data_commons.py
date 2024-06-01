
import datacommons_pandas as dc
            
def get_variables_for_fips(fips_list, variable_name_list):
    _df = dc.build_multivariate_dataframe(fips_list, variable_name_list)
    # _df = _df.fillna(0)
    # _df['name'] = _df.index.map(dc.get_property_values(_df.index, 'name'))
    _df.insert(0, 'Name', _df.index.map(dc.get_property_values(_df.index, 'name')))        
    _df['Name'] = _df['Name'].str[0]
    # _df['fips'] = _df.index.to_series().apply(lambda x: x.replace("geoId/", ""))
    return _df


def get_time_series_dataframe_for_fips(fips_list, variable_name):
    _df = dc.build_time_series_dataframe(fips_list, variable_name)
    # _df = _df.fillna(0)
    # _df['name'] = _df.index.map(dc.get_property_values(_df.index, 'name'))
    # _df['fips'] = _df.index.to_series().apply(lambda x: x.replace("geoId/", ""))
    _df.insert(0, 'Name', _df.index.map(dc.get_property_values(_df.index, 'name')))
    _df['Name'] = _df['Name'].str[0]
    _df.variable_name = variable_name
    return _df


def get_fips_from_county_name(county_name):
    simple_query = f"""
                    SELECT ?geoId
                    WHERE {
                      ?county typeOf City .
                      ?county name '{county_name}' .
                      ?county dcid ?geoId .
                    }
                    LIMIT 1
                 """
    try:
        # Execute the simple query
        fips_dict = dc.query(simple_query)
        fips = [ item['?geoId'] for item in fips_dict ]
        return fips
    except Exception as ex:
        return None
     
