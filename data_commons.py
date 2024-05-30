
import datacommons_pandas as dc
            
def get_variables_for_fips(fips_list, variable_name_list):
    _df = dc.build_multivariate_dataframe(fips_list, variable_name_list)
    _df['name'] = _df.index.map(dc.get_property_values(_df.index, 'name'))
    _df['name'] = _df['name'].str[0]
    _df['fips'] = _df.index.to_series().apply(lambda x: x.replace("geoId/", ""))
    return _df

def get_time_series_dataframe_for_fips(fips_list, variable_name):
    _df = dc.build_time_series_dataframe(fips_list, variable_name)
    _df['name'] = _df.index.map(dc.get_property_values(_df.index, 'name'))
    _df['name'] = _df['name'].str[0]
    _df['fips'] = _df.index.to_series().apply(lambda x: x.replace("geoId/", ""))
    return _df
