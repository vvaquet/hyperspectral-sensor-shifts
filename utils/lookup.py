sensor_lookup_id_name ={'Coffee_1': {0:'SWIR3116', 1:'SWIR3505', 2:'SWIR3516'},
    'Coffee_2' : {0:'SWIR3116', 1:'SWIR3122'},
    'Coffee_3' : {0:'SWIR3505', 1:'Headwall'}}

sensor_lookup_name_id ={'Coffee_1': {'SWIR3116':0, 'SWIR3505':1, 'SWIR3516':2},
    'Coffee_2' : {'SWIR3116':0, 'SWIR3122':1},
    'Coffee_3' : {'SWIR3505':0, 'Headwall':1}}

min_max_wavelength = {'Coffee_1':(1000, 2500),
    'Coffee_2':(1000, 2500),
    'Coffee_3':(1000, 2500),
    'Red':(408, 986)}

static_sensor_names = {'Coffee_1': ['SWIR3116', 'SWIR3505', 'SWIR3516'],
                        'Coffee_2': ['SWIR3116', 'SWIR3122'],
                        'Coffee_3': ['SWIR3505', 'Headwall'],
                        'SWIR3516_transversal' : ['before', 'after'],
                        'SWIR3505_transversal' : ['before', 'after'],
                        'SWIR3505_intensity' : ['before', 'after'],
                        'Red' : ['VNIR0864', 'VNIR0031']}


colors = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#984ea3', '#a65628', '#f781bf',
                    '#999999', '#e41a1c', '#dede00', 'green', 'red']
lines = ['solid', 'dashed', 'dashdot']

coffee_names = ['Arabica', 'Robusta', 'immature Arabica']