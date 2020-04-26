#!/usr/bin/env python
# coding: utf-8

# # Capstone week 3 part3
# ## Analysis of data, segmentation, clustering

# In[2]:


import pandas as pd
pc_locs_df = pd.read_csv('loc1.csv').drop('Unnamed: 0', axis=1)
pc_locs_df.head()


# ### Drop postal code column

# In[3]:


pc_locs_df = pc_locs_df.drop('PostalCode', axis=1)


# ### getting marked map function

# In[7]:


import folium

def get_marked_map(data, latitude, longitude):
    map_borough = folium.Map(location=[latitude, longitude], zoom_start=11)

    # add markers to map
    for lat, lng, label in zip(data['latitude'], data['longitude'], data['Neighborhood']):
        label = folium.Popup(label, parse_html=True)
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            popup=label,
            color='blue',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7,
            parse_html=False).add_to(map_borough)
    return map_borough


# ## Marked map with all areas in Toronto

# In[8]:


from geopy.geocoders import Nominatim

# get cords of Toronto
address = 'Toronto, Canada'
geolocator = Nominatim(user_agent="canada_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))

# create map of canada using latitude and longitude values
map_newyork = get_marked_map(pc_locs_df, latitude, longitude)
map_newyork


# ### Function for getting cords using address

# In[10]:


def get_lat_lon(address):
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    return latitude, longitude


# ## Listing different areas in Toronto

# In[11]:


set(pc_locs_df['Borough'].values)


# ### Doing analysis only for Central Toronto

# In[12]:


def borough_data(area):
    return pc_locs_df[pc_locs_df['Borough'] == 'Central Toronto'].reset_index(drop=True)


# ## Get data related to central toronto

# In[13]:


borough_data =  borough_data('Central Toronto')
borough_data.head()


# ## Get cords of central toronto

# In[14]:


borough_address = '{}, Toronto'.format('Central Toronto')
latitude, longitude = get_lat_lon(borough_address)
print('The geograpical coordinate of {} are {}, {}.'.format(borough, latitude, longitude))


# In[15]:


map_borough = folium.Map(location=[latitude, longitude], zoom_start=11)
map_borough = get_marked_map(borough_data, latitude, longitude)
map_borough


# Next, we are going to start utilizing the Foursquare API to explore the neighborhoods and segment them.
# ### Define Foursquare Credentials and Version

# In[16]:


CLIENT_ID = 'xxxxxxxxxxxxxxxxx' # your Foursquare ID
CLIENT_SECRET = 'xxxxxxxxxxxxxxxxx' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 100


# In[17]:


borough_data.loc[0, 'Neighborhood']


# In[18]:


neighborhood_latitude = borough_data.loc[0, 'latitude'] # neighborhood latitude value
neighborhood_longitude = borough_data.loc[0, 'longitude'] # neighborhood longitude value

neighborhood_name = borough_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# ## getting top 100 places in 500m radius

# In[19]:


radius = 500
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[21]:


import requests

results = requests.get(url).json()
results


# In[22]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[26]:


from pandas.io.json import json_normalize

venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[27]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# ## 2. Explore Neighborhoods in Central Toronto

# In[28]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[29]:


borough_venues = getNearbyVenues(names=borough_data['Neighborhood'],
                                   latitudes=borough_data['latitude'],
                                   longitudes=borough_data['longitude']
                                  )


# In[30]:


print(borough_venues.shape)
borough_venues.head()


# In[31]:


borough_venues.groupby('Neighborhood').count()


# In[32]:


print('There are {} uniques categories.'.format(len(borough_venues['Venue Category'].unique())))


# ## 3. Analyze Each Neighborhood

# In[33]:


# one hot encoding
borough_onehot = pd.get_dummies(borough_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
borough_onehot['Neighborhood'] = borough_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [borough_onehot.columns[-1]] + list(borough_onehot.columns[:-1])
borough_onehot = borough_onehot[fixed_columns]

borough_onehot.head()


# In[34]:


borough_onehot.shape


# In[35]:


borough_grouped = borough_onehot.groupby('Neighborhood').mean().reset_index()
borough_grouped


# In[36]:


borough_grouped.shape


# In[37]:


num_top_venues = 5

for hood in borough_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = borough_grouped[borough_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[38]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[40]:


import numpy as np

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = borough_grouped['Neighborhood']

for ind in np.arange(borough_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(borough_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ### 4. Cluster Neighborhoods

# In[42]:


from sklearn.cluster import KMeans

# set number of clusters
kclusters = 5

borough_grouped_clustering = borough_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(borough_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[43]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

borough_merged = borough_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
borough_merged = borough_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

borough_merged.head() # check the last columns!


# In[46]:


import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(
    borough_merged['latitude'], borough_merged['longitude'], borough_merged['Neighborhood'], borough_merged['Cluster Labels']
):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## 5. Examine Clusters

# In[47]:


def examine(label):
    return borough_merged.loc[
        borough_merged['Cluster Labels'] == 0,
        borough_merged.columns[
            [1] + list(range(5, borough_merged.shape[1]))
        ]
    ]


# In[48]:


examine(0)


# In[49]:


examine(1)


# In[50]:


examine(2)


# In[51]:


examine(3)


# In[52]:


examine(4)


# In[ ]:




