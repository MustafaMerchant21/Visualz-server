from streetview import search_panoramas, get_panorama, get_streetview

# panos = search_panoramas(lat=19.994736776611337, lon=73.79987218425471)
# first = panos[0]
# print(first)

# image = get_panorama(pano_id=f"{first.pano_id}")
# # image = get_panorama(pano_id=f"UhNdHJ3LSfyHNCT5x5DpyA")

# # image = get_streetview(
# #     pano_id="UhNdHJ3LSfyHNCT5x5DpyA",
# #     api_key="AIzaSyCzCP6gRKwENsXbi0Xr1sfURbOLRSPExwo",
# # )

# image.save("image.jpg", "jpeg")

# Import google_streetview for the api module
import google_streetview.api

# Define parameters for street view api
params = [{
	'size': '600x300', # max 640x640 pixels
	'location': '19.994736776611337,73.79987218425471',
	'heading': '151.78',
	'pitch': '-0.76',
	'key': 'AIzaSyCzCP6gRKwENsXbi0Xr1sfURbOLRSPExwo'
}]

# Create a results object
results = google_streetview.api.results(params)

# Download images to directory 'downloads'
results.download_links('downloads')