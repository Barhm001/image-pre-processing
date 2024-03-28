#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:04:33 2024

@author: Barhm001
"""

# import os
# import numpy as np
# import pandas as pd
# from skimage.io import imread
# from skimage.segmentation import slic, mark_boundaries
# from skimage.measure import regionprops, label
# from skimage.util import img_as_float
# from skimage.filters import threshold_otsu
# import matplotlib.pyplot as plt
 
# def calculate_luminance(r, g, b):
#     return 0.2126 * r + 0.7152 * g + 0.0722 * b

# def process_and_plot_image(sky_image_path, mask_image, output_folder):
#     sky_image = imread(sky_image_path)
#     sky_image_float = img_as_float(sky_image)

#     mask = mask_image > 0.5
#     masked_sky_image = sky_image_float.copy()
#     masked_sky_image[~mask] = 0

#     segments_slic = slic(masked_sky_image, n_segments=250, compactness=10, sigma=1, start_label=1, mask=mask)
#     labeled_segments = label(segments_slic + 1)
#     segmented_image = mark_boundaries(sky_image_float, labeled_segments, color=(1, 0, 0))

#     plt.figure(figsize=(10, 10))
#     plt.imshow(segmented_image)
#     plt.axis('off')
#     plt.savefig(os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_segmented.png'))
#     plt.close()

#     superpixel_features = []
#     for region in regionprops(labeled_segments, intensity_image=sky_image_float):
#         if np.all(region.intensity_image == 0):
#             continue

#         mean_intensity_R = np.mean(region.intensity_image[:, :, 0])
#         mean_intensity_G = np.mean(region.intensity_image[:, :, 1])
#         mean_intensity_B = np.mean(region.intensity_image[:, :, 2])
#         luminance = calculate_luminance(mean_intensity_R, mean_intensity_G, mean_intensity_B)
#         red_blue_ratio = mean_intensity_R / mean_intensity_B if mean_intensity_B > 0 else 0

#         features = {
#             'Superpixel Label': region.label,
#             'Area': region.area,
#             'Mean Intensity R': mean_intensity_R,
#             'Mean Intensity G': mean_intensity_G,
#             'Mean Intensity B': mean_intensity_B,
#             'Mean Intensity': luminance,
#             'Red-Blue Ratio': red_blue_ratio,
#             'Luminance': luminance
#         }
#         superpixel_features.append(features)

#     max_red_blue_ratio = max(feature['Red-Blue Ratio'] for feature in superpixel_features)
#     candidates = [f for f in superpixel_features if f['Red-Blue Ratio'] == max_red_blue_ratio]
#     sun_superpixel = max(candidates, key=lambda x: x['Luminance'])
#     sun_superpixel_label = sun_superpixel['Superpixel Label']

#     otsu_threshold = threshold_otsu(np.array([f['Red-Blue Ratio'] for f in superpixel_features]))

#     for feature in superpixel_features:
#         if feature['Superpixel Label'] == sun_superpixel_label:
#             feature['Cluster Label'] = 3
#         elif feature['Red-Blue Ratio'] < otsu_threshold:
#             feature['Cluster Label'] = 2
#         elif feature['Red-Blue Ratio'] == otsu_threshold:
#             feature['Cluster Label'] = 1
#         else:
#             feature['Cluster Label'] = 0

#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(sky_image_float)
#     ax.axis('off')
#     for feature in superpixel_features:
#         segment_mask = labeled_segments == feature['Superpixel Label']
#         region = regionprops(segment_mask.astype(int))[0]
#         y0, x0 = region.centroid
#         ax.text(x0, y0, str(feature['Cluster Label']), color='red', ha='center', va='center')

#     plt.savefig(os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_clusters.png'))
#     plt.close()

#     superpixel_features_df = pd.DataFrame(superpixel_features)
#     csv_filename = os.path.splitext(os.path.basename(sky_image_path))[0] + '_features.csv'
#     superpixel_features_df.to_csv(os.path.join(output_folder, csv_filename), index=False)

#     red_blue_ratios = [f['Red-Blue Ratio'] for f in superpixel_features]

#     plt.figure(figsize=(10, 6))
#     plt.hist(red_blue_ratios, bins=20, color='blue', edgecolor='blue')
#     plt.axvline(otsu_threshold, color='r', linestyle='dashed', linewidth=1)
#     plt.title('Histogram of Red-Blue Ratios with Otsu Threshold')
#     plt.xlabel('Red-Blue Ratio')
#     plt.ylabel('Frequency')
#     plt.savefig(os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_histogram.png'))
#     plt.close()

#     print(f'Otsu Threshold for {os.path.basename(sky_image_path)}: {otsu_threshold}')

# def main():
#     mask_image_path = '/Users/Barhm001/Desktop/CNN/SVM_CSI/Mask.png'
#     mask_image = imread(mask_image_path, as_gray=True)

#     sky_images_folder = '/Users/Barhm001/Desktop/CNN/SVM_CSI/20230708/'
#     output_folder = '/Users/Barhm001/Desktop/CNN/SVM_CSI/Features_file'

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for filename in os.listdir(sky_images_folder):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             sky_image_path = os.path.join(sky_images_folder, filename)
#             process_and_plot_image(sky_image_path, mask_image, output_folder)

# if __name__ == '__main__':
#     main()


#%%%

# import streamlit as st
# import os
# import datetime
# import numpy as np
# import pandas as pd
# from skimage.io import imread
# from skimage.segmentation import slic, mark_boundaries
# from skimage.measure import regionprops, label
# from skimage.util import img_as_float
# from skimage.filters import threshold_otsu
# import matplotlib.pyplot as plt
# from datetime import datetime


# def calculate_luminance(r, g, b):
#     return 0.2126 * r + 0.7152 * g + 0.0722 * b

# def process_and_plot_image(sky_image_path, mask_image, output_folder):
#     # Read and preprocess image
#     sky_image = imread(sky_image_path)
#     sky_image_float = img_as_float(sky_image)
#     mask = mask_image > 0.5
#     masked_sky_image = sky_image_float.copy()
#     masked_sky_image[~mask] = 0

#     # SLIC segmentation
#     segments_slic = slic(masked_sky_image, n_segments=250, compactness=10, sigma=1, start_label=1, mask=mask)
#     labeled_segments = label(segments_slic + 1)
#     segmented_image = mark_boundaries(sky_image_float, labeled_segments, color=(1, 0, 0))

#     # Save segmented image
#     segmented_img_path = os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_segmented.png')
#     plt.figure(figsize=(10, 10))
#     plt.imshow(segmented_image)
#     plt.axis('off')
#     plt.savefig(segmented_img_path)
#     plt.close()

#     # Feature extraction
#     superpixel_features = []
#     for region in regionprops(labeled_segments, intensity_image=sky_image_float):
#         if np.all(region.intensity_image == 0):
#             continue
#         mean_intensity_R = np.mean(region.intensity_image[:, :, 0])
#         mean_intensity_G = np.mean(region.intensity_image[:, :, 1])
#         mean_intensity_B = np.mean(region.intensity_image[:, :, 2])
#         luminance = calculate_luminance(mean_intensity_R, mean_intensity_G, mean_intensity_B)
#         red_blue_ratio = mean_intensity_R / mean_intensity_B if mean_intensity_B > 0 else 0

#         features = {
#             'Superpixel Label': region.label,
#             'Area': region.area,
#             'Mean Intensity R': mean_intensity_R,
#             'Mean Intensity G': mean_intensity_G,
#             'Mean Intensity B': mean_intensity_B,
#             'Luminance': luminance,
#             'Red-Blue Ratio': red_blue_ratio
#         }
#         superpixel_features.append(features)

#     # Save features to CSV
#     superpixel_features_df = pd.DataFrame(superpixel_features)
#     csv_filename = os.path.splitext(os.path.basename(sky_image_path))[0] + '_features.csv'
#     superpixel_features_df.to_csv(os.path.join(output_folder, csv_filename), index=False)

#     red_blue_ratios = [f['Red-Blue Ratio'] for f in superpixel_features]
#     otsu_threshold = threshold_otsu(np.array(red_blue_ratios))

#     # Visualization and saving of the segmented image
#     segmented_img_path = os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_segmented.png')
#     plt.figure(figsize=(10, 10))
#     plt.imshow(segmented_image)
#     plt.axis('off')
#     plt.savefig(segmented_img_path)
#     plt.close()

#     # Visualization and saving of clusters on the original image
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(sky_image_float)
#     ax.axis('off')
#     for feature in superpixel_features:
#         if feature['Red-Blue Ratio'] < otsu_threshold:
#             cluster_label = 2
#         else:
#             cluster_label = 3  # Assigning cluster label based on the threshold
#         segment_mask = labeled_segments == feature['Superpixel Label']
#         region = regionprops(segment_mask.astype(int))[0]
#         y0, x0 = region.centroid
#         ax.text(x0, y0, str(cluster_label), color='red', ha='center', va='center')
#     clusters_img_path = os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_clusters.png')
#     plt.savefig(clusters_img_path)
#     plt.close()

#     # Visualization and saving of the histogram of Red-Blue Ratios
#     plt.figure(figsize=(10, 6))
#     plt.hist(red_blue_ratios, bins=20, color='blue', edgecolor='blue')
#     plt.axvline(otsu_threshold, color='r', linestyle='dashed', linewidth=1)
#     plt.title('Histogram of Red-Blue Ratios with Otsu Threshold')
#     plt.xlabel('Red-Blue Ratio')
#     plt.ylabel('Frequency')
#     histogram_img_path = os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_histogram.png')
#     plt.savefig(histogram_img_path)
#     plt.close()

#     return segmented_img_path, clusters_img_path, histogram_img_path

# def calculate_cloud_cover(features_df):
#     # Assuming the DataFrame columns as before
#     mean_intensity_column = 'Luminance'
#     area_column = 'Area'
    
#     threshold_intensity = features_df[mean_intensity_column].quantile(0.75)
#     features_df['is_cloud'] = features_df[mean_intensity_column] > threshold_intensity
#     total_cloud_area = features_df[features_df['is_cloud']][area_column].sum()
#     total_image_area = features_df[area_column].sum()
#     cloud_cover_percentage = (total_cloud_area / total_image_area) * 100
#     return cloud_cover_percentage

# def app():
#     st.title("Sky Image Processing with Datetime Selection")

#     # Load the mask image
#     mask_image_path = '/Users/Barhm001/Desktop/CNN/SVM_CSI/Mask.png'  # Update this path
#     mask_image = imread(mask_image_path, as_gray=True)

#     # User inputs for date and time
#     chosen_date = st.date_input("Choose a date", datetime.now())
#     chosen_hour = st.number_input("Hour", min_value=0, max_value=23, value=0)
#     chosen_minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
#     chosen_second = st.number_input("Second", min_value=0, max_value=59, value=0)

#     # Constructing datetime string from user inputs
#     datetime_str = f"{chosen_date.strftime('%Y%m%d')}{chosen_hour:02d}{chosen_minute:02d}{chosen_second:02d}"


#     # Path setup
#     sky_images_folder = '/Users/Barhm001/Desktop/DT/20240324/'  # Update this path if needed
#     output_folder = '/Users/Barhm001/Desktop/DT/streamlit/'  # Update this path if needed

#     if st.button("Process Image"):
#         # Attempt to find a matching image
#         matching_files = [f for f in os.listdir(sky_images_folder) if f.startswith(datetime_str) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         if matching_files:
#             sky_image_path = os.path.join(sky_images_folder, matching_files[0])
#             st.write("Processing and displaying the images for the selected datetime...")

#             # Process the image using the mask
#             segmented_img_path, clusters_img_path, histogram_img_path = process_and_plot_image(sky_image_path, mask_image, output_folder)

#             # Display the processed images
#             st.image(segmented_img_path, caption="Segmented Image")
#             st.image(clusters_img_path, caption="Clusters on Image")
#             st.image(histogram_img_path, caption="Histogram of Red-Blue Ratios")

#             # Load the generated features CSV file
#             csv_filename = os.path.splitext(os.path.basename(sky_image_path))[0] + '_features.csv'
#             features_df = pd.read_csv(os.path.join(output_folder, csv_filename))
#             cloud_cover_percentage = calculate_cloud_cover(features_df)

#             # Create a DataFrame for the cloud cover result and display it in a table
#             cloud_cover_result_df = pd.DataFrame({
#                 'Metric': ['Cloud Cover Percentage'],
#                 'Value': [f"{cloud_cover_percentage:.2f}%"]
#             })
#             st.write("Cloud Cover Result:")
#             st.table(cloud_cover_result_df)
#         else:
#             st.write("No original image found for the selected datetime.")

# if __name__ == "__main__":
#     app()

#%%

import streamlit as st
import numpy as np
import os
import pandas as pd
from datetime import datetime
from skimage.io import imread, imshow
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops, label
from skimage.util import img_as_float
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

def calculate_luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def process_and_plot_image(sky_image_path, mask_image, output_folder):
    # Read and preprocess image
    sky_image = imread(sky_image_path)
    sky_image_float = img_as_float(sky_image)
    mask = mask_image > 0.5
    masked_sky_image = sky_image_float.copy()
    masked_sky_image[~mask] = 0

    # SLIC segmentation
    segments_slic = slic(masked_sky_image, n_segments=250, compactness=10, sigma=1, start_label=1, mask=mask)
    labeled_segments = label(segments_slic + 1)
    segmented_image = mark_boundaries(sky_image_float, labeled_segments, color=(1, 0, 0))

    # Save segmented image
    segmented_img_path = os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_segmented.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.savefig(segmented_img_path)
    plt.close()

    # Feature extraction
    superpixel_features = []
    for region in regionprops(labeled_segments, intensity_image=sky_image_float):
        if np.all(region.intensity_image == 0):
            continue
        mean_intensity_R = np.mean(region.intensity_image[:, :, 0])
        mean_intensity_G = np.mean(region.intensity_image[:, :, 1])
        mean_intensity_B = np.mean(region.intensity_image[:, :, 2])
        luminance = calculate_luminance(mean_intensity_R, mean_intensity_G, mean_intensity_B)
        red_blue_ratio = mean_intensity_R / mean_intensity_B if mean_intensity_B > 0 else 0

        features = {
            'Superpixel Label': region.label,
            'Area': region.area,
            'Mean Intensity R': mean_intensity_R,
            'Mean Intensity G': mean_intensity_G,
            'Mean Intensity B': mean_intensity_B,
            'Luminance': luminance,
            'Red-Blue Ratio': red_blue_ratio
        }
        superpixel_features.append(features)

    # Save features to CSV
    superpixel_features_df = pd.DataFrame(superpixel_features)
    csv_filename = os.path.splitext(os.path.basename(sky_image_path))[0] + '_features.csv'
    superpixel_features_df.to_csv(os.path.join(output_folder, csv_filename), index=False)

    red_blue_ratios = [f['Red-Blue Ratio'] for f in superpixel_features]
    otsu_threshold = threshold_otsu(np.array(red_blue_ratios))

    # Visualization and saving of the segmented image
    segmented_img_path = os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_segmented.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.savefig(segmented_img_path)
    plt.close()

    # Visualization and saving of clusters on the original image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(sky_image_float)
    ax.axis('off')
    for feature in superpixel_features:
        if feature['Red-Blue Ratio'] < otsu_threshold:
            cluster_label = 2
        else:
            cluster_label = 3  # Assigning cluster label based on the threshold
        segment_mask = labeled_segments == feature['Superpixel Label']
        region = regionprops(segment_mask.astype(int))[0]
        y0, x0 = region.centroid
        ax.text(x0, y0, str(cluster_label), color='red', ha='center', va='center')
    clusters_img_path = os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_clusters.png')
    plt.savefig(clusters_img_path)
    plt.close()

    # Visualization and saving of the histogram of Red-Blue Ratios
    plt.figure(figsize=(10, 6))
    plt.hist(red_blue_ratios, bins=20, color='blue', edgecolor='blue')
    plt.axvline(otsu_threshold, color='r', linestyle='dashed', linewidth=1)
    plt.title('Histogram of Red-Blue Ratios with Otsu Threshold')
    plt.xlabel('Red-Blue Ratio')
    plt.ylabel('Frequency')
    histogram_img_path = os.path.join(output_folder, os.path.splitext(os.path.basename(sky_image_path))[0] + '_histogram.png')
    plt.savefig(histogram_img_path)
    plt.close()

    return segmented_img_path, clusters_img_path, histogram_img_path

def calculate_cloud_cover(features_df):
    # Assuming the DataFrame columns as before
    mean_intensity_column = 'Luminance'
    area_column = 'Area'
    
    threshold_intensity = features_df[mean_intensity_column].quantile(0.75)
    features_df['is_cloud'] = features_df[mean_intensity_column] > threshold_intensity
    total_cloud_area = features_df[features_df['is_cloud']][area_column].sum()
    total_image_area = features_df[area_column].sum()
    cloud_cover_percentage = (total_cloud_area / total_image_area) * 100
    return cloud_cover_percentage

def sky_image_processing_page():
    st.title("Sky Image Processing with Datetime Selection")

    mask_image_path = 'Mask/Mask.png'  # Update this path
    mask_image = imread(mask_image_path, as_gray=True)

    chosen_date = st.date_input("Choose a date", datetime.now())
    chosen_hour = st.number_input("Hour", min_value=0, max_value=23, value=0)
    chosen_minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
    chosen_second = st.number_input("Second", min_value=0, max_value=59, value=0)

    datetime_str = f"{chosen_date.strftime('%Y%m%d')}{chosen_hour:02d}{chosen_minute:02d}{chosen_second:02d}"

    sky_images_folder = '20240324'  # Update this path
    output_folder = 'Output'  # Update this path

    if st.button("Process Image"):
        matching_files = [f for f in os.listdir(sky_images_folder) if f.startswith(datetime_str) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if matching_files:
            sky_image_path = os.path.join(sky_images_folder, matching_files[0])
            st.write("Processing and displaying the images for the selected datetime...")

            segmented_img_path, clusters_img_path, histogram_img_path = process_and_plot_image(sky_image_path, mask_image, output_folder)

            st.image(segmented_img_path, caption="Segmented Image")
            st.image(clusters_img_path, caption="Clusters on Image")
            st.image(histogram_img_path, caption="Histogram of Red-Blue Ratios")

            csv_filename = os.path.splitext(os.path.basename(sky_image_path))[0] + '_features.csv'
            features_df = pd.read_csv(os.path.join(output_folder, csv_filename))
            cloud_cover_percentage = calculate_cloud_cover(features_df)

            cloud_cover_result_df = pd.DataFrame({
                'Metric': ['Cloud Cover Percentage'],
                'Value': [f"{cloud_cover_percentage:.2f}%"]
            })
            st.write("Cloud Cover Result:")
            st.table(cloud_cover_result_df)
        else:
            st.write("No original image found for the selected datetime.")

# Main app function
def app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Home", "Sky Image Processing"])

    if page == "Home":
        st.title("Welcome to the Image Processing App")
        st.write("Use the sidebar to navigate through the pages.")
    elif page == "Sky Image Processing":
        sky_image_processing_page()

if __name__ == "__main__":
    app()
