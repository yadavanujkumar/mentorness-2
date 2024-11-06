# import matplotlib.pyplot as plt
# import numpy as np

# # Data for the bar chart
# categories = ['Cuboid Proposal Network (CPN)', 'Short Tubelet Detection', 'Tubelet Linking', 
#               'AlexNet', 'GoogleNet', 'Inception-ResNet-V2']
# accuracy = [85, 87, 92, 75, 85, 97]
# processing_speed = [80, 78, 85, 90, 85, 80]

# x = np.arange(len(categories))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots(figsize=(12, 8))

# # Define hatch patterns for each bar set
# bar1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', hatch='//')
# bar2 = ax.bar(x + width/2, processing_speed, width, label='Processing Speed (%)', hatch='..')

# # Adding labels, title, and custom x-axis tick labels, etc.
# ax.set_xlabel('Object Detection Techniques and CNN Models')
# ax.set_ylabel('Performance (%)')
# ax.set_title('Comparison of Object Detection Techniques and CNN Models by Accuracy and Processing Speed')
# ax.set_xticks(x)
# ax.set_xticklabels(categories, rotation=45, ha="right")
# ax.legend()

# # Adding percentage labels on top of each bar
# def add_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height}%',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# add_labels(bar1)
# add_labels(bar2)

# plt.tight_layout()
# plt.show()



import matplotlib.pyplot as plt
import pandas as pd

# Data for face detection techniques
data = {
    "Technique": [
        "Viola-Jones",
        "CNNs",
        "YOLO",
        "RetinaFace",
        "Dlib's HOG + SVM"
    ],
    "Speed (1-10)": [9, 6, 10, 5, 7],
    "Accuracy (1-10)": [5, 9, 7, 10, 6],
    "Resource Requirements (1-10)": [8, 4, 5, 2, 6],
    "Best Use Case": [
        "Access Control",
        "Public Surveillance",
        "Traffic Monitoring",
        "Criminal Identification",
        "Lightweight Tasks"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the data
ax.barh(df["Technique"], df["Speed (1-10)"], color='skyblue', label='Speed')
ax.barh(df["Technique"], df["Accuracy (1-10)"], left=df["Speed (1-10)"], color='lightgreen', label='Accuracy')
ax.barh(df["Technique"], df["Resource Requirements (1-10)"], left=df["Speed (1-10)"] + df["Accuracy (1-10)"], color='salmon', label='Resource Requirements')

# Adding annotations for clarity
for index, value in enumerate(df["Speed (1-10)"]):
    ax.text(value + 0.2, index, str(value), va='center', color='black')
for index, value in enumerate(df["Accuracy (1-10)"]):
    ax.text(df["Speed (1-10)"][index] + value + 0.2, index, str(value), va='center', color='black')
for index, value in enumerate(df["Resource Requirements (1-10)"]):
    ax.text(df["Speed (1-10)"][index] + df["Accuracy (1-10)"][index] + value + 0.2, index, str(value), va='center', color='black')

# Setting labels and title
ax.set_xlabel('Rating (1-10)')
ax.set_title('Comparison of Face Detection Techniques')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
