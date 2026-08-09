import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN


def cluster_edges_with_dbscan(
    output_path,
    edges,
    eps=5,
    min_samples=5,
    min_cluster_size=10,
    save_label_map=True,
    prefix="image"
):
    """
    DBSCAN clustering on Canny edge pixels.

    Returns:
        clustered_img:
            RGB visualization of clustered edge points.

        cluster_label_map:
            2D uint16 label image.
            0 = background / noise
            1, 2, 3... = valid DBSCAN clusters

        cluster_info:
            list of cluster-level information.
    """

    y_coords, x_coords = np.where(edges > 0)

    if len(x_coords) == 0:
        print("No edge points found.")

        clustered_img = np.zeros((*edges.shape, 3), dtype=np.uint8)
        cluster_label_map = np.zeros(edges.shape, dtype=np.uint16)
        cluster_info = []

        return clustered_img, cluster_label_map, cluster_info

    edge_points = np.column_stack((x_coords, y_coords))

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    dbscan_labels = dbscan.fit_predict(edge_points)

    clustered_img = np.zeros((*edges.shape, 3), dtype=np.uint8)
    clustered_img_with_text = np.zeros((*edges.shape, 3), dtype=np.uint8)

    cluster_label_map = np.zeros(edges.shape, dtype=np.uint16)

    unique_labels = sorted(set(dbscan_labels))

    # valid_labels = [
    #     lb for lb in unique_labels
    #     if lb != -1 and np.sum(dbscan_labels == lb) >= min_cluster_size
    # ]
    #
    # colors = plt.cm.get_cmap("tab20", max(len(valid_labels), 1))

    valid_labels = [
        lb for lb in unique_labels
        if lb != -1 and np.sum(dbscan_labels == lb) >= min_cluster_size
    ]

    colors = plt.cm.get_cmap("tab20", max(len(valid_labels), 1))

    # can showing the removed point as white，but not write into label map
    valid_label_set = set(valid_labels)

    noise_like_mask = np.array([
        (lb == -1) or (lb not in valid_label_set)
        for lb in dbscan_labels
    ])

    noise_points = edge_points[noise_like_mask]

    if len(noise_points) > 0:
        nx = noise_points[:, 0]
        ny = noise_points[:, 1]

        clustered_img[ny, nx] = (255, 255, 255)
        clustered_img_with_text[ny, nx] = (255, 255, 255)



    cluster_info = []
    new_label_id = 1

    for color_idx, old_label in enumerate(valid_labels):
        cluster_mask_points = dbscan_labels == old_label

        points = edge_points[cluster_mask_points]

        x = points[:, 0]
        y = points[:, 1]

        points_in_cluster = len(points)

        color = np.array(colors(color_idx)[:3]) * 255
        color = color.astype(np.uint8)

        clustered_img[y, x] = color
        clustered_img_with_text[y, x] = color

        # Write DBSCAN cluster ID back to label map original size
        cluster_label_map[y, x] = new_label_id

        cluster_center = np.mean(points, axis=0).astype(int)

        cv2.putText(
            clustered_img_with_text,
            str(new_label_id),
            (cluster_center[0], cluster_center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cluster_info.append({
            "new_label_id": int(new_label_id),
            "dbscan_original_label": int(old_label),
            "num_points": int(points_in_cluster),
            "center_x": int(cluster_center[0]),
            "center_y": int(cluster_center[1]),
            "bbox_xmin": int(np.min(x)),
            "bbox_ymin": int(np.min(y)),
            "bbox_xmax": int(np.max(x)),
            "bbox_ymax": int(np.max(y))
        })

        new_label_id += 1

    os.makedirs(output_path, exist_ok=True)

    # cv2.imwrite(
    #     os.path.join(output_path, f"{prefix}_clustered_img.tiff"),
    #     cv2.cvtColor(clustered_img_with_text, cv2.COLOR_RGB2BGR)
    # )

    # if save_label_map:
    #     cv2.imwrite(
    #         os.path.join(output_path, f"{prefix}_cluster_label_map.tiff"),
    #         cluster_label_map.astype(np.uint16)
    #     )
    #
    #     np.save(
    #         os.path.join(output_path, f"{prefix}_cluster_label_map.npy"),
    #         cluster_label_map
    #     )
    #
    #     np.save(
    #         os.path.join(output_path, f"{prefix}_cluster_info.npy"),
    #         np.array(cluster_info, dtype=object)
    #     )

    print(f"Number of valid clusters: {len(cluster_info)}")

    return clustered_img, cluster_label_map, cluster_info
