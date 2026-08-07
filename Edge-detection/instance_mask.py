import cv2
import numpy as np

def make_instance_mask_from_cluster_label_map(
    cluster_label_map,
    display_image,
    min_area=50,
    max_area=1000000,
    keep_original_cluster_id=False
):
    """
    直接使用 DBSCAN cluster_label_map 生成 instance mask。

    cluster_label_map:
        0 = background / noise
        1, 2, 3... = DBSCAN cluster id

    instance_mask:
        0 = background
        1, 2, 3... = instance id
    """

    height, width = cluster_label_map.shape

    instance_mask = np.zeros(
        (height, width),
        dtype=np.uint16
    )

    overlay = display_image.copy()

    kernel = np.ones((3, 3), np.uint8)

    cluster_ids = np.unique(cluster_label_map)
    cluster_ids = cluster_ids[cluster_ids > 0]

    new_instance_id = 1

    for cluster_id in cluster_ids:

        #  DBSCAN cluster related with edge mask
        single_cluster_edge = (
            cluster_label_map == cluster_id
        ).astype(np.uint8) * 255

        # Morphological connectivity of individual clusters
        single_cluster_edge = cv2.morphologyEx(
            single_cluster_edge,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        single_cluster_edge = cv2.dilate(
            single_cluster_edge,
            kernel,
            iterations=1
        )

        ys, xs = np.where(single_cluster_edge > 0)

        if len(xs) == 0:
            continue

        minr = int(np.min(ys))
        maxr = int(np.max(ys)) + 1
        minc = int(np.min(xs))
        maxc = int(np.max(xs)) + 1

        touches_border = (
            minr <= 0 or
            minc <= 0 or
            maxr >= height or
            maxc >= width
        )

        if touches_border:
            filled_region = fill_closed_or_border_touching_edge(
                single_cluster_edge
            )
        else:
            contours, _ = cv2.findContours(
                single_cluster_edge,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            filled_region = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            cv2.drawContours(
                filled_region,
                contours,
                contourIdx=-1,
                color=255,
                thickness=cv2.FILLED
            )

        area = int(np.sum(filled_region > 0))

        if area < min_area:
            continue

        if area > max_area:
            continue

        if keep_original_cluster_id:
            instance_id = int(cluster_id)
        else:
            instance_id = int(new_instance_id)

        # Avoid overwriting
        write_mask = (filled_region > 0) & (instance_mask == 0)

        if np.sum(write_mask) == 0:
            continue

        instance_mask[write_mask] = instance_id

        contours, _ = cv2.findContours(
            filled_region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 255, 0),
            2
        )

        if not keep_original_cluster_id:
            new_instance_id += 1

    binary_mask = (instance_mask > 0).astype(np.uint8) * 255

    return instance_mask, binary_mask, overlay



def fill_closed_or_border_touching_edge(edge_mask):
    """
    Fill edge mask.
    If edge touches image border, use image boundary to close the object.

    edge_mask:
        uint8 binary image, edge = 255, background = 0

    return:
        filled uint8 mask, foreground = 255
    """

    edge_mask = (edge_mask > 0).astype(np.uint8) * 255

    h, w = edge_mask.shape

    kernel = np.ones((3, 3), np.uint8)

    edge_closed = cv2.morphologyEx(
        edge_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    edge_closed = cv2.dilate(
        edge_closed,
        kernel,
        iterations=1
    )

    flood = edge_closed.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    seeds = [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1)
    ]

    for seed in seeds:
        x, y = seed

        if flood[y, x] == 0:
            cv2.floodFill(
                flood,
                ff_mask,
                seedPoint=seed,
                newVal=255
            )

    filled_inside = cv2.bitwise_not(flood)

    filled = cv2.bitwise_or(
        filled_inside,
        edge_closed
    )

    return filled