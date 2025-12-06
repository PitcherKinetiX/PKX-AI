# mapping_table.py
# Joint indices follow COCO format used in your extractor.

FEATURE_TO_JOINT = {
    0: 7,    # Left elbow angle
    1: 8,    # Right elbow angle
    2: 5,    # Left shoulder angle
    3: 6,    # Right shoulder angle
    4: 11,   # Left hip angle
    5: 12,   # Right hip angle
    6: 13,   # Left knee angle
    7: 14,   # Right knee angle

    8: 14,   # Right knee extension velocity
    9: 12,   # Pelvis rotation velocity
    10: 6,   # Trunk rotation velocity
    11: 8,   # Right elbow extension velocity
    12: 10   # Shoulder internal rotation velocity
}
