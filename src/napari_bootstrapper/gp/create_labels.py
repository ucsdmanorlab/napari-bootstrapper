import random

import gunpowder as gp
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    maximum_filter,
)
from skimage.measure import label
from skimage.morphology import (
    disk,
    ellipse,
    star,
)
from skimage.segmentation import watershed


class CreateLabels(gp.BatchProvider):
    """
    A provider node for generating synthetic 3D labels arrays.

    Args:
        array_key (gp.ArrayKey): The key of the array to provide labels for.
        anisotropy_range (tuple): The range of anisotropy values to use for label generation.
        shape (tuple): The shape of the labels array.
        dtype (numpy.dtype): The data type of the labels.
        voxel_size (tuple): The voxel size of the labels.
    """

    def __init__(
        self,
        array_key,
        anisotropy_range=None,
        shape=(20, 20, 20),
        dtype=np.uint32,
        voxel_size=None,
    ):
        self.array_key = array_key
        self.anisotropy_range = anisotropy_range
        self.shape = shape
        self.dtype = dtype
        self.voxel_size = voxel_size
        self.ndims = None

    def setup(self):
        spec = gp.ArraySpec()

        if self.voxel_size is None:
            voxel_size = gp.Coordinate((1,) * len(self.shape))
        else:
            voxel_size = gp.Coordinate(self.voxel_size)

        spec.voxel_size = voxel_size
        self.ndims = len(spec.voxel_size)

        if self.anisotropy_range is None:
            self.anisotropy_range = (
                1,
                max(3, int(voxel_size[0] / voxel_size[1])),
            )

        offset = gp.Coordinate((0,) * self.ndims)
        spec.roi = gp.Roi(offset, gp.Coordinate(self.shape) * spec.voxel_size)
        spec.dtype = self.dtype
        spec.interpolatable = False

        self.provides(self.array_key, spec)

    def provide(self, request):
        batch = gp.Batch()

        request_spec = request.array_specs[self.array_key]
        voxel_size = self.spec[self.array_key].voxel_size

        # scale request roi to voxel units
        dataset_roi = request_spec.roi / voxel_size

        # shift request roi into dataset
        dataset_roi = (
            dataset_roi
            - self.spec[self.array_key].roi.get_offset() / voxel_size
        )

        # create array spec
        array_spec = self.spec[self.array_key].copy()
        array_spec.roi = request_spec.roi

        labels = self._generate_labels(dataset_roi.to_slices())

        batch.arrays[self.array_key] = gp.Array(labels, array_spec)

        return batch

    def _generate_labels(self, slices):
        shape = tuple(s.stop - s.start for s in slices)
        labels = np.zeros(shape, self.dtype)
        anisotropy = random.randint(*self.anisotropy_range)
        labels = np.concatenate([labels] * anisotropy)
        shape = labels.shape

        choice = random.choice(["tubes", "random"])

        if choice == "tubes":
            num_points = random.randint(5, 5 * anisotropy)
            for _ in range(num_points):
                z = random.randint(1, labels.shape[0] - 1)
                y = random.randint(1, labels.shape[1] - 1)
                x = random.randint(1, labels.shape[2] - 1)
                labels[z, y, x] = 1

            for z in range(labels.shape[0]):
                dilations = random.randint(1, 10)
                structs = [
                    generate_binary_structure(2, 2),
                    disk(random.randint(1, 4)),
                    star(random.randint(2, 4)),
                    ellipse(random.randint(2, 4), random.randint(2, 4)),
                ]
                dilated = binary_dilation(
                    labels[z],
                    structure=random.choice(structs),
                    iterations=dilations,
                )
                labels[z] = dilated.astype(labels.dtype)

            labels = label(labels)

            distance = labels.shape[0]
            distances, indices = distance_transform_edt(
                labels == 0, return_indices=True
            )
            expanded_labels = np.zeros_like(labels)
            dilate_mask = distances <= distance
            masked_indices = [
                dimension_indices[dilate_mask] for dimension_indices in indices
            ]
            nearest_labels = labels[tuple(masked_indices)]
            expanded_labels[dilate_mask] = nearest_labels
            labels = expanded_labels

            labels[labels == 0] = np.max(labels) + 1
            labels = label(labels)

        if choice == "random":
            np.random.seed()
            peaks = np.random.random(shape).astype(np.float32)
            peaks = gaussian_filter(peaks, sigma=10.0)
            max_filtered = maximum_filter(peaks, 15)
            maxima = max_filtered == peaks
            seeds = label(maxima, connectivity=1)
            labels = watershed(1.0 - peaks, seeds)

        # black out a percentage of label ids
        for divisor in [2, 3, 5]:
            if np.random.random() < 0.1:
                labels[labels % divisor == 0] = 0

        # make random walks come out of existing labels
        labels = self._random_walks(labels)

        # make anisotropic
        labels = labels[::anisotropy].astype(np.uint32)

        return labels

    def _random_walks(self, labels):
        """Generate random walk processes from randomly selected labels using vectorized operations.

        Creates dendrite-like arms that grow out from cell bodies via random walks
        that can branch, invade other labels, and restart when hitting boundaries.

        Args:
            labels: 3D uint array with labeled regions and background (0)

        Returns:
            labels: Modified labels array with random walk processes
        """
        if np.max(labels) == 0:
            return labels

        # Get unique labels (excluding background)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            return labels

        # Parameters for random walks
        num_walks_per_label = random.randint(1, 3)
        walk_length_range = (50, 120)
        branch_probability = 0.5
        max_branches = 5
        walk_thickness_range = (5, 20)  # Thickness of walks in pixels

        # 3D movement directions (26-connectivity)
        directions = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if not (dz == 0 and dy == 0 and dx == 0):
                        directions.append((dz, dy, dx))
        directions = np.array(directions)

        shape = labels.shape

        # generate random walks
        for label_id in random.choices(unique_labels, k=num_walks_per_label):

            # Find boundary positions of this label
            label_mask = labels == label_id
            if not np.any(label_mask):
                continue

            # Find boundary voxels by checking for adjacent non-label voxels
            boundary_mask = np.zeros_like(label_mask)

            # Check all 3 face-adjacent directions for boundary detection
            for dz, dy, dx in [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]:
                # Shift the label mask
                shifted_mask = np.zeros_like(label_mask)

                # Handle boundaries properly
                z_slice = slice(
                    max(0, dz),
                    min(shape[0], shape[0] + dz) if dz < 0 else None,
                )
                y_slice = slice(
                    max(0, dy),
                    min(shape[1], shape[1] + dy) if dy < 0 else None,
                )
                x_slice = slice(
                    max(0, dx),
                    min(shape[2], shape[2] + dx) if dx < 0 else None,
                )

                z_src = slice(
                    max(0, -dz),
                    min(shape[0], shape[0] - dz) if dz > 0 else None,
                )
                y_src = slice(
                    max(0, -dy),
                    min(shape[1], shape[1] - dy) if dy > 0 else None,
                )
                x_src = slice(
                    max(0, -dx),
                    min(shape[2], shape[2] - dx) if dx > 0 else None,
                )

                shifted_mask[z_slice, y_slice, x_slice] = label_mask[
                    z_src, y_src, x_src
                ]

                # Boundary voxels are those in the label that have non-label neighbors
                boundary_mask |= label_mask & ~shifted_mask

            # Get boundary coordinates
            boundary_positions = np.where(boundary_mask)
            if len(boundary_positions[0]) == 0:
                # Fallback: if no boundary found (single voxel), use all positions
                boundary_positions = np.where(label_mask)

            boundary_coords = list(
                zip(
                    boundary_positions[0],
                    boundary_positions[1],
                    boundary_positions[2],
                    strict=False,
                )
            )

            for _ in range(num_walks_per_label):
                if not boundary_coords:
                    break

                # Start from a random boundary position
                start_pos = random.choice(boundary_coords)
                current_pos = np.array(start_pos)

                # Generate walk with total remaining steps
                total_steps = random.randint(*walk_length_range)
                walk_thickness = random.randint(*walk_thickness_range)
                active_walks = [
                    (current_pos.copy(), total_steps, walk_thickness)
                ]

                while active_walks:
                    new_walks = []

                    for pos, remaining_steps, thickness in active_walks:
                        if remaining_steps <= 0:
                            continue

                        # Choose random direction
                        direction = directions[
                            random.randint(0, len(directions) - 1)
                        ]
                        new_pos = pos + direction

                        # Check if out of bounds
                        if (
                            new_pos[0] < 0
                            or new_pos[0] >= shape[0]
                            or new_pos[1] < 0
                            or new_pos[1] >= shape[1]
                            or new_pos[2] < 0
                            or new_pos[2] >= shape[2]
                        ):

                            # Out of bounds - restart walk from a random boundary position
                            if remaining_steps > 1 and boundary_coords:
                                restart_pos = random.choice(boundary_coords)
                                new_walks.append(
                                    (
                                        np.array(restart_pos),
                                        remaining_steps - 1,
                                        thickness,
                                    )
                                )
                            continue

                        # Always invade - set position to current label_id with thickness
                        self._paint_thick_voxel(
                            labels, new_pos, label_id, thickness, shape
                        )

                        # Continue this walk
                        new_walks.append(
                            (new_pos, remaining_steps - 1, thickness)
                        )

                        # Chance to branch
                        if (
                            random.random() < branch_probability
                            and remaining_steps > 5
                            and len(new_walks) < max_branches
                        ):

                            # Create a branch with shorter length and potentially different thickness
                            branch_length = min(
                                remaining_steps // 2, random.randint(5, 15)
                            )
                            branch_thickness = random.randint(
                                *walk_thickness_range
                            )
                            new_walks.append(
                                (
                                    new_pos.copy(),
                                    branch_length,
                                    branch_thickness,
                                )
                            )

                    active_walks = new_walks

        return labels

    def _paint_thick_voxel(
        self, labels, center_pos, label_id, thickness, shape
    ):
        """Paint a thick spherical region around a center position.

        Args:
            labels: 3D array to modify
            center_pos: Center position (z, y, x)
            label_id: Label value to paint
            thickness: Radius of the thick region
            shape: Shape of the labels array
        """
        z_center, y_center, x_center = center_pos
        radius = thickness // 2

        # Define bounds for the thick region
        z_min = max(0, z_center - radius)
        z_max = min(shape[0], z_center + radius + 1)
        y_min = max(0, y_center - radius)
        y_max = min(shape[1], y_center + radius + 1)
        x_min = max(0, x_center - radius)
        x_max = min(shape[2], x_center + radius + 1)

        # Create spherical pattern within the region
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # Calculate distance from center
                    dist_sq = (
                        (z - z_center) ** 2
                        + (y - y_center) ** 2
                        + (x - x_center) ** 2
                    )
                    if dist_sq <= radius**2:
                        labels[z, y, x] = label_id
