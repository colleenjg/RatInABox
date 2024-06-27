import copy
import itertools
import pprint
import warnings

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from ratinabox import Environment, utils

class EnvironmentWarning(UserWarning):
    pass

warnings.simplefilter("once", EnvironmentWarning)

class ParamsManagerMixin:
    """
    Contributer: Colleen Gillon c.gillon@imperial.ac.uk
    Date: 06/27/2024

    This mixin handles ignored and fixed parameters for classes like OpenField (defined below).
    """
    @property
    def have_ignored_params_been_checked(self):
        if not hasattr(self, "_have_ignored_params_been_checked"):
            self._have_ignored_params_been_checked = False
        return self._have_ignored_params_been_checked

    @classmethod
    def check_ignored_params_for_class(cls, params):
        """Check if parameters are ignored.

        Args:
            params (dict): parameters to check

        Raises:
            KeyError: if parameters are not ignored
        """

        # collect all class parameters for the specified dictionary name
        all_ignored_params_dict = utils.collect_all_params(
            cls, dict_name="ignored_params"
        )

        for key in all_ignored_params_dict:
            if key in params.keys():
                warnings.warn(
                    f"'{key}' should not be provided for {cls.__name__}. "
                    "Will be ignored."
                )

    def check_if_ignored_params(self, params):
        """Check if parameters are ignored.

        Args:
            params (dict): parameters to check

        Raises:
            KeyError: if parameters are not ignored
        """

        if hasattr(self, "_have_ignored_params_been_checked"):
            return

        self.check_ignored_params_for_class(params)
        self._have_ignored_params_been_checked = True

    @classmethod
    def get_all_fixed_params(cls, verbose=False):
        """Returns a dictionary of all the default parameters of the class, including
        those inherited from its parents.

        Args:
            verbose (bool, optional): If True, prints the parameters. Defaults to False.

        Returns:
            dict: dictionary of all the default parameters of the class, including

        """
        all_fixed_params = dict()
        all_fixed_params.update(
            utils.collect_all_params(cls, dict_name="fixed_params")
        )
        if verbose:
            pprint.pprint(all_fixed_params)
        return all_fixed_params

    def add_fixed_params(self, params=dict()):
        """Sets fixed parameters for the class."""

        all_fixed_params = self.get_all_fixed_params()

        params = copy.copy(
            params
        )  # avoid deep copy to preserve reference to input layers
        for key, value in all_fixed_params.items():
            if key in params.keys() and value != params[key]:
                raise ValueError(
                    f"'{key}' parameter should not be passed, unless it is set to "
                    f"'{value}'."
                )
            params[key] = value

        return params


class OpenField(Environment, ParamsManagerMixin):
    """
    Contributer: Colleen Gillon c.gillon@imperial.ac.uk
    Date: 06/27/2024

    The OpenField class is a subclass of Environment() and inherits its properties/plotting functions.

    OpenField ....

    Note that it only works with solid boundary conditions, and no holes.

    The following parameters can be provided:
        • init_random_reward_obj (default 1): number of reward objects to add, randomly
        • init_random_novel_obj (default 5): number of novel objects to add, randomly
        • init_random_walls (default 5): number of walls to add, randomly
        • init_random_teleport_pairs (default 2): number of teleportation pairs to add, randomly
        • wall_lengths (default [0.1, 0.2]): range of wall lengths to sample from
        • min_dist (default 0.1): minimum distance between objects (walls is half)
        • init_seed (default None): random seed for reproducibility

    List of properties:
        • object_df_columns
        • object_df
        • object_type_num_to_name_dict
        • type_name_to_num_dict
        • type_num_to_plot_params_dict
        • teleport_pairs_dict

    List of functions:
        • add_object()
        • add_reward_objects()
        • add_novel_objects()
        • add_teleport_pairs()
        • get_teleport_coords()
        • get_teleport_pair_orientation()
        • get_dist_from_coords_to_closest_object()
        • get_dist_from_coords_to_closest_wall()
        • plot_environment()
        • sample_coords()
    """

    default_params = {
        "init_random_reward_obj": 1,
        "init_random_novel_obj": 5,
        "init_random_walls": 5,
        "init_random_teleport_pairs": 2,
        "wall_lengths": [0.1, 0.2],
        "min_dist": 0.1,  # between objects (walls is half)
        "init_seed": None,
    }

    ignored_param_keys = list()
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "holes": [],  # no holes,
        "boundary": None,
    }

    def __init__(self, params=dict()):
        """Initialize the environment."""

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(self.params)

        if min(self.wall_lengths) <= 0:
            raise ValueError("Wall lengths must be positive.")

        self.num_teleport_pairs = 0

        if self.init_seed is None:
            self.rng = np.random
        else:
            self.rng = np.random.RandomState(self.init_seed)

        self.add_reward_objects(self.init_random_reward_obj)
        self.add_novel_objects(self.init_random_novel_obj)
        self.add_teleport_pairs(self.init_random_teleport_pairs)
        self.add_walls(self.init_random_walls)

    @property
    def object_df_columns(self):
        if not hasattr(self, "_object_df_columns"):
            self._object_df_columns = [
                "object_type_num",
                "object_type_name",
                "idx_within_type",
                "position_x",
                "position_y",
                "teleport_pair_num",
                "teleport_direction",
            ]

        return self._object_df_columns

    @property
    def object_df(self):
        if not hasattr(self, "_object_df"):
            object_df = pd.DataFrame(columns=self.object_df_columns)
            self._object_df = object_df

        return self._object_df

    def get_new_teleport_pair_object_type_nums(self, first=None):
        """Get object type numbers for a new teleport pair.

        Args:
            first (int): First object type number to use. If None, use the next
                available number. Defaults to None.

        Returns:
            object_type_nums (dict): Dictionary of object type numbers for the
                teleport pair.
        """

        if first is None:
            first = np.max(list(self.object_type_num_to_name_dict.keys())) + 1

        first = int(first)

        object_type_nums = {
            "in": first,
            "out": first + 1,
        }

        return object_type_nums

    def reset_object_type_dicts(self):
        """Reset the object type dictionaries."""

        dict_attr_names = [
            "_object_type_num_to_name_dict",
            "_type_num_to_plot_params_dict",
            "_teleport_pairs_dict",
        ]

        for dict_attr_name in dict_attr_names:
            if hasattr(self, dict_attr_name):
                delattr(self, dict_attr_name)

    @property
    def object_type_num_to_name_dict(self):
        """Dictionary for getting object type name from number."""

        if not hasattr(self, "_object_type_num_to_name_dict"):
            object_type_num_to_name_dict = {
                0: "reward",
                1: "novel",
            }

            for n in range(self.num_teleport_pairs):
                object_type_nums = self.get_new_teleport_pair_object_type_nums(
                    first=np.max(list(object_type_num_to_name_dict.keys())) + 1
                )
                for direction, i in object_type_nums.items():
                    object_type_num_to_name_dict[i] = f"teleport_{n}_{direction}"
            self._object_type_num_to_name_dict = object_type_num_to_name_dict

        return self._object_type_num_to_name_dict

    @property
    def type_name_to_num_dict(self):
        """Dictionary for getting object type number from name."""

        object_type_name_to_num_dict = {
            val: key for key, val in self.object_type_num_to_name_dict.items()
        }

        return object_type_name_to_num_dict

    @property
    def type_num_to_plot_params_dict(self):
        """Dictionary for getting object type number from name."""

        if not hasattr(self, "_type_num_to_plot_params_dict"):
            teleport_nums = [
                val.replace("teleport_", "").replace("in_", "")
                for val in self.object_type_num_to_name_dict.values()
                if val.startswith("teleport") and "_in" in val
            ]
            teleport_vals = np.linspace(0.5, 1, len(teleport_nums))
            teleport_colors = plt.get_cmap("Oranges")(teleport_vals)

            type_num_to_plot_params_dict = dict()
            for num, name in self.object_type_num_to_name_dict.items():
                if name == "reward":
                    type_num_to_plot_params_dict[num] = {
                        "name": name,
                        "marker": "o",
                        "color": "blue",
                        "s": 20,
                        "zorder": 5,
                    }
                elif name == "novel":
                    type_num_to_plot_params_dict[num] = {
                        "name": name,
                        "marker": "o",
                        "color": "green",
                        "s": 20,
                        "zorder": 5,
                    }
                elif name.startswith("teleport"):
                    direc = "in" if "_in" in name else "out"
                    teleport_num = int(
                        name.replace("teleport_", "").replace(f"_{direc}", "")
                    )
                    color = teleport_colors[teleport_num]
                    type_num_to_plot_params_dict[num] = {
                        "name": name,
                        "marker": self.get_teleport_pair_marker(
                            teleport_num, direction=direc
                        ),
                        "color": color,
                        "s": 20,
                        "zorder": 5,
                    }
                else:
                    raise ValueError(f"Unknown object type name: {name}")

            self._type_num_to_plot_params_dict = type_num_to_plot_params_dict

        return self._type_num_to_plot_params_dict

    def get_teleport_coords(self, teleport_pair_num, direction="in"):
        """Get the teleport coordinates for the given teleport pair.

        Args:
            teleport_pair_num (int): The teleport pair to get the coordinates for.
            direction (str, optional): The direction to get the coordinates for.
                Defaults to "in".

        Returns:
            np.ndarray: The teleport coordinates.
        """

        teleport_coords = self.teleport_pairs_dict[teleport_pair_num][direction][1]

        return teleport_coords

    def get_teleport_pair_orientation(self, teleport_pair_num=1):
        """Get the orientation of a teleport pair.

        Args:
            teleport_pair_num (int): teleport pair number.

        Returns:
            str: orientation of the teleport pair.
        """

        if teleport_pair_num % 2 == 0:
            orientation = "vertical"
        else:
            orientation = "horizontal"

        return orientation

    def get_number_object_types_split(self):
        """Get the number of each object type.

        Returns:
            tuple: number of novel, reward, and teleport objects.
        """

        num_novel, num_reward, num_teleport = 0, 0, 0
        for object_type in self.objects["object_types"]:
            object_name = self.object_type_num_to_name_dict[object_type]
            if object_name == "novel":
                num_novel += 1
            elif object_name == "reward":
                num_reward += 1
            elif "teleport" in object_name:
                num_teleport += 1

        if num_teleport % 2:
            raise RuntimeError("Number of teleport pairs should be even.")

        return num_novel, num_reward, num_teleport

    def get_teleport_pair_marker(self, teleport_pair_num=1, direction="in"):
        """Get the orientation of a teleport pair.

        Args:
            teleport_pair_num (int): teleport pair number.

        Returns:
            str: orientation of the teleport pair.
        """

        orientation = self.get_teleport_pair_orientation(teleport_pair_num)

        if orientation == "vertical":
            marker = "v" if direction == "in" else "^"
        else:
            marker = "<" if direction == "in" else ">"

        return marker

    def add_fixed_params(self, params=dict()):
        """Sets fixed parameters."""

        all_fixed_params = self.get_all_fixed_params()

        params = copy.copy(
            params
        )  # avoid deep copy to preserve reference to input layers
        for key, value in all_fixed_params.items():
            if key in params.keys() and value != params[key]:
                raise ValueError(
                    f"'{key}' parameter should not be passed, unless it is set to "
                    f"'{value}'."
                )

            params[key] = value

        return params

    def get_dist_from_coords_to_closest_object(self, coords):
        """Get the distance from a set of coordinates to the closest objects.

        Args:
            coords (np.ndarray): coordinates to get distance from.

        Returns:
            float: closest distance.
        """

        if len(self.objects["objects"]) == 0:
            return np.inf

        closest_distances = list()
        for object_coords in self.objects["objects"]:
            closest_distance = np.linalg.norm(coords - np.asarray(object_coords), ord=2)
            closest_distances.append(closest_distance)

        closest_dist = float(np.min(closest_distances))

        return closest_dist

    def get_dist_from_coords_to_closest_wall(self, coords):
        """Get the distance from a set of coordinates to the closest wall.

        Args:
            coords (np.ndarray): coordinates to get distance from.

        Returns:
            float: closest distance.
        """

        if len(self.walls) == 0:
            return np.inf

        # returns points (1) x vectors x coords
        closest_dist = float(
            np.min(shortest_distances_from_points_to_lines(coords, self.walls))
        )

        return closest_dist

    def sample_coords(
        self, min_dist: float | None = None, max_attempts: int = 1000
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Sample coordinates situated at least min_dist from the closest
        object (half for walls).

        Args:
            min_dist (float, optional): minimum distance to closest object or
                wall. Defaults to None.
            max_attempts (int, optional): maximum number of attempts to sample
                valid coordinates. Defaults to 1000.

        Raises:
            ValueError: if could not sample valid coordinates after max_attempts
                attempts.

        Returns:
            coords (1d array): sampled coordinates [x, y].
        """

        if min_dist is None:
            min_dist = float(self.min_dist)

        i = 0
        while True:
            x = self.rng.uniform(self.extent[0], self.extent[1])
            y = self.rng.uniform(self.extent[2], self.extent[3])

            coords = np.array([x, y])

            # check distance to objects, then walls
            if self.get_dist_from_coords_to_closest_object(coords) >= min_dist:
                if self.get_dist_from_coords_to_closest_wall(coords) >= min_dist / 2:
                    break
            if i > max_attempts:
                raise ValueError(
                    "Could not sample valid coordinates situated at least "
                    f"{min_dist} from the closest objects (or half for walls)."
                )
            i += 1

        return coords

    def sample_wall_end(
        self,
        start_coords: np.ndarray[tuple[int], np.dtype[np.float64]],
        min_dist: float | None = None,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """Sample valid coordinates for the end of a wall given the start coordinates.

        Args:
            start_coords (1d array): start of wall.
            min_dist (float, optional): minimum distance to closest object.
                Defaults to None.

        Returns:
            end_coords (1d array): sampled end of wall coordinates [x, y].
                Returns None if could not sample valid end coordinates.
        """

        if not self.check_if_position_is_in_environment(start_coords):
            return None

        if min_dist is None:
            min_dist = float(self.min_dist / 2)

        # sample wall length
        wall_length = self.rng.uniform(*self.wall_lengths)

        # sample orientation + direction, then cycle through if needed, before
        # abandoning each time check that the wall's max distance from another
        # objects is reasonable.
        wall_orientations = ["x", "y"]
        wall_directions = [-1, 1]
        wall_ori_direcs = list(itertools.product(wall_orientations, wall_directions))

        shuffle_order = np.arange(len(wall_ori_direcs))
        self.rng.shuffle(shuffle_order)
        wall_ori_direcs = [wall_ori_direcs[i] for i in shuffle_order]

        end_coords = None
        for wall_ori, wall_direc in wall_ori_direcs:
            c = 0 if wall_ori == "x" else 1
            end_coords = np.array(start_coords)  # new array
            end_coords[c] += wall_length * wall_direc

            # check that end_coords are within bounds
            if not self.check_if_position_is_in_environment(end_coords):
                end_coords = None

            # check that end_coords are far enough from objects, if there are any
            if end_coords is not None and len(self.objects["objects"]) != 0:
                closest_dist = np.min(
                    shortest_distances_from_points_to_lines(
                        self.objects["objects"], [start_coords, end_coords]
                    )
                )

                if closest_dist < min_dist:
                    end_coords = None

            if end_coords is not None:
                break

        return end_coords

    def add_object(self, object, object_type="new"):
        super().add_object(object, type=object_type)

        # add to object dataframe
        sub_df = self.object_df[self.object_df["object_type_num"] == object_type]
        if len(sub_df) == 0:
            idx_within_type = 0
        else:
            idx_within_type = sub_df["idx_within_type"].max() + 1

        object_type_name = self.object_type_num_to_name_dict[int(object_type)]

        new_object = {
            "object_type_num": object_type,
            "object_type_name": object_type_name,
            "idx_within_type": idx_within_type,
            "position_x": object[0],
            "position_y": object[1],
        }

        if "teleport" in object_type_name:
            _, teleport_pair_num, teleport_direction = object_type_name.split("_")
            new_object["teleport_pair_num"] = int(teleport_pair_num)
            new_object["teleport_direction"] = teleport_direction

        self.object_df.loc[len(self.object_df)] = new_object

    def add_reward_objects(self, num=1, coords=None):
        """Add reward objects.

        Args:
            num (int, optional): number of reward objects to add. Defaults to 1.
        """

        reward_type = self.type_name_to_num_dict["reward"]

        if coords is not None:
            num = len(coords)

        for n in range(num):
            if coords is None:
                coord = self.sample_coords()
            else:
                coord = np.asarray(coords[n], dtype=np.float64).reshape(2)
                self.check_if_position_is_in_environment(coord)
            self.add_object(coord, object_type=reward_type)

        if num > 0:
            self.reset_object_type_dicts()

    def add_novel_objects(self, num=1, coords=None):
        """Add novel objects.

        Args:
            num (int, optional): number of novel objects to add. Defaults to 1.
        """

        novel_type = self.type_name_to_num_dict["novel"]

        if coords is not None:
            num = len(coords)

        for n in range(num):
            if coords is None:
                coord = self.sample_coords()
            else:
                coord = np.asarray(coords[n], dtype=np.float64).reshape(2)
                self.check_if_position_is_in_environment(coord)

            self.add_object(coord, object_type=novel_type)

        if num > 0:
            self.reset_object_type_dicts()

    def add_teleport_pairs(self, num=1, coord_pairs=None):
        """Add teleport pairs (directional).

        Args:
            num (int, optional): number of teleport pairs to add. Defaults to 1.
        """

        if coord_pairs is not None:
            num = len(coord_pairs)

        def format_teleport_pair(coord_pair):
            try:
                coords_in, coords_out = coord_pair
            except ValueError as err:
                if "values to unpack" in str(err):
                    raise ValueError(
                        f"Expected two coordinates per teleport pair, but got {len(coords)}."
                    )
                elif "unpack non-iterable" in str(err):
                    raise ValueError(
                        f"Each coordinate pair must be an iterable of length 2."
                    )
            coord_pair = [coords_in, coords_out]
            for c in range(2):
                coord_pair[c] = np.asarray(coord_pair[c], dtype=np.float64).reshape(2)
                self.check_if_position_is_in_environment(coord_pair[c])

            return coord_pair

        for n in range(num):
            object_type_nums = self.get_new_teleport_pair_object_type_nums()
            self.num_teleport_pairs += 1
            self.reset_object_type_dicts()  # within loop, so that teleport pair object types are not reused
            if coord_pairs is not None:
                coord_pair = format_teleport_pair(coord_pairs[n])
            for o, object_type_num in enumerate(object_type_nums.values()):
                if coord_pairs is None:
                    coords = self.sample_coords()
                else:
                    coords = coord_pair[o]
                self.add_object(coords, object_type=object_type_num)

    @property
    def teleport_pairs_dict(self):
        """Returns dictionary of teleport pairs (directional)."""

        if not hasattr(self, "_teleport_pairs_dict"):
            teleport_pairs_dict = dict()
            for name, object_type in self.type_name_to_num_dict.items():
                if name.startswith("teleport_") and "in" in name:
                    object_type_in = object_type
                    teleport_pair = int(
                        name.replace("teleport_", "").replace("_in", "")
                    )
                    out_key = f"teleport_{teleport_pair}_out"
                    if out_key not in self.type_name_to_num_dict.keys():
                        raise RuntimeError(
                            f"Teleport in {teleport_pair} does not have 'out' pair."
                        )
                    object_type_out = self.type_name_to_num_dict[out_key]

                    coords = list()
                    for object_type in [object_type_in, object_type_out]:
                        object_idxs = np.where(
                            self.objects["object_types"] == object_type
                        )[0]
                        if len(object_idxs) != 1:
                            raise RuntimeError(
                                f"Expected teleport in {teleport_pair} to correspond "
                                f"to exactly one object, but found {len(object_idxs)}."
                            )
                        coords.append(self.objects["objects"][object_idxs[0]])

                    teleport_pairs_dict[teleport_pair] = {
                        "in": (object_type_in, coords[0]),
                        "out": (object_type_out, coords[1]),
                    }

            self._teleport_pairs_dict = teleport_pairs_dict

        return self._teleport_pairs_dict

    def check_if_walls_ends_too_close(self, new_wall_coords, min_dist=None):
        """
        Checks whether a new wall's ends is too close to the ends of existing
        walls.

        Specifically checks whether an end of the new wall intersects at less
        than 45 degrees near the end of an existing wall, forming an V shape
        with small ends sticking out. If so, returns True, else False.

        Does NOT check whether the new wall overlaps exactly with an existing
        wall, or intersects near the middle of either wall.

        Args:
            new_wall_coords (list or 2D array): coordinates of new wall,
                with dims [[x1, y1], [x2, y2]]
            min_dist (float, optional): minimum distance between walls.
                Defaults to None.

        Returns:
            bool: True if the ends of a new wall are too close to an existing wall,
                else False.
        """

        if len(self.walls) == 0:
            return False

        if min_dist is None:
            min_dist = float(self.min_dist)

        new_wall = np.asarray(new_wall_coords)

        for wall in self.walls:
            # get angle between two vectors
            angle = get_angle_between_vectors(
                np.diff(new_wall_coords, axis=0)[0], np.diff(wall, axis=0)[0]
            )

            if angle > 45:
                continue

            # if angle is less than 45 degrees, check any ends of the walls are too
            # close together
            distances, coords = list(), []
            for c1, c2 in itertools.product([0, 1], [0, 1]):
                coords.append([c1, c2])
                distances.append(np.linalg.norm(wall[c1] - new_wall[c2], ord=2))

            order = np.argsort(distances)

            if distances[order[0]] < min_dist:
                # farther must be at least as far as if the walls
                # intersected only at their ends (no intersection)
                farthest = distances[order[-1]]
                c1, c2 = coords[order[-1]]

                end1 = wall[c1] - wall[1 - c1]
                end2 = new_wall[c2] - new_wall[1 - c2]
                exp_dist = np.linalg.norm(end1 - end2, ord=2)

                if farthest < exp_dist:
                    return True

        return False

    def add_walls(self, num=1, max_attempts=1000):
        """Add walls.

        Checks that walls are not too close to objects and that they do not
        overlap too much with one another.

        Does NOT check whether new wall creates a hole.

        Args:
            num (int, optional): number of walls to add. Defaults to 1.
            max_attempts (int, optional): maximum number of attempts to sample
                valid wall start and end coordinates. Defaults to 1000.

        Raises:
            ValueError: if could not sample valid wall start and end coordinates
                after max_attempts attempts.
        """

        if num > 0:
            warnings.warn(
                "add_walls() does not check whether a new wall will create a hole "
                "in the environment. Be sure to check environment visually.",
                category=EnvironmentWarning,
            )

        for _ in range(num):
            i = 0
            while True:
                start_coords = self.sample_coords()
                end_coords = self.sample_wall_end(start_coords)
                if end_coords is not None:
                    # check that wall ends are not too close to another
                    if self.check_if_walls_ends_too_close(
                        np.asarray([start_coords, end_coords])
                    ):
                        end_coords = None

                if end_coords is not None:
                    self.add_wall([start_coords, end_coords])
                    break
                if i > max_attempts:
                    raise ValueError(
                        "Could not sample valid wall start and end coordinates."
                    )
                i += 1

    def plot_environment(
        self, fig=None, ax=None, plot_objects=True, no_legend=False, autosave=None,
        **kwargs,
    ):
        """Plot the environment.

        Args:
            fig (matplotlib figure, optional): figure to plot on. Defaults to None.
            ax (matplotlib axis, optional): axis to plot on. Defaults to None.
            plot_objects (bool, optional): whether to plot objects. Defaults to True.
            no_legend (bool, optional): whether to remove legend. Defaults to False.
            autosave (bool, optional): whether to save the plot. Defaults to None.

        Returns:
            fig (matplotlib figure): figure with environment plotted.
            ax (matplotlib axis): axis with environment plotted.
        """

        if ax is None:
            env_width = self.extent[1] - self.extent[0]
            add_x = 0
            if plot_objects:
                add_x = 3 * env_width  # for legend and labels
            fig, ax = plt.subplots(
                figsize=(3 * env_width + add_x, 3 * (self.extent[3] - self.extent[2]))
            )

        fig, ax = super().plot_environment(
            fig=fig, ax=ax, autosave=False, plot_objects=False, **kwargs
        )

        if ax is None:
            raise RuntimeError("ax is None.")

        if plot_objects:
            type_num_to_plot_params_dict = copy.deepcopy(
                self.type_num_to_plot_params_dict
            )
            for coords, object_type in zip(
                self.objects["objects"], self.objects["object_types"]
            ):
                label = None
                if "name" in type_num_to_plot_params_dict[object_type].keys():
                    label = type_num_to_plot_params_dict[object_type].pop("name")
                ax.scatter(
                    *coords,
                    **type_num_to_plot_params_dict[object_type],
                    label=label,
                )

            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        legend = ax.get_legend()
        if no_legend and legend is not None:
            legend.remove()

        if fig is None:
            fig = ax.figure

        utils.save_figure(fig, "OpenField", save=autosave)

        return fig, ax


def get_angle_between_vectors(v1, v2, directional=False):
    """Get angle between two vectors.

    Args:
        v1 (1d array): vector 1.
        v2 (1d array): vector 2.
        directional (bool): whether to return the directional angle
            (i.e., first vector to second, with same start points: 0 to 360 degrees)
            or non-directional (i.e., between 0 and 90 degrees). Defaults to False.

    Returns:
        angle (float): angle between vectors.
    """

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    angle = np.rad2deg(np.dot(unit_v1, unit_v2)) % 360
    if not directional:
        angle = angle % 180
        angle = min(angle, 180 - angle)
    return angle


def shortest_distances_from_points_to_lines(positions, vectors):
    """Calculates the shortest distances between points and lines.

    Args:
        positions (2d array): positions of points,
            with shape points x coordinates (2)
            [(x1, y1), (x2, y2), ...]
        vectors (3d array): vectors defining lines,
            with shape vectors x coordinates (2)
            [[(x11, y11), (x12, y12)], [(x21, y21), (x22, y22)], ...]

    Returns:
        closest_distances (nd array): shortest distances between points and lines
            (points x vectors)
    """

    positions = np.asarray(positions)
    if len(positions.shape) == 1:  # expand if only one point is provided
        positions = np.expand_dims(positions, axis=0)

    vectors = np.asarray(vectors)
    if len(vectors.shape) == 2:  # expand if only one vector is provided
        vectors = np.expand_dims(vectors, axis=0)

    # returns points x vectors x coords
    shortest_vectors = utils.shortest_vectors_from_points_to_lines(positions, vectors)

    closest_distances = np.linalg.norm(shortest_vectors, ord=2, axis=-1)

    return closest_distances


if __name__ == "__main__":
    # Example usage
    params = {
        "init_random_reward_obj": 1,
        "init_random_novel_obj": 5,
        "init_random_walls": 5,
        "init_random_teleport_pairs": 2,
        "wall_lengths": [0.1, 0.2],
        "min_dist": 0.1,
        "init_seed": None,
    }
    env = OpenField(params)

    fig, ax = env.plot_environment()
    plt.show()