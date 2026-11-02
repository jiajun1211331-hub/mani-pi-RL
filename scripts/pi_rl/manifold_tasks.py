# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task identities and final-state outcome rules for LIBERO-10 manifold runs."""

TASKS = {
    1: {
        "name": "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "goals": [
            ("in", "cream_cheese_1", "basket_1_contain_region"),
            ("in", "butter_1", "basket_1_contain_region"),
        ],
        "classes": ["neither", "cheese_only", "butter_only", "complete"],
        "levels": [0, 1, 1, 2],
    },
    2: {
        "name": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "goals": [
            ("turnon", "flat_stove_1"),
            ("on", "moka_pot_1", "flat_stove_1_cook_region"),
        ],
        "classes": ["neither", "stove_only", "pot_only", "complete"],
        "levels": [0, 1, 1, 2],
    },
    3: {
        "name": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "goals": [
            ("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
            ("close", "white_cabinet_1_bottom_region"),
        ],
        "classes": [
            "not_inserted_open",
            "inserted_not_closed",
            "empty_closed",
            "complete",
        ],
        "levels": [0, 1, 0, 2],
    },
    9: {
        "name": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "goals": [
            ("in", "white_yellow_mug_1", "microwave_1_heating_region"),
            ("close", "microwave_1"),
        ],
        "classes": [
            "not_inserted_open",
            "inserted_not_closed",
            "empty_closed",
            "complete",
        ],
        "levels": [0, 1, 0, 2],
    },
}


def goal_values(state, task_id):
    """Read the two target predicates by identity, independently of BDDL order."""
    values = {
        tuple(g): v for g, v in zip(state["goals"], state["predicates"], strict=True)
    }
    expected = TASKS[task_id]["goals"]
    if set(values) != set(expected):
        raise ValueError(
            f"Expected task{task_id} goals {expected}, found {list(values)}"
        )
    return tuple(bool(values[g]) for g in expected)


def outcome(task_id, values, success):
    """Classify retained terminal achievements; success conflicts stay disputed."""
    if bool(success) != all(values):
        return "disputed", "disputed"
    index = int(values[0]) + 2 * int(values[1])
    spec = TASKS[task_id]
    return spec["classes"][index], f"level_{spec['levels'][index]}"
