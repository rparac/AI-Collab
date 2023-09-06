from .wrappers import LabelingFunctionWrapper


# TODO: stopped here
class AICollabLabellingFunctionWrapper(LabelingFunctionWrapper):
    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        labels = []

        unwrapped_obs = obs['A']
        agent_id = unwrapped_obs["agent_id"]

        if unwrapped_obs["nearby_obj_weight"] > 0 and unwrapped_obs["nearby_obj_danger"] == 1:
            labels.append('a1d')

        if any(obj_info["carried_by"][agent_id] for obj_info in unwrapped_obs["object_infos"]) > 0:
            labels.append('a1l')

        pos = unwrapped_obs["agent_infos"][agent_id]["pos"]
        if pos in self.goal_coords:
            labels.append('a1z')

        return labels
