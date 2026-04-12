from . import attnmaps, daam, heads, matrix, util

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for m in (util, matrix, heads, attnmaps, daam):
    NODE_CLASS_MAPPINGS.update(m.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(m.NODE_DISPLAY_NAME_MAPPINGS)
