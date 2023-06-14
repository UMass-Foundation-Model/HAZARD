import numpy as np
from tdw.output_data import SegmentationColors
from tdw.output_data import ReplicantSegmentationColors
from typing import Union

class SegmentationID:
    """
    replicant and object ids should all be distinct
    """
    def __init__(self):
        self.segmentation_colors = dict()
        self.names = dict()
        self.categories = dict()
        self.reverse_id = dict()
        self.reverse_id[0] = 0
    
    def process(self, segm: Union[SegmentationColors, ReplicantSegmentationColors], id_renumbering: dict = dict()):
        if isinstance(segm, SegmentationColors):
            for j in range(segm.get_num()):
                idx = segm.get_object_id(j)
                self.segmentation_colors[idx] = np.array(segm.get_object_color(j))
                col = self.segmentation_colors[idx]
                # with open("utils/seg_id.txt", "a") as f:
                #     print("idx:", idx, "col:", col, "name:", segm.get_object_name(j).lower(), "category:", segm.get_object_category(j), file=f)
                #     f.close()
                col = col[0] + col[1] * 256 + col[2] * 256 * 256
                if len(id_renumbering) == 0:
                    self.reverse_id[col] = idx
                else:
                    self.reverse_id[col] = id_renumbering[idx] if idx in id_renumbering else 0
                self.names[idx] = segm.get_object_name(j).lower()
                self.categories[idx] = segm.get_object_category(j)
        elif isinstance(segm, ReplicantSegmentationColors):
            for j in range(segm.get_num()):
                idx = segm.get_id(j)
                self.segmentation_colors[idx] = np.array(segm.get_segmentation_color(j))
                col = self.segmentation_colors[idx]
                # with open("utils/seg_id.txt", "a") as f:
                #     print("idx:", idx, "col:", col, "name:", "replicant", "category:", "replicant", file=f)
                #     f.close()
                # print("col:", col)
                col = col[0] + col[1] * 256 + col[2] * 256 * 256
                # print("col:", col)
                if len(id_renumbering) == 0:
                    self.reverse_id[col] = idx
                else:
                    self.reverse_id[col] = id_renumbering[idx] if idx in id_renumbering else 0
                self.names[idx] = "replicant"
                self.categories[idx] = "replicant"
        else:
            raise Exception("Unknown type of segmentation colors")
    
    def get_seg_mask(self, idPass: np.ndarray, none_id: int = 0):
        idPass[idPass == None] = none_id
        compressed = idPass.astype(np.uint32)
        compressed = compressed[:, :, 0] + compressed[:, :, 1] * 256 + compressed[:, :, 2] * 256 * 256
        
        ret = np.vectorize(self.reverse_id.get)(compressed).astype(np.int64)
        # l = np.unique(ret)
        # l = list(l)
        # with open("utils/seg_id.txt", "a") as f:
        #     print("l:", l, file=f)
        #     f.close()
        # for v in l:
        #     if v == 0:
        #         continue
        #     if v not in self.segmentation_colors:
        #         print("BAD BAD BAD!")
        #         print("v:", v)
        #         print(self.reverse_id)
        #         while True:
        #             pass
        return ret