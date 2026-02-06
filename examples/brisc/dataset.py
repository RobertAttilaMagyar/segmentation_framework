from pathlib import Path

import ijson

from segworks.data_utils import ImageMaskPair, SegmentationDataset, register_dataset


@register_dataset("brisc")
class BRISCData(SegmentationDataset):
    def __init__(
        self,
        manifest_path: str | Path,
        transforms=None,
        filters: dict[str, str] | None = None,
    ):
        super().__init__(transforms)

        self.manifest_path = Path(manifest_path)
        if filters is None:
            filters = {}
        self.filters = filters
        self.obtain_pairs()

    def obtain_pairs(self):
        with self.manifest_path.open("rb") as f:
            for record in ijson.items(f, "item"):
                skip_record = False
                if record.get("task") != "segmentation":
                    continue
                if not record.get("is_mask"):
                    continue
                for k, v in self.filters.items():
                    if k in record and record[k] != v:
                        skip_record = True

                if skip_record:
                    continue

                mask_path = self.manifest_path.parent / record.get("relative_path")
                img_path = self.manifest_path.parent / record.get("linked_image")
                self.path_pairs.append(ImageMaskPair(img_path, mask_path))


def build_brisc_dataset(cfg, transforms=None) -> BRISCData:
    if "manifest_path" not in cfg:
        raise KeyError("'manifest_path' is missing from configuration")
    return BRISCData(cfg["manifest_path"], filters=cfg["filters"])


if __name__ == "__main__":
    from segworks.config_parsing import build_pipeline

    pipeline = build_pipeline(
        r"C:\Users\rober\Desktop\projects\lane_detection\examples\brisc\brisc_config.yml"
    )
    print(len(pipeline["training_dataset"]))
    print(len(pipeline["validation_dataset"]))

    pipeline["validation_dataset"].path_pairs[10].visualize()
