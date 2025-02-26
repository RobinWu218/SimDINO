from simdinov2.data.datasets import ImageNet

# the <ROOT> and <EXTRA> directories do not have to be distinct directories.
for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()