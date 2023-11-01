import json


class PipelineDescriptor:

    def __init__(self,
                 descriptor_folder: str,
                 descriptor_name: str
                 ):
        self.descriptor_path = f"{descriptor_folder}{descriptor_name}"

    def open(self) -> dict:
        with open(self.descriptor_path) as file_pointer:
            descriptor: dict = json.load(file_pointer)

        return descriptor

