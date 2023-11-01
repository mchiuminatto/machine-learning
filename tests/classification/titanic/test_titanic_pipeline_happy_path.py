from classification.titanic.titanic_pipeline import TitanicPipeline
from pipelines import constants


class TestTitanic:

    def test_open_descriptor(self):
        descriptor_folder = constants.ROOT_FOLDER+"/tests/classification/titanic/input/"
        descriptor_file: str = "titanic.json"
        titanic_pipeline = TitanicPipeline(descriptor_folder=descriptor_folder, descriptor_file=descriptor_file)
        assert titanic_pipeline.descriptor["datasets"]["train"]["storage"]["storage-type"] == "file-system"

    def test_pre_process_data(self):
        descriptor_folder = constants.ROOT_FOLDER+"/tests/classification/titanic/input/"
        descriptor_file: str = "titanic.json"
        titanic_pipeline = TitanicPipeline(descriptor_folder=descriptor_folder, descriptor_file=descriptor_file)
        titanic_pipeline.pre_process_data()

        assert len(titanic_pipeline._train_dataset) == 891
        assert len(titanic_pipeline._test_dataset) == 418
        assert (len(titanic_pipeline._features)) == 891
        assert len(titanic_pipeline._test_dataset) == 418
        assert titanic_pipeline._features.isna().sum().sum() == 0
        assert titanic_pipeline._test_dataset.isna().sum().sum() == 0

    def test_train_estimator(self):
        descriptor_folder = constants.ROOT_FOLDER+"/tests/classification/titanic/input/"
        descriptor_file: str = "titanic.json"
        titanic_pipeline = TitanicPipeline(descriptor_folder=descriptor_folder, descriptor_file=descriptor_file)
        titanic_pipeline.pre_process_data()
        titanic_pipeline.split_train_test()
        grid_search = titanic_pipeline.train_estimator("lgr")
        breakpoint()






